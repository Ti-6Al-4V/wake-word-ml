//! Загрузка датасета и разбиение на train/val/test (70/15/15).
//!
//! Главное правило: **все аугментации одного исходника попадают в один
//! сплит**. Если дубль germes_real_0020 и его копия с шумом
//! germes_real_0020_v7 окажутся в разных сплитах, модель «узнает»
//! копию в тесте не потому что обобщила, а потому что уже видела
//! оригинал на тренировке. Метрики будут врать (утечка данных).
//!
//! Детерминизм: сид 42 → разбиение одинаковое при каждом запуске,
//! эксперименты воспроизводимы.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

pub const SEED: u64 = 42;

#[derive(Clone, Debug)]
pub struct Sample {
    pub path: PathBuf,
    pub label: f32, // 1.0 = слово («Гермес»), 0.0 = не слово
}

pub struct Splits {
    pub train: Vec<Sample>,
    pub val: Vec<Sample>,
    pub test: Vec<Sample>,
}

/// Ключ группировки: имя файла без суффикса аугментации `_vN`.
/// germes_real_0020_v7 → «germes_real_0020», и сам germes_real_0020
/// даёт тот же ключ — значит они неразлучны при разбиении.
/// У негативов суффикса нет: каждый файл — сам себе группа.
pub fn group_key(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    match stem.rfind("_v") {
        Some(i) => stem[..i].to_string(),
        None => stem.to_string(),
    }
}

/// Фишер–Йетс: честное перемешивание массива данным ГПСЧ.
/// (Идём с конца: для каждой позиции i тянем случайного соседа
/// слева включительно и меняем местами.)
fn shuffle<T>(v: &mut [T], rng: &mut ChaCha8Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

/// Разбивает один класс (позитивы или негативы) по группам.
/// 70% групп → train, 15% → val, остаток → test.
fn split_class(files: &mut [PathBuf], label: f32, rng: &mut ChaCha8Rng) -> (Vec<Sample>, Vec<Sample>, Vec<Sample>) {
    // Собираем группы: ключ → файлы. Vec сохраняет порядок вставки.
    let mut keys: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
    for f in files.iter() {
        let k = group_key(f);
        if !groups.contains_key(&k) {
            keys.push(k.clone());
        }
        groups.entry(k).or_default().push(f.clone());
    }

    // Детерминированно перемешиваем ПОРЯДОК ГРУПП, режем по долям.
    keys.sort(); // сначала сортировка: перемешивание из одного состояния
    shuffle(&mut keys, rng);

    let n = keys.len();
    let n_train = n * 70 / 100;
    let n_val = n * 15 / 100;
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        let bucket = match i {
            i if i < n_train => &mut train,
            i if i < n_train + n_val => &mut val,
            _ => &mut test,
        };
        for f in groups.remove(k).unwrap() {
            bucket.push(Sample { path: f, label });
        }
    }
    (train, val, test)
}

/// Сканирует positive/ и negative/, возвращает три сплита.
pub fn load(pos_dir: &str, neg_dir: &str) -> Splits {
    let collect = |dir: &str| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("не открыть {dir}: {e}"))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "wav"))
            .collect();
        v.sort();
        v
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut pos = collect(pos_dir);
    let mut neg = collect(neg_dir);
    let (pt, pv, pte) = split_class(&mut pos, 1.0, &mut rng);
    let (nt, nv, nte) = split_class(&mut neg, 0.0, &mut rng);

    let concat = |mut a: Vec<Sample>, mut b: Vec<Sample>| {
        a.append(&mut b);
        a
    };
    Splits {
        train: concat(pt, nt),
        val: concat(pv, nv),
        test: concat(pte, nte),
    }
}

/// WAV → окно сэмплов → MFCC-матрица [40][20] (см. src/mfcc.rs).
/// Этим train превращает каждый файл в тензор.
pub fn features(path: &Path) -> Vec<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).expect("wav не читается");
    assert_eq!(reader.spec().sample_rate, 16_000, "датасет должен быть 16kHz (make preprocess)");
    let samples: Vec<f32> = reader.samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();
    crate::mfcc::wav_to_mfcc(&samples)
}
