//! Загрузка датасета и разбиение на train/val/test (70/15/15).
//!
//! Главное правило: **все аугментации одного исходника попадают в один
//! сплит**. Если дубль germes_real_0020 и его копия с шумом
//! germes_real_0020_v7 окажутся в разных сплитах, модель «узнает»
//! копию в тесте не потому что обобщила, а потому что уже видела
//! оригинал на тренировке. Метрики будут врать (утечка данных).
//!
//! Сид 42 повторяет split при неизменном наборе файлов. Добавление
//! файлов может переместить старые группы. Сессии/дикторы и общие фоны
//! не проверяются автоматически; ограничения — docs/02-dataset.md.

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

/// Ключ группировки: файлы одного «источника правды» неразлучны при
/// разбиении. Правила (по имени файла, без расширения):
///
/// - `germes_real_0020_v7`  → `germes_real_0020`  (аугментация _vN)
/// - `golos_p00042_c03`     → `golos_p00042`      (нарезка фразы _cN:
///   все куски одной фразы — один диктор, одна сессия)
/// - `self_s01_c0007`       → `self_s01`          (нарезка своей речи)
/// - `germes_t0_Dmitry_plus25` → `germes_t0_Dmitry` (5 скоростей одного
///   TTS-голоса — почти один и тот же файл, держим вместе)
/// - остальное              → сам себе группа
///
/// Суффикс снимается только если после `_v`/`_c` идут ЦИФРЫ: имя вроде
/// `germes_voice` не пострадает.
pub fn group_key(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    // 1. Снять суффикс аугментации/нарезки, если он есть.
    let mut base = stem;
    for tag in ["_v", "_c"] {
        if let Some(i) = base.rfind(tag) {
            let tail = &base[i + tag.len()..];
            if !tail.is_empty() && tail.chars().all(|ch| ch.is_ascii_digit()) {
                base = &base[..i];
                break;
            }
        }
    }
    // 2. TTS: germes_t<N>_<голос>_<скорость> → без скорости.
    if base.starts_with("germes_t") {
        if let Some(i) = base.rfind('_') {
            return base[..i].to_string();
        }
    }
    base.to_string()
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
    let (samples, rate) = crate::audio::read_wav(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(rate, 16_000, "датасет должен быть 16kHz (make preprocess): {}", path.display());
    crate::mfcc::wav_to_mfcc(&samples)
}

/// Плоский вектор признаков (40·20 = 800 чисел) — форма, в которой
/// батч уходит в тензор.
pub fn features_flat(path: &Path) -> Vec<f32> {
    features(path).into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::group_key;
    use std::path::Path;

    #[test]
    fn group_key_rules() {
        let k = |s: &str| group_key(Path::new(&format!("dataset/x/{s}.wav")));
        assert_eq!(k("germes_real_0020_v7"), "germes_real_0020");
        assert_eq!(k("germes_real_0020"), "germes_real_0020");
        assert_eq!(k("golos_p00042_c03"), "golos_p00042");
        assert_eq!(k("self_s01_c0007"), "self_s01");
        assert_eq!(k("germes_t0_DmitryNeural_plus25"), "germes_t0_DmitryNeural");
        assert_eq!(k("germes_t0_DmitryNeural_norm_v3"), "germes_t0_DmitryNeural");
        assert_eq!(k("negative_sc_00017"), "negative_sc_00017");
        assert_eq!(k("conf_termes_0003"), "conf_termes_0003");
        assert_eq!(k("germes_voice"), "germes_voice");
    }
}
