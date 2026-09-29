//! Загрузка датасета и разбиение на train/val/test.
//!
//! Главное правило: **все файлы одной группы попадают в один сплит**.
//! Группа — диктор (`p017__...` → `p017`) или исходник со всеми его
//! аугментациями. Иначе модель «узнаёт» в val знакомого человека или
//! копию дубля, а не слово, и метрики врут (утечка данных).
//!
//! Test — отдельные папки `dataset/holdout/{positive,negative}`: живые люди,
//! которых нет в train/val, без аугментаций. Тогда positive/negative
//! делятся 85/15 на train/val. Без holdout — старый режим 70/15/15 с
//! предупреждением: такой test содержит синтетику и тех же дикторов.
//!
//! Сид 42 повторяет split при неизменном наборе файлов. Добавление
//! файлов может переместить старые группы; ограничения — docs/02-dataset.md.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

pub const SEED: u64 = 42;

/// Честный test: живые дикторы, которых нет в positive/negative.
pub const HOLDOUT_POS: &str = "dataset/holdout/positive";
pub const HOLDOUT_NEG: &str = "dataset/holdout/negative";

#[derive(Clone, Debug)]
pub struct Sample {
    pub path: PathBuf,
    pub label: f32, // 1.0 = слово («Гермес»), 0.0 = не слово
}

pub struct Splits {
    pub train: Vec<Sample>,
    pub val: Vec<Sample>,
    pub test: Vec<Sample>,
    /// true — test взят из holdout-папок, false — отрезан от основных.
    pub holdout: bool,
}

/// Ключ группировки: файлы одного «источника правды» неразлучны при
/// разбиении. Правила (по имени файла, без расширения):
///
/// - `qwen_cv0042__germes_03_v4` → `qwen_cv0042` (всё до `__` — явный
///   ключ диктора; основной формат для новых данных)
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
    // 0. Явный ключ диктора имеет приоритет над эвристиками ниже.
    if let Some((speaker, _)) = stem.split_once("__") {
        return speaker.to_string();
    }
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

/// Разбивает один класс (позитивы или негативы) по группам:
/// `train_pct`% групп → train, `val_pct`% → val, остаток → test.
fn split_class(files: &mut [PathBuf], label: f32, rng: &mut ChaCha8Rng, train_pct: usize, val_pct: usize) -> (Vec<Sample>, Vec<Sample>, Vec<Sample>) {
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
    let n_train = n * train_pct / 100;
    // Доли на 100% в сумме: остаток от округления уходит в val, а не в test.
    let n_val = if train_pct + val_pct == 100 { n - n_train } else { n * val_pct / 100 };
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

fn wavs(dir: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("не открыть {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "wav"))
        .collect();
    v.sort();
    v
}

/// Сканирует positive/ и negative/ (+ holdout, если он есть), возвращает три сплита.
pub fn load(pos_dir: &str, neg_dir: &str) -> Splits {
    let holdout = [HOLDOUT_POS, HOLDOUT_NEG].iter().all(|d| Path::new(d).is_dir());
    let splits = load_with(pos_dir, neg_dir, holdout.then_some((HOLDOUT_POS, HOLDOUT_NEG)));
    if !splits.holdout {
        eprintln!("ВНИМАНИЕ: нет {HOLDOUT_POS} и {HOLDOUT_NEG} — test отрезан от основного набора \
                   (синтетика, те же дикторы). Для честной цифры собери holdout (make holdout).");
    }
    splits
}

pub fn load_with(pos_dir: &str, neg_dir: &str, holdout: Option<(&str, &str)>) -> Splits {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let (train_pct, val_pct) = if holdout.is_some() { (85, 15) } else { (70, 15) };

    let (pt, pv, mut pte) = split_class(&mut wavs(pos_dir), 1.0, &mut rng, train_pct, val_pct);
    let (nt, nv, mut nte) = split_class(&mut wavs(neg_dir), 0.0, &mut rng, train_pct, val_pct);
    if let Some((hpos, hneg)) = holdout {
        let as_samples = |dir: &str, label: f32| -> Vec<Sample> {
            wavs(dir).into_iter().map(|path| Sample { path, label }).collect()
        };
        assert!(pte.is_empty() && nte.is_empty());
        pte = as_samples(hpos, 1.0);
        nte = as_samples(hneg, 0.0);
    }

    let concat = |mut a: Vec<Sample>, mut b: Vec<Sample>| {
        a.append(&mut b);
        a
    };
    Splits {
        train: concat(pt, nt),
        val: concat(pv, nv),
        test: concat(pte, nte),
        holdout: holdout.is_some(),
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
    use super::{group_key, load_with, split_class, SEED};
    use rand::SeedableRng;
    use std::path::Path;

    #[test]
    fn explicit_speaker_prefix_groups_everything_before_double_underscore() {
        let k = |s: &str| group_key(Path::new(&format!("dataset/x/{s}.wav")));
        assert_eq!(k("qwen_cv0042__germes_03"), "qwen_cv0042");
        assert_eq!(k("qwen_cv0042__termes_01_v4"), "qwen_cv0042");
        assert_eq!(k("p017__germes_0005"), "p017");
    }

    #[test]
    fn with_holdout_nothing_from_main_dirs_goes_to_test() {
        let root = std::env::temp_dir().join(format!("ww-split-{}", std::process::id()));
        let dirs = ["pos", "neg", "hpos", "hneg"].map(|d| root.join(d));
        for d in &dirs {
            std::fs::create_dir_all(d).unwrap();
        }
        for i in 0..40 {
            std::fs::write(dirs[0].join(format!("p{i:03}__germes_1.wav")), b"").unwrap();
            std::fs::write(dirs[0].join(format!("p{i:03}__germes_2.wav")), b"").unwrap();
            std::fs::write(dirs[1].join(format!("neg_{i:03}.wav")), b"").unwrap();
        }
        std::fs::write(dirs[2].join("h001__germes_1.wav"), b"").unwrap();
        std::fs::write(dirs[3].join("h001__speech_c0001.wav"), b"").unwrap();
        let s = |d: &std::path::PathBuf| d.to_str().unwrap().to_string();
        let splits = load_with(&s(&dirs[0]), &s(&dirs[1]), Some((&s(&dirs[2]), &s(&dirs[3]))));
        std::fs::remove_dir_all(&root).unwrap();

        assert!(splits.holdout);
        assert_eq!(splits.test.len(), 2);
        assert!(splits.test.iter().all(|x| x.path.to_str().unwrap().contains("/h")));
        assert_eq!(splits.train.len() + splits.val.len(), 120);
        // 40 групп позитивов → 34 train / 6 val; дубли одного диктора вместе.
        let pos_val: Vec<_> = splits.val.iter().filter(|x| x.label > 0.5).collect();
        assert_eq!(pos_val.len(), 12);
        for x in &pos_val {
            let key = group_key(&x.path);
            assert!(splits.train.iter().all(|t| group_key(&t.path) != key));
        }
    }

    #[test]
    fn full_train_val_split_leaves_no_rounding_remainder_for_test() {
        let mut files: Vec<std::path::PathBuf> =
            (0..7).map(|i| format!("p{i}__germes.wav").into()).collect();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(SEED);
        let (train, val, test) = split_class(&mut files, 1.0, &mut rng, 85, 15);
        assert_eq!((train.len(), val.len(), test.len()), (5, 2, 0));
    }

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
