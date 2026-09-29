// Аугментация: из каждого позитива делает 8 вариантов, из негатива — 2.
// Цель — научить модель узнавать слово независимо от фона, темпа,
// уровня шума и положения в окне.
//
// Варианты (v1-v8):
//   v1, v2     — РЕАЛЬНЫЙ фон из dataset/raw/background (вода, посуда,
//                велотренажёр, розовый шум...) с SNR 15 и 5 дБ. Кусок фона
//                берётся со случайного места случайного файла.
//   v3, v4     — «ленточная скорость» ×0.9 и ×1.1 (меняет и темп, и высоту:
//                медленнее = ниже голос, быстрее = выше; простейшая и очень
//                эффективная аугментация для коротких слов)
//   v5, v6, v7 — белый шум с SNR 20, 10 и 5 дБ (5 дБ — жёсткий, по 07-quality)
//   v8         — слово переставлено в СЛУЧАЙНОЕ место окна (не ±10%, а
//                куда угодно, лишь бы влезло целиком). В стриминге слово
//                оказывается в любой позиции — модель должна это видеть.
//
// Негативы (флаг --negatives): по NEG_VARIANTS разных варианта из v1–v7
// на исходник. Если шум, фоны и скорость видит только класс «Гермес»,
// модели проще выучить «зашумлено = Гермес», чем само слово. v8 негативам
// не нужен: слова, которое надо двигать, в них нет.
//
// Почему нет вариантов «громкость ×0.7 / ×1.3» (были v1/v2 раньше):
// после log-mel и нормализации по коэффициентам масштаб амплитуды
// исчезает полностью — такие копии в пространстве признаков совпадают
// с оригиналом. Это была иллюзия ×9 при реальных ×7.
//
// Детерминизм: сид ГПСЧ = хэш имени файла + номер варианта, поэтому
// повторный запуск даёт те же результаты и идемпотентен (готовые файлы
// пропускаются). Хвосты после растяжки/сдвига заполняются дизером.
//
// ВАЖНО: исходниками служат только файлы БЕЗ "_v<цифры>" в имени — иначе
// повторный запуск начал бы аугментировать уже аугментированное.
//
// Запуск: cargo run --bin augment -- <вход> <выход> [папка фонов] [--negatives]
// Пример: cargo run --bin augment -- dataset/positive dataset/positive
//         cargo run --bin augment -- dataset/negative dataset/negative --negatives

use std::path::{Path, PathBuf};

use rand::{Rng, RngExt};
use wake_word_ml::audio;

const REAL_SNRS_DB: [f32; 2] = [15.0, 5.0];         // v1, v2
const WHITE_SNRS_DB: [f32; 3] = [20.0, 10.0, 5.0];  // v5, v6, v7
const DEFAULT_BG_DIR: &str = "dataset/raw/background";
/// Сколько вариантов делать на один негатив.
const NEG_VARIANTS: usize = 2;

fn main() {
    let negatives = std::env::args().any(|a| a == "--negatives");
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    if args.len() < 3 {
        eprintln!("Использование: augment <вход> <выход> [папка фонов] [--negatives]");
        std::process::exit(1);
    }
    let in_dir = &args[1];
    let out_dir = &args[2];
    let bg_dir = args.get(3).map(String::as_str).unwrap_or(DEFAULT_BG_DIR);
    std::fs::create_dir_all(out_dir).expect("не создать выходную папку");

    // Фоны: длинные 16kHz-записи. Грузим целиком в память (их 6, по ~1 мин).
    let backgrounds = load_backgrounds(bg_dir);
    if backgrounds.is_empty() {
        eprintln!("В {bg_dir} нет фонов — v1/v2 будут белым шумом. Нужен make negatives (копирует _background_noise_).");
    } else {
        println!("Фонов загружено: {} из {bg_dir}", backgrounds.len());
    }

    // Собираем исходники: только .wav и только не-аугментированные.
    let mut files: Vec<PathBuf> = std::fs::read_dir(in_dir)
        .unwrap_or_else(|e| panic!("не открыть {in_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let is_wav = p.extension().map_or(false, |x| x == "wav");
            let is_source = p.file_stem()
                .and_then(|s| s.to_str())
                .map_or(true, |s| !is_augmented(s));
            is_wav && is_source
        })
        .collect();
    files.sort();

    let (mut done, mut skipped) = (0usize, 0usize);
    for path in &files {
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();

        // Читаем исходное окно (уже 16kHz моно после preprocess).
        // Битый/чужой файл не роняет весь прогон — пропускаем с предупреждением.
        let (samples, rate) = match audio::read_wav(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[пропуск, не читается] {}: {e}", path.display());
                continue;
            }
        };

        let variants = variants_for(&stem, negatives);
        for &v in &variants {
            let out_path = Path::new(out_dir).join(format!("{stem}_v{v}.wav"));
            if out_path.exists() {
                skipped += 1;
                continue;
            }

            // Свой детерминированный ГПСЧ на каждый вариант.
            let mut rng = audio::rng_for(&stem, v as u64);

            let aug: Vec<f32> = match v {
                1 | 2 => {
                    let snr = REAL_SNRS_DB[v - 1];
                    if backgrounds.is_empty() {
                        add_white_noise(&samples, snr, &mut rng)
                    } else {
                        add_background(&samples, &backgrounds, snr, &mut rng)
                    }
                }
                3 => tape_speed(&samples, 0.9), // медленнее, ниже
                4 => tape_speed(&samples, 1.1), // быстрее, выше
                5 | 6 | 7 => add_white_noise(&samples, WHITE_SNRS_DB[v - 5], &mut rng),
                _ => random_position(&samples, rate, &mut rng),
            };

            // Возврат к длине исходного окна (растяжка меняет длину);
            // недостающее — дизер.
            let window = audio::fit_to_len(&aug, samples.len(), 0, &mut rng);
            audio::write_wav(&out_path, &window, rate);
            done += 1;
        }
        println!("[{done}] {stem} → варианты {variants:?}");
    }
    println!("\nГотово: создано {done}, уже было {skipped}. Выход: {out_dir}");
}

/// Номера вариантов для исходника: позитивам все v1–v8, негативу —
/// NEG_VARIANTS разных из v1–v7, выбранных детерминированно по имени.
fn variants_for(stem: &str, negatives: bool) -> Vec<usize> {
    if !negatives {
        return (1..=8).collect();
    }
    let mut pool: Vec<usize> = (1..=7).collect();
    let mut rng = audio::rng_for(stem, 0);
    for i in 0..NEG_VARIANTS {
        let j = rng.random_range(i..pool.len());
        pool.swap(i, j);
    }
    pool.truncate(NEG_VARIANTS);
    pool
}

/// «_v» + цифры в конце имени = аугментированный файл.
fn is_augmented(stem: &str) -> bool {
    match stem.rfind("_v") {
        Some(i) => {
            let tail = &stem[i + 2..];
            !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}

fn load_backgrounds(dir: &str) -> Vec<Vec<f32>> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "wav"))
        .collect();
    paths.sort();
    paths.iter()
        .filter_map(|p| match audio::read_wav(p) {
            Ok((s, 16_000)) if s.len() > 16_000 => Some(s),
            Ok((_, rate)) => { eprintln!("[фон пропущен: {rate}Hz, нужен 16kHz] {}", p.display()); None }
            Err(e) => { eprintln!("[фон не читается] {}: {e}", p.display()); None }
        })
        .collect()
}

/// «Ленточная скорость»: линейная интерполяция при чтении с шагом factor.
/// factor > 1 — читаем быстрее: слово короче и выше. factor < 1 — наоборот.
fn tape_speed(samples: &[f32], factor: f32) -> Vec<f32> {
    let out_len = (samples.len() as f32 / factor) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f32 * factor;          // позиция в исходном массиве
        let i0 = pos as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let frac = pos - i0 as f32;           // доля для линейной интерполяции
        out.push(samples[i0] * (1.0 - frac) + samples[i1] * frac);
    }
    out
}

/// Уровень шума под заданный SNR относительно энергии сигнала:
/// SNR_dB = 20·log10(rms_signal / rms_noise) → rms_noise = rms_signal / 10^(SNR/20).
fn noise_rms_for(samples: &[f32], snr_db: f32) -> f32 {
    audio::rms(samples) * 10.0_f32.powf(-snr_db / 20.0)
}

/// Белый шум. У равномерного шума в [−A, A] среднеквадратичное = A/√3.
fn add_white_noise(samples: &[f32], snr_db: f32, rng: &mut impl Rng) -> Vec<f32> {
    let amp = noise_rms_for(samples, snr_db) * 3.0_f32.sqrt();
    samples.iter().map(|s| s + rng.random_range(-amp..amp)).collect()
}

/// Реальный фон: случайный файл, случайное смещение, отмасштабирован
/// под нужный SNR по его собственной RMS.
fn add_background(samples: &[f32], bgs: &[Vec<f32>], snr_db: f32, rng: &mut impl Rng) -> Vec<f32> {
    let bg = &bgs[rng.random_range(0..bgs.len())];
    let start = rng.random_range(0..bg.len() - samples.len());
    let piece = &bg[start..start + samples.len()];
    let target = noise_rms_for(samples, snr_db);
    let gain = target / audio::rms(piece).max(1e-6);
    samples.iter().zip(piece).map(|(s, n)| s + n * gain).collect()
}

/// Слово — в случайное место окна. Границы слова находим по энергии,
/// остальное окно заполняем дизером. Так модель видит слово и в начале,
/// и в середине, и в конце — как в стриминге на ESP32.
fn random_position(samples: &[f32], rate: u32, rng: &mut impl Rng) -> Vec<f32> {
    let (start, end) = audio::speech_bounds(samples, rate, 0.1);
    // Небольшой запас вокруг границ, чтобы не резать тихие края слова.
    let pad = (rate / 20) as usize; // 50мс
    let start = start.saturating_sub(pad);
    let end = (end + pad).min(samples.len());
    let word = &samples[start..end];
    let max_offset = samples.len() - word.len();
    let offset = rng.random_range(0..=max_offset);
    audio::fit_to_len(word, samples.len(), offset, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positives_get_all_eight_variants() {
        assert_eq!(variants_for("germes_real_0001", false), (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn negatives_get_distinct_deterministic_variants_without_shift() {
        for stem in ["golos_p00042_c03", "negative_sc_00017", "self_s01_c0007"] {
            let v = variants_for(stem, true);
            assert_eq!(v, variants_for(stem, true));
            assert_eq!(v.len(), NEG_VARIANTS);
            assert!(v.iter().all(|x| (1..=7).contains(x)));
            assert_ne!(v[0], v[1]);
        }
    }
}
