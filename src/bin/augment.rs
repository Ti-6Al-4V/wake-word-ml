// Аугментация позитивов: из каждого исходного окна делает 8 вариантов.
// Цель — научить модель узнавать слово независимо от громкости, темпа,
// уровня шума и положения в окне.
//
// Варианты (v1-v8):
//   v1, v2 — громкость ×0.7 и ×1.3
//   v3, v4 — «ленточная скорость» ×0.9 и ×1.1 (меняет и темп, и высоту:
//             медленнее = ниже голос, быстрее = выше; простейшая и очень
//             эффективная аугментация для коротких слов)
//   v5, v6, v7 — белый шум с SNR 20, 10 и 5 дБ (5 дБ — жёсткий, по 07-quality)
//   v8 — сдвиг слова по времени внутри окна
//
// Детерминизм: сид ГПСЧ = хэш имени файла + номер варианта, поэтому
// повторный запуск даёт те же результаты и идемпотентен (готовые файлы
// пропускаются).
//
// ВАЖНО: исходниками служат только файлы БЕЗ "_v" в имени — иначе
// повторный запуск начал бы аугментировать уже аугментированное.
//
// Запуск: cargo run --bin augment -- <вход> <выход>
// Пример: cargo run --bin augment -- dataset/positive dataset/positive

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SNRS_DB: [f32; 3] = [20.0, 10.0, 5.0]; // для v5, v6, v7

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Использование: augment <вход> <выход>");
        std::process::exit(1);
    }
    let in_dir = &args[1];
    let out_dir = &args[2];
    std::fs::create_dir_all(out_dir).expect("не создать выходную папку");

    // Собираем исходники: только .wav и только не-аугментированные.
    let mut files: Vec<PathBuf> = std::fs::read_dir(in_dir)
        .unwrap_or_else(|e| panic!("не открыть {in_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let is_wav = p.extension().map_or(false, |x| x == "wav");
            let is_source = p.file_stem()
                .and_then(|s| s.to_str())
                .map_or(true, |s| !s.contains("_v"));
            is_wav && is_source
        })
        .collect();
    files.sort();

    let (mut done, mut skipped) = (0usize, 0usize);
    for path in &files {
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();

        // Читаем исходное окно (уже 16kHz моно после preprocess).
        // Битый/чужой файл не роняет весь прогон — пропускаем с предупреждением.
        let mut reader = match hound::WavReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[пропуск, не читается] {}: {e}", path.display());
                continue;
            }
        };
        let rate = reader.spec().sample_rate;
        let samples: Vec<f32> = reader.samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect();

        // 8 вариантов: v1..v8
        for v in 1..=8usize {
            let out_path = Path::new(out_dir).join(format!("{stem}_v{v}.wav"));
            if out_path.exists() {
                skipped += 1;
                continue;
            }

            // Свой детерминированный ГПСЧ на каждый вариант:
            // хэш имени + номер варианта. ChaCha8 — быстрый и воспроизводимый.
            let mut hasher = DefaultHasher::new();
            (&stem, v).hash(&mut hasher);
            let mut rng = ChaCha8Rng::seed_from_u64(hasher.finish());

            let aug: Vec<f32> = match v {
                1 => volume(&samples, 0.7),
                2 => volume(&samples, 1.3),
                3 => tape_speed(&samples, 0.9), // медленнее, ниже
                4 => tape_speed(&samples, 1.1), // быстрее, выше
                5 => add_noise(&samples, SNRS_DB[0], &mut rng),
                6 => add_noise(&samples, SNRS_DB[1], &mut rng),
                7 => add_noise(&samples, SNRS_DB[2], &mut rng),
                _ => time_shift(&samples, &mut rng),
            };

            // Возврат к длине исходного окна (растяжка меняет длину).
            let window = fit_to_len(&aug, samples.len());
            write_wav(&out_path, &window, rate);
            done += 1;
        }
        println!("[{done}/8] {stem} → 8 вариантов");
    }
    println!("\nГотово: создано {done}, уже было {skipped}. Выход: {out_dir}");
}

// Громкость: просто умножить. 1.3 на нормализованных до 0.9 файлах
// даёт пик ~1.17 — clamp в write_wav срежет, это осознанный лёгкий перегрев.
fn volume(samples: &[f32], gain: f32) -> Vec<f32> {
    samples.iter().map(|s| s * gain).collect()
}

// «Ленточная скорость»: линейная интерполяция при чтении с шагом factor.
// factor > 1 — читаем быстрее: слово короче и выше. factor < 1 — наоборот.
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

// Белый шум с заданным SNR относительно энергии сигнала.
// SNR_dB = 20*log10(rms_signal / rms_noise) →
// rms_noise = rms_signal / 10^(SNR/20).
// У равномерного шума в [-A, A] среднеквадратичное = A/sqrt(3).
fn add_noise(samples: &[f32], snr_db: f32, rng: &mut impl Rng) -> Vec<f32> {
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let noise_rms = rms * 10.0_f32.powf(-snr_db / 20.0);
    let amp = noise_rms * 3.0_f32.sqrt();
    samples.iter()
        .map(|s| s + rng.random_range(-amp..amp))
        .collect()
}

// Циклический сдвиг на случайную величину до ±10% окна.
// Слово в центре, края — тишина, так что перенос через границу безвреден.
fn time_shift(samples: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    let n = samples.len();
    let max_shift = n / 10;
    let shift = rng.random_range(0..(2 * max_shift + 1)) as isize - max_shift as isize;
    // rem_euclid — остаток без отрицательных значений (в отличие от %).
    (0..n)
        .map(|i| samples[(i as isize + shift).rem_euclid(n as isize) as usize])
        .collect()
}

// Довести до нужной длины: обрезать или дополнить тишиной.
fn fit_to_len(samples: &[f32], len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    let n = samples.len().min(len);
    out[..n].copy_from_slice(&samples[..n]);
    out
}

// Запись WAV 16-bit моно (та же, что в record.rs).
fn write_wav(path: &Path, samples: &[f32], rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}
