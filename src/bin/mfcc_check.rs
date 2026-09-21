// Проверка MFCC на живом файле: печатает статистику матрицы и
// PCM RMS и норму MFCC по кадрам — это разные величины.
//
// Запуск: cargo run --bin mfcc_check -- [путь к wav]
// Без пути берёт первый файл из dataset/positive.

use wake_word_ml::mfcc;

fn main() {
    // args(): [бинарь, arg1, ...] — путь это первый аргумент, nth(1)
    let path = std::env::args().nth(1).unwrap_or_else(first_positive);
    println!("Файл: {path}");

    let (samples, rate) = wake_word_ml::audio::read_wav(std::path::Path::new(&path))
        .expect("нужен корректный WAV PCM16");
    println!("Сэмплов: {} ({:.2}с при {rate}Hz)", samples.len(), samples.len() as f32 / rate as f32);
    assert_eq!(rate as usize, mfcc::SAMPLE_RATE, "сначала приведи файл к 16kHz");
    assert_eq!(samples.len(), mfcc::WINDOW_SAMPLES, "нужно ровно 1.2с (make preprocess)");

    let m = mfcc::wav_to_mfcc(&samples);
    println!("Матрица MFCC: {} кадров × {} коэффициентов (ожидаем {}×{})",
        m.len(), m[0].len(), mfcc::NUM_FRAMES, mfcc::N_MFCC);

    let flat: Vec<f32> = m.iter().flatten().cloned().collect();
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
    let std: f32 = (flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
    let (lo, hi) = flat.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    println!("Значения: min={lo:.2} max={hi:.2} mean={mean:.3} std={std:.3}");
    println!("(mean≈0; std≈1 только для коэффициентов с достаточной ненулевой дисперсией)\n");

    println!("PCM RMS — уровень исходного звука (полная шкала 1.0):");
    for (i, frame) in samples.chunks(mfcc::FRAME_SIZE).enumerate() {
        let level = wake_word_ml::audio::rms(frame);
        let bar = "#".repeat((level * 50.0).round() as usize);
        println!("  кадр {i:2} ({:4}мс) RMS={level:.5} {bar}", i * 30);
    }
    println!("\nНорма MFCC после CMVN — НЕ энергия звука (относительная шкала):");
    let norms: Vec<f32> = m.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();
    let max_norm = norms.iter().copied().fold(0.0_f32, f32::max);
    for (i, norm) in norms.iter().enumerate() {
        let width = if max_norm > 0.0 { (norm / max_norm * 50.0) as usize } else { 0 };
        println!("  кадр {i:2} norm={norm:.3} {}", "#".repeat(width));
    }
}

fn first_positive() -> String {
    let mut files: Vec<_> = std::fs::read_dir("dataset/positive")
        .expect("нет dataset/positive — запусти make preprocess")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "wav"))
        .collect();
    files.sort();
    files.first().expect("в dataset/positive пусто").to_str().unwrap().to_string()
}
