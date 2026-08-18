// Проверка MFCC на живом файле: печатает статистику матрицы и
// ASCII-карту энергии по кадрам — видно, где в окне звучит слово.
//
// Запуск: cargo run --bin mfcc_check -- [путь к wav]
// Без пути берёт первый файл из dataset/positive.

use wake_word_ml::mfcc;

fn main() {
    // args(): [бинарь, arg1, ...] — путь это первый аргумент, nth(1)
    let path = std::env::args().nth(1).unwrap_or_else(first_positive);
    println!("Файл: {path}");

    let mut reader = hound::WavReader::open(&path).expect("wav не читается");
    let rate = reader.spec().sample_rate;
    let samples: Vec<f32> = reader.samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();
    println!("Сэмплов: {} ({:.2}с при {rate}Hz)", samples.len(), samples.len() as f32 / rate as f32);
    if rate != mfcc::SAMPLE_RATE as u32 {
        println!("ВНИМАНИЕ: частота не 16kHz — сначала make preprocess");
    }

    let m = mfcc::wav_to_mfcc(&samples);
    println!("Матрица MFCC: {} кадров × {} коэффициентов (ожидаем {}×{})",
        m.len(), m[0].len(), mfcc::NUM_FRAMES, mfcc::N_MFCC);

    let flat: Vec<f32> = m.iter().flatten().cloned().collect();
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
    let std: f32 = (flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
    let (lo, hi) = flat.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    println!("Значения: min={lo:.2} max={hi:.2} mean={mean:.3} std={std:.3}");
    println!("(после нормализации ждём mean≈0, std≈1)\n");

    // Энергия кадра = сумма квадратов коэффициентов: где слово — там горячее.
    println!("Энергия по кадрам (30мс каждый, слово должно гореть в середине):");
    let energies: Vec<f32> = m.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>())
        .collect();
    let max_e = energies.iter().cloned().fold(0.0f32, f32::max);
    for (i, e) in energies.iter().enumerate() {
        let bar = "#".repeat((e / max_e * 50.0) as usize);
        println!("  кадр {i:2} ({:4.0}мс) {bar}", i as f32 * 30.0);
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
