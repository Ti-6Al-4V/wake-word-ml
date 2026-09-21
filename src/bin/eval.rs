// Оценка обученной модели. Три режима:
//
//   eval [val|test]      кривая порогов на сплите: для порогов 0.05..0.95
//                        TPR и FA-доля. Val — чтобы ВЫБРАТЬ порог,
//                        test — один раз в конце, для честной цифры.
//   score <папка> [порог] прогнать любые wav (16kHz, 1.2с) и напечатать
//                        распределение вероятностей. Диагностика shortcut'ов:
//                        скорми свою речь без «Гермеса» — модель должна
//                        молчать; скорми двойники — тоже.
//   stream <файл.wav> [шаг_мс] прогон длинной записи скользящим окном,
//                        как на ESP32: сколько ложных тревог В ЧАС при
//                        каждом пороге. Файл — 16kHz моно любой длины
//                        (ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav).
//
// Модель берётся из models/hermes.bin (лучшая эпоха train).
//
// Запуск: cargo run --release --bin eval -- <режим> [аргументы]

use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use std::path::{Path, PathBuf};

use wake_word_ml::audio;
use wake_word_ml::dataset::{self, Sample};
use wake_word_ml::mfcc::{self, N_MFCC, NUM_FRAMES, WINDOW_SAMPLES};
use wake_word_ml::model::{batch_tensor, sigmoid, HermesNet};

type B = Wgpu;
const FEAT: usize = NUM_FRAMES * N_MFCC;
const BATCH: usize = 256;
const THRESHOLDS: [f32; 19] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

fn load_model(device: &WgpuDevice) -> HermesNet<B> {
    HermesNet::<B>::new(device, 0.0)
        .load_file("models/hermes", &BinFileRecorder::<FullPrecisionSettings>::new(), device)
        .expect("нет models/hermes.bin — сначала make train")
}

/// Вероятности для набора плоских признаков.
fn predict(model: &HermesNet<B>, feats: &[f32], device: &WgpuDevice) -> Vec<f32> {
    let n = feats.len() / FEAT;
    let mut out = Vec::with_capacity(n);
    for start in (0..n).step_by(BATCH) {
        let end = (start + BATCH).min(n);
        let x = batch_tensor::<B>(feats[start * FEAT..end * FEAT].to_vec(), end - start, device);
        let logits: Vec<f32> = model.forward(x).into_data().into_vec::<f32>().expect("логиты не снять с GPU");
        out.extend(logits.into_iter().map(sigmoid));
    }
    out
}

fn wavs_in(dir: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("не открыть {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "wav"))
        .collect();
    v.sort();
    v
}

/// Режим eval: кривая порогов на сплите.
fn eval_split(split: &str, device: &WgpuDevice) {
    let splits = dataset::load("dataset/positive", "dataset/negative");
    let samples: &[Sample] = match split {
        "val" => &splits.val,
        "test" => &splits.test,
        other => panic!("сплит {other}: ожидаю val или test"),
    };
    assert!(samples.iter().any(|s| s.label > 0.5) && samples.iter().any(|s| s.label < 0.5),
        "для оценки нужны оба класса в сплите");
    let model = load_model(device);
    let feats: Vec<f32> = samples.iter().flat_map(|s| dataset::features_flat(&s.path)).collect();
    let probs = predict(&model, &feats, device);

    let n_pos = samples.iter().filter(|s| s.label > 0.5).count();
    let n_neg = samples.len() - n_pos;
    let n_pos_groups = samples.iter().filter(|s| s.label > 0.5)
        .map(|s| dataset::group_key(&s.path)).collect::<std::collections::HashSet<_>>().len();
    println!("сплит {split}: {} файлов — позитивов {n_pos} (групп исходников {n_pos_groups}), негативов {n_neg}\n",
        samples.len());
    println!("Кривая порогов. TPR — доля узнанных «Гермес»; FA — доля негативов, на которые сработало.");
    println!("Порог выбираем на val/development, на test только оцениваем заранее выбранный. Clip FA не равна FA/h.\n");
    println!("{:>6} {:>7} {:>8} {:>6} {:>6}", "порог", "TPR", "FA-доля", "FN", "FP");
    for &t in &THRESHOLDS {
        let (mut tp, mut fp) = (0usize, 0usize);
        for (p, s) in probs.iter().zip(samples) {
            if *p > t {
                if s.label > 0.5 { tp += 1 } else { fp += 1 }
            }
        }
        let tpr = tp as f32 / n_pos.max(1) as f32;
        let fa = fp as f32 / n_neg.max(1) as f32;
        println!("{t:>6.2} {tpr:>7.3} {fa:>8.4} {:>6} {fp:>6}", n_pos - tp);
    }
    // Самые уверенные ошибки — что модель путает.
    let mut errs: Vec<(f32, &Sample)> = probs.iter().zip(samples)
        .filter(|(p, s)| (**p > 0.5) != (s.label > 0.5))
        .map(|(p, s)| (*p, s)).collect();
    errs.sort_by(|a, b| (b.0 - 0.5).abs().partial_cmp(&(a.0 - 0.5).abs()).unwrap());
    if !errs.is_empty() {
        println!("\nСамые уверенные ошибки при пороге 0.5 (первые 10):");
        for (p, s) in errs.iter().take(10) {
            let kind = if s.label > 0.5 { "пропуск" } else { "ложняк" };
            println!("  {p:.3} {kind:8} {}", s.path.file_name().unwrap().to_string_lossy());
        }
    }
    println!("\nTPR выше взвешен по файлам, включая аугментации. Групп позитивов: {n_pos_groups}; независимость сессий/дикторов не гарантирована. Интервал оценивай по независимому live-test (docs/07-quality.md).");
}

/// Режим score: распределение вероятностей по папке.
fn score_dir(dir: &str, threshold: f32, device: &WgpuDevice) {
    assert!(threshold.is_finite() && (0.0..=1.0).contains(&threshold), "порог должен быть в [0,1]");
    let files = wavs_in(dir);
    if files.is_empty() { panic!("в {dir} нет wav"); }
    let model = load_model(device);
    let feats: Vec<f32> = files.iter().flat_map(|p| dataset::features_flat(p)).collect();
    let probs = predict(&model, &feats, device);

    let mut hist = [0usize; 10];
    for p in &probs { hist[((p * 10.0) as usize).min(9)] += 1; }
    println!("{}: {} файлов, порог {threshold}\n", dir, files.len());
    println!("Гистограмма score «это Гермес» (не проверенная калибровка):");
    for (i, c) in hist.iter().enumerate() {
        let bar = "#".repeat(c * 60 / files.len().max(1));
        println!("  {:.1}–{:.1} {c:5} {bar}", i as f32 / 10.0, (i + 1) as f32 / 10.0);
    }
    let above = probs.iter().filter(|&&p| p > threshold).count();
    println!("\nВыше порога: {above} из {} ({:.1}%)", files.len(), 100.0 * above as f32 / files.len() as f32);
    let mut top: Vec<(f32, &PathBuf)> = probs.iter().cloned().zip(&files).collect();
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("Самые «гермесные» файлы:");
    for (p, f) in top.iter().take(8) {
        println!("  {p:.3} {}", f.file_name().unwrap().to_string_lossy());
    }
}

/// Режим stream: скользящее окно по длинной записи, FA/час.
fn stream(path: &str, hop_ms: usize, device: &WgpuDevice) {
    assert!((1..=2000).contains(&hop_ms), "шаг должен быть от 1 до 2000 мс");
    let (samples, rate) = audio::read_wav(Path::new(path)).expect("wav не читается");
    assert_eq!(rate as usize, mfcc::SAMPLE_RATE, "нужен 16kHz: ffmpeg -i in -ac 1 -ar 16000 out.wav");
    assert!(samples.len() >= WINDOW_SAMPLES, "для потока нужно минимум 1.2с аудио");
    println!("FA/h корректна только для записи БЕЗ целевого слова. Энергетический гейт выключен.");
    let hours = samples.len() as f32 / rate as f32 / 3600.0;
    let hop = rate as usize * hop_ms / 1000;
    let model = load_model(device);

    // Все окна → признаки → вероятности (батчами, чтобы не держать всё в памяти).
    let n_windows = if samples.len() >= WINDOW_SAMPLES { (samples.len() - WINDOW_SAMPLES) / hop + 1 } else { 0 };
    println!("{path}: {:.2} ч, окно 1.2с, шаг {hop_ms}мс → {n_windows} окон", hours);
    let mut probs = Vec::with_capacity(n_windows);
    let mut feats = Vec::with_capacity(BATCH * FEAT);
    for w in 0..n_windows {
        let start = w * hop;
        let m = mfcc::wav_to_mfcc(&samples[start..start + WINDOW_SAMPLES]);
        feats.extend(m.into_iter().flatten());
        if feats.len() == BATCH * FEAT || w + 1 == n_windows {
            probs.extend(predict(&model, &feats, device));
            feats.clear();
        }
    }

    // Детекция как на устройстве: порог превышен в 2 окнах подряд,
    // после срабатывания — cooldown 2с (не считать одно событие много раз).
    println!("\nЛожные тревоги в час (2 окна подряд, cooldown floor(2000/шаг) шагов):");
    println!("{:>6} {:>10} {:>10} {:>12}", "порог", "окон>порога", "детекций", "FA/час");
    let cooldown = 2000 / hop_ms.max(1);
    for &t in &THRESHOLDS {
        let (mut above, mut det, mut run, mut cool) = (0usize, 0usize, 0usize, 0usize);
        for &p in &probs {
            if p > t { above += 1; }
            if cool > 0 { cool -= 1; run = 0; continue; }
            if p > t { run += 1; } else { run = 0; }
            if run >= 2 { det += 1; cool = cooldown; run = 0; }
        }
        println!("{t:>6.2} {above:>10} {det:>10} {:>12.2}", det as f32 / hours.max(1e-6));
    }
    println!("\nЭто наблюдаемые частоты. Если событий 0, пуассоновская верхняя 95%-граница ≈ {:.3}/ч (допущения: docs/07-quality.md).", -0.05_f32.ln() / hours);
    println!("Цель: ≤0.1 FA/ч вместе с event recall ≥0.95 на отдельном размеченном потоке; clip TPR из make eval его не заменяет.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("eval");
    let device = WgpuDevice::DefaultDevice;
    match mode {
        "eval" => eval_split(args.get(2).map(String::as_str).unwrap_or("val"), &device),
        "score" => {
            let dir = args.get(2).expect("score <папка> [порог]");
            let t: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            score_dir(dir, t, &device);
        }
        "stream" => {
            let file = args.get(2).expect("stream <файл.wav> [шаг_мс]");
            let hop: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
            stream(file, hop, &device);
        }
        other => {
            eprintln!("Неизвестный режим {other}. Режимы: eval [val|test] | score <папка> [порог] | stream <wav> [шаг_мс]");
            std::process::exit(1);
        }
    }
}
