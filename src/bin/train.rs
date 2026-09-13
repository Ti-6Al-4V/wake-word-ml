// Обучение CNN на burn. Архитектура — src/model.rs (общая с eval/export).
//
// Ручной цикл обучения вместо burn-train Learner: так видно КАЖДЫЙ шаг
// (батч → forward → loss → backward → optimizer.step). Для маленькой
// модели этого достаточно и честнее для понимания.
//
// Что делает помимо самого цикла:
//   - сидирует бэкенд: веса при старте одинаковы от запуска к запуску,
//     порядок батчей тоже → эксперименты сравнимы (меняешь один параметр —
//     видишь его эффект, а не лотерею инициализации);
//   - считает MFCC один раз для всех файлов и держит в памяти
//     (9К файлов × 800 чисел ≈ 29 МБ) — эпоха идёт секунды, а не минуты;
//   - на val каждую эпоху считает и loss, и метрики при пороге 0.5;
//   - пишет models/train_log.csv — по нему рисуется кривая обучения;
//   - сохраняет ЛУЧШУЮ по val-loss эпоху в models/hermes.bin, последнюю —
//     в models/hermes_last.bin. Последняя эпоха не обязана быть лучшей:
//     при переобучении val-loss растёт, пока train-loss падает.
//
// Запуск:  cargo run --bin train --release -- [эпохи] [батч] [lr] [dropout] [weight_decay]
// Пример:  cargo run --bin train --release -- 30 64 0.001 0.2 0.0001
//
// ВАЖНО: --release. В dev-сборке тензорная математика в разы медленнее.

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;

use wake_word_ml::dataset::{self, Sample, SEED};
use wake_word_ml::mfcc::{N_MFCC, NUM_FRAMES};
use wake_word_ml::model::{batch_tensor, sigmoid, HermesNet};

// Два типа бэкенда: с автодифференцированием для обучения (считает
// градиенты) и чистый Wgpu для инференса/валидации (быстрее, без графов).
type TrainBackend = Autodiff<Wgpu>;
type InferBackend = Wgpu;

const FEAT: usize = NUM_FRAMES * N_MFCC; // 800 чисел на файл

/// Признаки одного сплита, посчитанные один раз: плоский массив + метки.
struct Cached {
    feats: Vec<f32>,   // [n × FEAT]
    labels: Vec<f32>,  // [n]
}

fn cache(samples: &[Sample]) -> Cached {
    let mut feats = Vec::with_capacity(samples.len() * FEAT);
    let mut labels = Vec::with_capacity(samples.len());
    for s in samples {
        feats.extend(dataset::features_flat(&s.path));
        labels.push(s.label);
    }
    Cached { feats, labels }
}

/// Метрики на одном сплите при пороге 0.5 + средний BCE-loss.
struct ValStats { loss: f32, acc: f32, tpr: f32, fa: f32 }

fn evaluate(model: &HermesNet<InferBackend>, data: &Cached, batch: usize, device: &WgpuDevice) -> ValStats {
    let n = data.labels.len();
    let (mut tp, mut fn_, mut tn, mut fp) = (0usize, 0usize, 0usize, 0usize);
    let mut loss_sum = 0.0f32;
    for start in (0..n).step_by(batch) {
        let end = (start + batch).min(n);
        let b = end - start;
        let x = batch_tensor::<InferBackend>(data.feats[start * FEAT..end * FEAT].to_vec(), b, device);
        // Логиты снимаем на CPU, сигмоиду, порог и loss считаем здесь.
        let logits: Vec<f32> = model.forward(x).into_data().into_vec::<f32>().expect("логиты не снять с GPU");
        for (logit, &y) in logits.into_iter().zip(&data.labels[start..end]) {
            let p = sigmoid(logit);
            // BCE = −[y·ln p + (1−y)·ln(1−p)]; clamp — чтобы ln(0) не дал −inf
            let pc = p.clamp(1e-7, 1.0 - 1e-7);
            loss_sum += -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln());
            match (p > 0.5, y > 0.5) {
                (true, true) => tp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
            }
        }
    }
    ValStats {
        loss: loss_sum / n.max(1) as f32,
        acc: (tp + tn) as f32 / n.max(1) as f32,
        tpr: tp as f32 / (tp + fn_).max(1) as f32,   // узнавание слова
        fa: fp as f32 / (fp + tn).max(1) as f32,      // ложные срабатывания на «нет»
    }
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i).and_then(|s| s.parse::<f64>().ok());
    let epochs = arg(1).unwrap_or(10.0) as usize;
    let batch = arg(2).unwrap_or(64.0) as usize;
    let lr = arg(3).unwrap_or(1e-3);
    let dropout = arg(4).unwrap_or(0.0);
    let weight_decay = arg(5).unwrap_or(0.0) as f32;
    println!("эпох {epochs} | батч {batch} | lr {lr} | dropout {dropout} | weight_decay {weight_decay}");

    let device = WgpuDevice::DefaultDevice;
    // Сид бэкенда: инициализация весов воспроизводима.
    TrainBackend::seed(&device, SEED);

    let splits = dataset::load("dataset/positive", "dataset/negative");
    let count = |v: &[Sample]| v.iter().filter(|s| s.label > 0.5).count();
    println!("train={} (позитивов {}) val={} (позитивов {}) test={}",
        splits.train.len(), count(&splits.train), splits.val.len(), count(&splits.val), splits.test.len());

    // Признаки считаем один раз. Test здесь НЕ трогаем — он для eval.
    let t0 = std::time::Instant::now();
    let train = cache(&splits.train);
    let val = cache(&splits.val);
    println!("MFCC посчитаны за {:.1}с", t0.elapsed().as_secs_f32());

    let mut model = HermesNet::<TrainBackend>::new(&device, dropout);
    // AdamConfig::new() — дефолтные beta 0.9/0.999. Weight decay (L2-штраф
    // на величину весов) — 0 по умолчанию, включается аргументом.
    let mut adam = AdamConfig::new();
    if weight_decay > 0.0 {
        adam = adam.with_weight_decay(Some(WeightDecayConfig::new(weight_decay)));
    }
    let mut optim = adam.init::<TrainBackend, HermesNet<TrainBackend>>();
    // with_logits(true): на вход логиты, сигмоида внутри и численно аккуратно.
    let loss_fn = BinaryCrossEntropyLossConfig::new().with_logits(true).init::<TrainBackend>(&device);

    std::fs::create_dir_all("models").expect("не создать models/");
    let mut log = std::fs::File::create("models/train_log.csv").expect("не создать train_log.csv");
    writeln!(log, "epoch,train_loss,val_loss,val_acc,val_tpr,val_fa").unwrap();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let n_train = train.labels.len();
    let mut best: Option<(usize, f32)> = None; // (эпоха, val_loss)

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();

        // Детерминированное перемешивание индексов train: свой сид на эпоху
        // (порядок батчей разный, но прогон воспроизводимый).
        let mut idx: Vec<usize> = (0..n_train).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED + epoch as u64);
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }

        let mut loss_sum = 0.0f32;
        let mut n_batches = 0usize;
        for chunk in idx.chunks(batch) {
            // Собираем батч из кэша признаков.
            let mut feats = Vec::with_capacity(chunk.len() * FEAT);
            let mut labels = Vec::with_capacity(chunk.len());
            for &i in chunk {
                feats.extend_from_slice(&train.feats[i * FEAT..(i + 1) * FEAT]);
                labels.push(train.labels[i] as i64);
            }
            let b = chunk.len();
            // Тензоры: фичи [b, 1, 40, 20] (канал 1), метки [b] целые 0/1.
            let x = batch_tensor::<TrainBackend>(feats, b, &device);
            let y = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(labels, [b]), &device);

            // Шаг обучения — четыре строки, ради которых всё затевалось:
            let logits = model.forward(x);             // forward pass
            let loss = loss_fn.forward(logits, y);     // loss
            let grads = GradientsParams::from_grads(loss.backward(), &model); // backprop
            model = optim.step(lr, model, grads);      // Adam обновляет веса

            loss_sum += loss.into_scalar();
            n_batches += 1;
        }
        let train_loss = loss_sum / n_batches.max(1) as f32;

        // --- Валидация без градиентов: .valid() даёт модель на чистом Wgpu ---
        let vnet = model.clone().valid();
        let v = evaluate(&vnet, &val, batch, &device);
        println!("эпоха {:2}/{} за {:.1}с | loss {:.4} | val loss {:.4} | val acc {:.3} | TPR {:.3} | FA-доля {:.4}",
            epoch + 1, epochs, t0.elapsed().as_secs_f32(), train_loss, v.loss, v.acc, v.tpr, v.fa);
        writeln!(log, "{},{:.5},{:.5},{:.4},{:.4},{:.5}", epoch + 1, train_loss, v.loss, v.acc, v.tpr, v.fa).unwrap();

        // Чекпоинт лучшей эпохи по val-loss (не по acc: loss чувствителен
        // к уверенности ошибок, а acc при дисбалансе классов врёт).
        if best.map_or(true, |(_, l)| v.loss < l) {
            best = Some((epoch + 1, v.loss));
            vnet.clone().save_file("models/hermes", &recorder).expect("не сохранить модель");
        }
    }

    model.valid().save_file("models/hermes_last", &recorder).expect("не сохранить модель");
    if let Some((e, l)) = best {
        println!("\nЛучшая эпоха: {e} (val loss {l:.4}) → models/hermes.bin; последняя → models/hermes_last.bin");
    }
    println!("Кривая обучения: models/train_log.csv. Дальше: make eval");
}
