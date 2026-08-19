// Обучение CNN на burn. Первый настоящий запуск модели.
//
// Архитектура (по docs/03, адаптировано под burn 0.21):
//   вход [1, 40, 20] (1 канал, 40 кадров MFCC × 20 фичей)
//   → Conv2D 16 фильтров 3×3 → ReLU → MaxPool 2×2   [16, 19, 9]
//   → Conv2D  8 фильтров 3×3 → ReLU → MaxPool 2×2   [8, 8, 3]
//   → Flatten (192) → Dense 16 → ReLU → Dense 1 → logit
//
// Ручной цикл обучения вместо burn-train Learner: так видно КАЖДЫЙ шаг
// (батч → forward → loss → backward → optimizer.step). Для маленькой
// модели этого достаточно и честнее для понимания.
//
// Запуск:  cargo run --bin train --release -- [эпохи] [батч] [lr]
// Пример:  cargo run --bin train --release -- 10 64 0.001
//
// ВАЖНО: --release. В dev-сборке тензорная матемика в разы медленнее.

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Два типа бейкенда: с автодифференцированием для обучения (считает
// градиенты) и чистый Wgpu для инференса/валидации (быстрее, без графов).
type TrainBackend = Autodiff<Wgpu>;
type InferBackend = Wgpu;

#[derive(Module, Debug)]
pub struct HermesNet<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> HermesNet<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            // channels: [входные, выходные]; паддинг Valid — без дополнения краёв
            conv1: Conv2dConfig::new([1, 16], [3, 3]).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([16, 8], [3, 3]).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            // 192 = 8 каналов × 8 × 3 после второго пула (см. шапку файла)
            fc1: LinearConfig::new(192, 16).init(device),
            fc2: LinearConfig::new(16, 1).init(device),
        }
    }

    // Возвращаем ЛОГИТ (до сигмоиды): BCE-with-logits численно стабильнее,
    // чем sigmoid + обычный BCE.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 1> {
        let x = burn::tensor::activation::relu(self.conv1.forward(x));
        let x = self.pool1.forward(x);
        let x = burn::tensor::activation::relu(self.conv2.forward(x));
        let x = self.pool2.forward(x);
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c * h * w]); // Flatten
        let x = burn::tensor::activation::relu(self.fc1.forward(x));
        let x = self.fc2.forward(x); // [b, 1]
        x.reshape([b])                 // [b] — форма для BCE
    }
}

fn main() {
    let epochs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let batch: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    let splits = wake_word_ml::dataset::load("dataset/positive", "dataset/negative");
    println!("train={} val={} test={}", splits.train.len(), splits.val.len(), splits.test.len());

    let device = WgpuDevice::DefaultDevice;
    let mut model = HermesNet::<TrainBackend>::new(&device);
    // AdamConfig::new() — дефолтные beta 0.9/0.999; init привязывает
    // оптимизатор к типу модели и бейкенду.
    let mut optim = AdamConfig::new().init::<TrainBackend, HermesNet<TrainBackend>>();
    // with_logits(true): на вход логиты, сигмоида внутри и численно аккуратно.
    let loss_fn = BinaryCrossEntropyLossConfig::new().with_logits(true).init::<TrainBackend>(&device);

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();

        // Детерминированное перемешивание индексов train: свой сид на эпоху
        // (порядок батчей разный, но прогон воспроизводимый).
        let mut idx: Vec<usize> = (0..splits.train.len()).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(wake_word_ml::dataset::SEED + epoch as u64);
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }

        let mut loss_sum = 0.0f32;
        let mut n_batches = 0usize;
        for chunk in idx.chunks(batch) {
            // Собираем батч: MFCC считается на лету (CPU), затем уходит на GPU.
            // Для 6K файлов это доли секунды; закэшируем, если станет узким местом.
            let mut feats = Vec::with_capacity(chunk.len() * 40 * 20);
            let mut labels = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let s = &splits.train[i];
                let m = wake_word_ml::dataset::features(&s.path); // [40][20]
                feats.extend(m.iter().flatten().cloned());
                labels.push(s.label as i64);
            }
            let b = chunk.len();
            // Тензоры: фичи [b, 1, 40, 20] (канал 1), метки [b] целые 0/1.
            let x = Tensor::<TrainBackend, 4>::from_floats(
                TensorData::new(feats, [b, 1, 40, 20]), &device);
            let y = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(labels, [b]), &device);

            // Шаг обучения — четыре строки, ради которых всё затевалось:
            let logits = model.forward(x);             // forward pass
            let loss = loss_fn.forward(logits, y);     // loss
            let grads = GradientsParams::from_grads(loss.backward(), &model); // backprop
            model = optim.step(lr, model, grads);      // Adam обновляет веса

            loss_sum += loss.into_scalar();
            n_batches += 1;
        }

        // --- Валидация без градиентов: .valid() даёт модель на чистом Wgpu ---
        let vnet = model.clone().valid();
        let (mut tp, mut fn_, mut tn, mut fp) = (0usize, 0usize, 0usize, 0usize);
        for chunk in splits.val.chunks(batch) {
            let mut feats = Vec::with_capacity(chunk.len() * 40 * 20);
            let mut labels = Vec::with_capacity(chunk.len());
            for s in chunk {
                let m = wake_word_ml::dataset::features(&s.path);
                feats.extend(m.iter().flatten().cloned());
                labels.push(s.label as i64);
            }
            let b = chunk.len();
            let x = Tensor::<InferBackend, 4>::from_floats(
                TensorData::new(feats, [b, 1, 40, 20]), &device);
            // Логиты снимаем на CPU, сигмоиду и порог считаем здесь:
            // батч маленький, а так не зависит от тонкостей tensor-API.
            let logits: Vec<f32> = vnet.forward(x).into_data()
                .into_vec::<f32>().expect("логиты не снять с GPU");
            for (logit, s) in logits.into_iter().zip(chunk.iter()) {
                let prob = 1.0 / (1.0 + (-logit).exp()); // сигмоида
                let pred = prob > 0.5;
                match (pred, s.label == 1.0) {
                    (true, true) => tp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => tn += 1,
                    (true, false) => fp += 1,
                }
            }
        }
        let tpr = tp as f32 / (tp + fn_).max(1) as f32;          // узнавание слова
        let fa = fp as f32 / (fp + tn).max(1) as f32;            // ложные срабатывания на «нет»
        let acc = (tp + tn) as f32 / splits.val.len() as f32;
        println!("эпоха {:2}/{} за {:.1}с | loss {:.4} | val acc {:.3} | TPR {:.3} | FA-доля {:.4}",
            epoch + 1, epochs, t0.elapsed().as_secs_f32(), loss_sum / n_batches as f32, acc, tpr, fa);
    }

    // Сохраняем веса: бинарный формат, расширение файл получит сам.
    std::fs::create_dir_all("models").expect("не создать models/");
    model.save_file("models/hermes", &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("не сохранить модель");
    println!("\nМодель сохранена в models/");
    println!("test-прогон — следующий шаг (eval.rs).");
}
