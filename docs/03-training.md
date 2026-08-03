# Обучение модели на burn

[burn](https://github.com/tracel-ai/burn) — Rust ML framework. WGPU/CUDA backend, типобезопасные тензоры.

---

## Cargo.toml

```toml
[package]
name = "wake-word-ml"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "0.14", features = ["train", "wgpu"] }
hound = "3.5"
rustfft = "6.2"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }

[[bin]]
name = "train"
path = "src/train.rs"

[[bin]]
name = "record"
path = "src/record.rs"

[[bin]]
name = "export"
path = "src/export.rs"
```

---

## Модель на burn

```rust
// src/model.rs
use burn::{
    nn::{Conv2d, Conv2dConfig, Linear, LinearConfig, Relu, Sigmoid},
    tensor::{Tensor, backend::AutodiffBackend},
    module::Module,
};

#[derive(Module, Debug)]
pub struct WakeWordModel<B: AutodiffBackend> {
    conv1: Conv2d<B, 1>,      // 1 канал → 16 фильтров
    conv2: Conv2d<B, 16>,     // 16 → 8 фильтров
    fc1: Linear<B>,           // flatten → 16
    fc2: Linear<B>,           // 16 → 1
    relu: Relu,
    sigmoid: Sigmoid,
}

impl<B: AutodiffBackend> WakeWordModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([1, 16], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let conv2 = Conv2dConfig::new([16, 8], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        // После 2 MaxPool(2,2): [33×20] → [16×10] → [8×5] → flatten = 8*5*8 = 320
        let fc1 = LinearConfig::new(320, 16).init(device);
        let fc2 = LinearConfig::new(16, 1).init(device);

        Self {
            conv1, conv2, fc1, fc2,
            relu: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // x: [batch, 1, 33, 20]
        let x = self.conv1.forward(x);    // [batch, 16, 33, 20]
        let x = self.relu.forward(x);
        let x = x.max_pool_2d([2, 2], [2, 2], [0, 0]);  // [batch, 16, 16, 10]

        let x = self.conv2.forward(x);    // [batch, 8, 16, 10]
        let x = self.relu.forward(x);
        let x = x.max_pool_2d([2, 2], [2, 2], [0, 0]);  // [batch, 8, 8, 5]

        let x = x.flatten(1, 3);         // [batch, 320]
        let x = self.fc1.forward(x);     // [batch, 16]
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);     // [batch, 1]
        let x = self.sigmoid.forward(x); // [batch, 1]

        x.squeeze(1)                     // [batch]
    }
}
```

---

## Обучение

```rust
// src/train.rs
use burn::{
    config::Config,
    data::{DataLoaderBuilder, dataset::Dataset},
    optim::AdamConfig,
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep},
    tensor::{Tensor, backend::{AutodiffBackend, Backend}},
};

#[derive(Config)]
pub struct TrainConfig {
    #[config(default = 50)]
    pub epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
}

impl<B: AutodiffBackend> TrainStep<(Tensor<B, 4>, Tensor<B, 1>), ClassificationLoss<B>>
    for WakeWordModel<B>
{
    fn step(&self, (input, target): (Tensor<B, 4>, Tensor<B, 1>)) -> TrainOutput<ClassificationLoss<B>> {
        let prediction = self.forward(input);
        let loss = binary_cross_entropy(&prediction, &target);

        let gradients = loss.backward();

        TrainOutput::new(self, gradients, ClassificationLoss::new(loss))
    }
}

fn binary_cross_entropy<B: Backend>(
    pred: &Tensor<B, 2>,
    target: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let eps = 1e-7;
    let pred_clipped = pred.clone().clamp(eps, 1.0 - eps);
    let loss = target.clone() * pred_clipped.clone().log()
        + (1.0 - target.clone()) * (1.0 - pred_clipped).log();
    -loss.mean()
}

pub fn train<B: AutodiffBackend>(device: B::Device, config: TrainConfig) {
    let model = WakeWordModel::new(&device);
    let optimizer = AdamConfig::new().with_learning_rate(config.learning_rate);

    let train_loader = DataLoaderBuilder::new(batch)
        .build(WakeWordDataset::train());
    let val_loader = DataLoaderBuilder::new(batch)
        .build(WakeWordDataset::val());

    let learner = LearnerBuilder::new("models/")
        .devices(vec![device.clone()])
        .num_epochs(config.epochs)
        .build(model, optimizer, 1e-3);

    let model = learner.fit(train_loader, val_loader);

    // Сохраняем модель
    model.save("models/wake_word.json").unwrap();
}
```

---

## Экспорт весов для ESP32

После обучения извлекаем веса в бинарный формат для C++:

```rust
// src/export.rs
use std::io::Write;

pub fn export_weights(model: &WakeWordModel<CpuBackend>, output: &str) {
    let mut file = std::fs::File::create(output).unwrap();

    // Conv1: weights [16, 1, 3, 3] + bias [16]
    let conv1_weights = model.conv1.weight.to_data();
    let conv1_bias = model.conv1.bias.to_data();
    write_tensor(&mut file, &conv1_weights, &[16, 1, 3, 3]);
    write_tensor(&mut file, &conv1_bias, &[16]);

    // Conv2: weights [8, 16, 3, 3] + bias [8]
    let conv2_weights = model.conv2.weight.to_data();
    let conv2_bias = model.conv2.bias.to_data();
    write_tensor(&mut file, &conv2_weights, &[8, 16, 3, 3]);
    write_tensor(&mut file, &conv2_bias, &[8]);

    // FC1: weights [320, 16] + bias [16]
    let fc1_weights = model.fc1.weight.to_data();
    let fc1_bias = model.fc1.bias.to_data();
    write_tensor(&mut file, &fc1_weights, &[320, 16]);
    write_tensor(&mut file, &fc1_bias, &[16]);

    // FC2: weights [16, 1] + bias [1]
    let fc2_weights = model.fc2.weight.to_data();
    let fc2_bias = model.fc2.bias.to_data();
    write_tensor(&mut file, &fc2_weights, &[16, 1]);
    write_tensor(&mut file, &fc2_bias, &[1]);

    println!("Exported to {} ({} bytes)", output, file.metadata().unwrap().len());
}

fn write_tensor(file: &mut std::fs::File, data: &TensorData, shape: &[usize]) {
    // Header: [ndim, dim0, dim1, ...]
    file.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
    for &dim in shape {
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
    }
    // Data: f32 values
    for v in data.iter::<f32>() {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}
```

Выходной файл: `models/wake_word.bin` (~20 KB).

---

## Оценка

```rust
pub fn evaluate(model: &WakeWordModel<B>, test_data: &[(Matrix, f32)]) {
    let mut tp = 0; let mut fp = 0;
    let mut tn = 0; let mut fn_ = 0;

    for (mfcc, label) in test_data {
        let input = tensor_from_mfcc(mfcc);
        let score = model.forward(input.unsqueeze());
        let predicted = score > 0.5;

        match (predicted, *label > 0.5) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    let accuracy = (tp + tn) as f32 / test_data.len() as f32;
    let fpr = fp as f32 / (fp + tn) as f32;
    let fnr = fn_ as f32 / (fn_ + tp) as f32;

    println!("Accuracy: {:.3}", accuracy);
    println!("False positive rate: {:.4}", fpr);
    println!("False negative rate: {:.4}", fnr);
}
```

### Целевые метрики

| Метрика | Цель |
|---------|------|
| Accuracy | >90% |
| False positive rate | <3% |
| False negative rate | <10% |
| Model size | <25 KB |

---

## Запуск

```bash
# Установка Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Запись датасета (на ноутбуке)
cargo run --bin record -- --word "гермес" --count 500 --output dataset/positive/

# Обучение
cargo run --bin train

# Экспорт весов для ESP32
cargo run --bin export

# Результат:
# models/wake_word.json  — burn модель (для дообучения)
# models/wake_word.bin   — бинарные веса для ESP32 (~20 KB)
```

---

## Дальше

- [04-deploy.md](04-deploy.md) — ручной C++ inference на ESP32