//! Архитектура HermesNet — одна на train, eval и export.
//!
//! ```text
//! вход [b, 1, 40, 20]     (1 канал, 40 кадров MFCC × 20 коэффициентов)
//!  → Conv2D 1→16, 3×3, Valid → ReLU → MaxPool 2×2     [16, 19, 9]
//!  → Conv2D 16→8, 3×3, Valid → ReLU → MaxPool 2×2     [8, 8, 3]
//!  → Flatten (192) → Dropout → Dense 16 → ReLU → Dense 1 → логит
//! ```
//!
//! Параметров: 16·(9+1) + 8·(16·9+1) + 192·16+16 + 16+1 = 4425.
//!
//! Dropout — единственный «регулятор» внутри сети: с prob = 0 это
//! тождественное преобразование (по умолчанию), включается флагом
//! обучения, когда виден разрыв train/val. На инференс-бэкенде
//! (без autodiff) burn выключает его сам.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;

/// Число признаков после второго пула: 8 каналов × 8 × 3.
pub const FLAT: usize = 8 * 8 * 3;

#[derive(Module, Debug)]
pub struct HermesNet<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> HermesNet<B> {
    pub fn new(device: &B::Device, dropout: f64) -> Self {
        Self {
            // channels: [входные, выходные]; паддинг по умолчанию Valid — без дополнения краёв
            conv1: Conv2dConfig::new([1, 16], [3, 3]).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([16, 8], [3, 3]).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            dropout: DropoutConfig::new(dropout).init(),
            fc1: LinearConfig::new(FLAT, 16).init(device),
            fc2: LinearConfig::new(16, 1).init(device),
        }
    }

    /// Возвращает ЛОГИТ (до сигмоиды): BCE-with-logits численно
    /// стабильнее, чем sigmoid + обычный BCE. Сигмоида — на стороне того,
    /// кто интерпретирует (eval, ESP32).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 1> {
        let x = burn::tensor::activation::relu(self.conv1.forward(x));
        let x = self.pool1.forward(x);
        let x = burn::tensor::activation::relu(self.conv2.forward(x));
        let x = self.pool2.forward(x);
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c * h * w]); // Flatten
        let x = self.dropout.forward(x);
        let x = burn::tensor::activation::relu(self.fc1.forward(x));
        let x = self.fc2.forward(x); // [b, 1]
        x.reshape([b])                 // [b] — форма для BCE
    }
}

/// Сигмоида на CPU: логит → вероятность.
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// Батч плоских признаков → тензор [b, 1, 40, 20].
pub fn batch_tensor<B: Backend>(feats: Vec<f32>, b: usize, device: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 4>::from_floats(
        TensorData::new(feats, [b, 1, crate::mfcc::NUM_FRAMES, crate::mfcc::N_MFCC]),
        device,
    )
}
