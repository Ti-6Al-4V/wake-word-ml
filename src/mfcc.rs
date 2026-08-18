//! MFCC (Mel-Frequency Cepstral Coefficients) — признаки, которые
//! «видит» нейросеть вместо сырых сэмплов.
//!
//! Зачем: сырой PCM плох для классификации — одна и та же буква звучит
//! по-разному у разных людей и на разной громкости, а нам нужна форма
//! речевого тракта. Цепочка:
//!
//! ```text
//! сэмплы окна (19200, 1.2с)
//!   → pre-emphasis (поднять высокие частоты — они затухают при речи)
//!   → 40 кадров по 480 сэмплов (30мс), каждый под окном Хэмминга
//!   → FFT 512 → спектр мощности
//!   → 20 треугольных фильтров на mel-шкале (модель слуха)
//!   → log (слух воспринимает громкость логарифмически)
//!   → DCT (сжать и декоррелировать) → 20 коэффициентов на кадр
//!   → нормализация всей матрицы (среднее 0, дисперсия 1)
//! ```
//!
//! Итог: матрица [40 кадров × 20 коэффициентов] на окно 1.2с.

use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_SIZE: usize = 480;  // 30мс кадра: 16000 * 0.03
pub const HOP_SIZE: usize = 480;    // кадры идут встык, без перекрытия
pub const FFT_SIZE: usize = 512;    // ближайшая степень двойки ≥ FRAME_SIZE
pub const N_MELS: usize = 20;       // число mel-фильтров
pub const N_MFCC: usize = 20;       // сколько коэффициентов оставляем после DCT
pub const NUM_FRAMES: usize = 40;   // 1.2с окно: 19200 / 480
pub const WINDOW_SAMPLES: usize = 19_200; // 1.2с при 16kHz

const PRE_EMPHASIS: f32 = 0.97;

// --- Mel-шкала: как человек воспринимает высоту звука ---
// Низкие частоты мы различаем лучше высоких; mel-шкала это моделирует.

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

// FFT-бины: bin k соответствует частоте k * SAMPLE_RATE / FFT_SIZE.
fn hz_to_bin(hz: f32) -> usize {
    ((FFT_SIZE + 1) as f32 * hz / SAMPLE_RATE as f32) as usize
}

/// Матрица mel-фильтров: N_MELS треугольных фильтров, равномерно
/// расставленных ПО MEL-ШКАЛЕ от 0 до 8 кГц (частота Найквиста для 16kHz).
/// Каждый фильтр — веса для спектральных бинов: 0 вне полосы,
/// линейный подъём до центра, линейный спад после.
fn mel_filterbank() -> Vec<Vec<f32>> {
    let mel_max = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    // N_MELS + 2 точки: нужны левая, центральная и правая граница каждого фильтра
    let points: Vec<usize> = (0..N_MELS + 2)
        .map(|i| hz_to_bin(mel_to_hz(mel_max * i as f32 / (N_MELS + 1) as f32)))
        .collect();

    let n_bins = FFT_SIZE / 2 + 1; // спектр симметричен — храним половину+1
    let mut bank = vec![vec![0.0f32; n_bins]; N_MELS];
    for (m, row) in bank.iter_mut().enumerate() {
        let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
        for k in left..center {
            row[k] = (k - left) as f32 / (center - left).max(1) as f32; // подъём
        }
        for k in center..right {
            row[k] = (right - k) as f32 / (right - center).max(1) as f32; // спад
        }
    }
    bank
}

/// Матрица DCT-II: каждой строкой «проецируем» log-mel-энергии
/// в компактные коэффициенты. Низкие номера = огибающая спектра
/// (форма речевого тракта), высокие — мелкая рябь (не нужна).
fn dct_matrix() -> Vec<Vec<f32>> {
    (0..N_MFCC)
        .map(|m| {
            (0..N_MELS)
                .map(|k| (std::f32::consts::PI * m as f32 * (k as f32 + 0.5) / N_MELS as f32).cos())
                .collect()
        })
        .collect()
}

/// Главная функция: окно 19200 сэмплов → матрица [40][20].
/// Короткий вход дополняется тишиной, длинный обрезается.
pub fn wav_to_mfcc(samples: &[f32]) -> Vec<Vec<f32>> {
    // 1. Pre-emphasis: y[n] = x[n] - 0.97·x[n-1].
    // Речевой сигнал теряет ~6дБ/октаву с ростом частоты; фильтр
    // выравнивает спектр, чтобы ВЧ-согласные (с, ш, р) были лучше видны.
    let mut x = vec![0.0f32; WINDOW_SAMPLES];
    let n = samples.len().min(WINDOW_SAMPLES);
    x[0] = samples[0];
    for i in 1..n {
        x[i] = samples[i] - PRE_EMPHASIS * samples[i - 1];
    }

    let bank = mel_filterbank();
    let dct = dct_matrix();
    let hamming: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
        .collect();

    // FFT-план создаём один раз и переиспользуем на все кадры.
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut mfcc = vec![vec![0.0f32; N_MFCC]; NUM_FRAMES];

    for frame in 0..NUM_FRAMES {
        let start = frame * HOP_SIZE;

        // 2. Кадр + окно Хэмминга (гасит разрывы на краях кадра,
        //    чтобы FFT не видел ложных высоких частот).
        let mut buf = vec![Complex::new(0.0, 0.0); FFT_SIZE]; // нули = zero-padding
        for i in 0..FRAME_SIZE {
            if start + i < x.len() {
                buf[i] = Complex::new(x[start + i] * hamming[i], 0.0);
            }
        }

        // 3. FFT → спектр мощности |X[k]|².
        fft.process(&mut buf);
        let n_bins = FFT_SIZE / 2 + 1;
        let power: Vec<f32> = buf[..n_bins].iter().map(|c| c.norm_sqr()).collect();

        // 4-5. Mel-фильтры + log. Эпсилон внутри log защищает от log(0).
        let mut mel_log = vec![0.0f32; N_MELS];
        for m in 0..N_MELS {
            let energy: f32 = bank[m].iter().zip(&power).map(|(w, p)| w * p).sum();
            mel_log[m] = (energy + 1e-8).ln();
        }

        // 6. DCT → коэффициенты MFCC.
        for m in 0..N_MFCC {
            mfcc[frame][m] = dct[m].iter().zip(&mel_log).map(|(c, e)| c * e).sum();
        }
    }

    // 7. Глобальная нормализация матрицы: среднее 0, дисперсия 1.
    // Убирает остаточную зависимость от громкости записи.
    let all = NUM_FRAMES * N_MFCC;
    let mean: f32 = mfcc.iter().flatten().sum::<f32>() / all as f32;
    let std: f32 = (mfcc.iter().flatten().map(|v| (v - mean).powi(2)).sum::<f32>() / all as f32).sqrt() + 1e-8;
    for row in &mut mfcc {
        for v in row.iter_mut() {
            *v = (*v - mean) / std;
        }
    }
    mfcc
}
