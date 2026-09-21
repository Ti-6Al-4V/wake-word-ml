//! MFCC (Mel-Frequency Cepstral Coefficients) — признаки, которые
//! «видит» нейросеть вместо сырых сэмплов.
//!
//! Frontend: pre-emphasis → 40 кадров 30мс без перекрытия → Hamming
//! → FFT 512 → power → 20 mel-полос → ln(max(E, 1e-5))
//! → DCT-II 20→20 → CMVN по времени для каждого коэффициента.
//!
//! Это выбранное представление, не единственно возможное: модели могут
//! учиться на PCM и log-mel. После DCT ось коэффициентов не является
//! частотной осью. Пол mel-энергии не имеет прямой калибровки в dBFS
//! без учёта нормировки FFT, окна и фильтров. CMVN не гарантирует
//! независимость от микрофона/шума; подробности в docs/01-theory.md.

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
/// Нижняя планка энергии mel-полосы перед логарифмом (см. шапку файла).
pub const LOG_FLOOR: f32 = 1e-5;

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
/// в коэффициенты. Все 20 сохранены: DCT здесь не сокращает размер.
/// Низкие номера — плавная форма log-спектра, высокие — быстрые изменения.
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
    // Фильтр усиливает высокие частоты относительно низких;
    // полезность этой настройки проверяется на development-данных.
    let mut x = vec![0.0f32; WINDOW_SAMPLES];
    let n = samples.len().min(WINDOW_SAMPLES);
    if n == 0 {
        return vec![vec![0.0f32; N_MFCC]; NUM_FRAMES];
    }
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

        // 4-5. Mel-фильтры + log с нижней границей для каждой полосы.
        let mut mel_log = vec![0.0f32; N_MELS];
        for m in 0..N_MELS {
            let energy: f32 = bank[m].iter().zip(&power).map(|(w, p)| w * p).sum();
            mel_log[m] = energy.max(LOG_FLOOR).ln();
        }

        // 6. DCT → коэффициенты MFCC.
        for m in 0..N_MFCC {
            mfcc[frame][m] = dct[m].iter().zip(&mel_log).map(|(c, e)| c * e).sum();
        }
    }

    // 7. Нормализация по коэффициентам (CMVN): для каждого из 20
    // коэффициентов среднее 0 и дисперсия 1 по его 40 кадрам.
    // Убирает временное среднее и меняет масштаб; шум, log-floor и
    // состав окна ограничивают инвариантность к усилению/микрофону.
    for c in 0..N_MFCC {
        // f64 для статистик: суммирование одинаковых f32 в f32 может
        // сместить среднее. Деление этого остатка на epsilon превращало
        // постоянную тишину в заметный ненулевой признак.
        let mean = mfcc.iter().map(|row| row[c] as f64).sum::<f64>() / NUM_FRAMES as f64;
        let var = mfcc.iter().map(|row| (row[c] as f64 - mean).powi(2)).sum::<f64>() / NUM_FRAMES as f64;
        let std = var.sqrt() + 1e-5;
        for row in mfcc.iter_mut() {
            row[c] = ((row[c] as f64 - mean) / std) as f32;
        }
    }
    mfcc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_per_coefficient_normalization() {
        // Синус 440 Гц с дизером: матрица нужной формы, каждый коэффициент
        // после CMVN имеет среднее ≈0 и дисперсию ≈1 по кадрам.
        let mut rng = crate::audio::rng_for("t", 0);
        let s: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin()
                + crate::audio::dither(&mut rng))
            .collect();
        let m = wav_to_mfcc(&s);
        assert_eq!(m.len(), NUM_FRAMES);
        assert_eq!(m[0].len(), N_MFCC);
        for c in 0..N_MFCC {
            let mean: f32 = m.iter().map(|r| r[c]).sum::<f32>() / NUM_FRAMES as f32;
            assert!(mean.abs() < 1e-3, "коэф {c}: mean {mean}");
        }
    }

    #[test]
    fn zeros_do_not_explode() {
        // Цифровые нули: log-пол держит значения в разумных пределах,
        // а std=0 не даёт NaN.
        let m = wav_to_mfcc(&vec![0.0f32; WINDOW_SAMPLES]);
        assert!(m.iter().flatten().all(|v| v.is_finite()));
        assert!(m.iter().flatten().all(|v| v.abs() < 1e-6),
            "постоянная тишина должна давать нулевые CMVN-признаки");
        assert!(wav_to_mfcc(&[]).iter().flatten().all(|v| *v == 0.0));
    }
}
