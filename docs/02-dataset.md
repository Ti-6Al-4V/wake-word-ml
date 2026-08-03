# Датасет

Сбор и подготовка данных на Rust.

---

## Структура

```
dataset/
├── positive/          # "гермес"
│   ├── germes_001.wav
│   ├── germes_002.wav
│   └── ...
├── negative/          # другие слова + шум
│   ├── other_001.wav
│   ├── noise_001.wav
│   └── ...
└── background/        # тишина, музыка, улица
    ├── silence_001.wav
    └── ...
```

---

## Запись позитивных ("гермес")

Скрипт на Rust записывает с микрофона:

```rust
// training/src/record.rs
use hound::{WavWriter, WavSpec};
use cpal::{Stream, StreamConfig};

fn record_word(output_dir: &str, count: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    for i in 1..=count {
        println!("Запись {}/{} — скажи \"гермес\"", i, count);
        
        // Запись 1 сек
        let filename = format!("{}/germes_{:03}.wav", output_dir, i);
        let mut writer = WavWriter::create(&filename, spec).unwrap();
        
        // cpal захват аудио → writer.write_sample()
        // ...
        
        println!("Сохранено: {}", filename);
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}
```

### Аугментация на Rust

Из 100 реальных → 1000 аугментированных:

```rust
// training/src/augment.rs
use rustfft::{FftPlanner, num_complex::Complex};

pub fn augment(samples: &[f32], sr: u32) -> Vec<Vec<f32>> {
    let mut result = Vec::new();

    // Pitch shift через FFT
    for shift in [-2.0, 2.0] {
        result.push(pitch_shift(samples, sr, shift));
    }

    // Time stretch
    for rate in [0.9, 1.1] {
        result.push(time_stretch(samples, rate));
    }

    // Добавить шум
    for snr_db in [10.0, 20.0] {
        let noise: Vec<f32> = (0..samples.len())
            .map(|_| {
                let n = rand::random::<f32>() * 2.0 - 1.0;
                n * 0.01 * 10.0_f32.powf(-snr_db / 20.0)
            })
            .collect();
        result.push(samples.iter().zip(noise.iter()).map(|(s, n)| s + n).collect());
    }

    // Громкость
    for vol in [0.7, 1.3] {
        result.push(samples.iter().map(|s| s * vol).collect());
    }

    result
}
```

---

## Негативные

Источники:
- **Google Speech Commands** — 35 слов, ~100k записей (скачать WAV)
- **Похожие слова:** "германий", "серьёзно", "героизм" — чтобы модель отличала
- **Шум:** музыка, улица, офис (записать или скачать)

---

## Preprocessing

Все WAV → единый формат (16kHz, mono, 1 сек):

```rust
// training/src/preprocess.rs
use hound::{WavReader, WavSpec, WavWriter};

pub fn preprocess(input: &str, output: &str) {
    let mut reader = WavReader::open(input).unwrap();
    let spec = reader.spec();

    // Декодируем в f32
    let samples: Vec<f32> = reader.samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();

    // Нормализация громкости
    let max = samples.iter().cloned().fold(0.0f32, f32::max).abs();
    let normalized: Vec<f32> = samples.iter().map(|s| s / max * 0.9).collect();

    // Обрезка/пад до 16000 сэмплов (1 сек)
    let mut padded = vec![0.0f32; 16000];
    let len = normalized.len().min(16000);
    padded[..len].copy_from_slice(&normalized[..len]);

    // Запись
    let out_spec = WavSpec {
        channels: 1, sample_rate: 16000,
        bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output, out_spec).unwrap();
    for s in &padded {
        writer.write_sample((*s * 32767.0) as i16).unwrap();
    }
}
```

---

## MFCC извлечение на Rust

```rust
// training/src/mfcc.rs
use rustfft::{FftPlanner, num_complex::Complex};

pub const N_MFCC: usize = 20;
pub const FRAME_SIZE: usize = 480;  // 30мс при 16kHz
pub const HOP_SIZE: usize = 480;
pub const FFT_SIZE: usize = 512;
pub const NUM_FRAMES: usize = 33;

pub fn wav_to_mfcc(samples: &[f32], sr: u32) -> [[f32; N_MFCC]; NUM_FRAMES] {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Hamming window
    let hamming: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
        .collect();

    // Mel filterbank (предрассчитан)
    let mel_filters = compute_mel_filterbank();  // [N_MFCC][FFT_SIZE/2]

    // DCT matrix (предрассчитана)
    let dct_matrix = compute_dct_matrix();  // [N_MFCC][N_MFCC]

    let mut mfcc = [[0.0f32; N_MFCC]; NUM_FRAMES];

    for frame in 0..NUM_FRAMES {
        let start = frame * HOP_SIZE;

        // 1. Pre-emphasis + Hamming
        let mut windowed = vec![Complex::new(0.0, 0.0); FFT_SIZE];
        for i in 0..FRAME_SIZE.min(samples.len() - start) {
            windowed[i] = Complex::new(samples[start + i] * hamming[i], 0.0);
        }

        // 2. FFT → спектр мощности
        fft.process(&mut windowed);
        let power: Vec<f32> = windowed[..FFT_SIZE/2]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // 3. Mel filterbank → 20 полос
        let mut mel_energy = [0.0f32; N_MFCC];
        for m in 0..N_MFCC {
            for k in 0..FFT_SIZE/2 {
                mel_energy[m] += mel_filters[m][k] * power[k];
            }
        }

        // 4. Log
        for m in 0..N_MFCC {
            mel_energy[m] = (mel_energy[m] + 1e-8).ln();
        }

        // 5. DCT → MFCC
        for m in 0..N_MFCC {
            for k in 0..N_MFCC {
                mfcc[frame][m] += dct_matrix[m][k] * mel_energy[k];
            }
        }
    }

    // 6. Нормализация (mean=0, std=1)
    let mean: f32 = mfcc.iter().flat_map(|r| r.iter()).sum::<f32>()
        / (NUM_FRAMES * N_MFCC) as f32;
    let std = (mfcc.iter().flat_map(|r| r.iter())
        .map(|v| (v - mean).powi(2)).sum::<f32>()
        / (NUM_FRAMES * N_MFCC) as f32).sqrt() + 1e-8;

    for frame in 0..NUM_FRAMES {
        for m in 0..N_MFCC {
            mfcc[frame][m] = (mfcc[frame][m] - mean) / std;
        }
    }

    mfcc
}
```

---

## Разделение датасета

```rust
// 70% train, 15% val, 15% test
fn split_dataset(data: Vec<(Matrix, f32)>) -> (Vec<_>, Vec<_>, Vec<_>) {
    let mut rng = rand::thread_rng();
    let mut shuffled = data;
    shuffled.shuffle(&mut rng);

    let n = shuffled.len();
    let train_end = (n as f32 * 0.7) as usize;
    let val_end = (n as f32 * 0.85) as usize;

    let train = shuffled[..train_end].to_vec();
    let val = shuffled[train_end..val_end].to_vec();
    let test = shuffled[val_end..].to_vec();

    (train, val, test)
}
```

---

## Дальше

- [03-training.md](03-training.md) — обучаем на burn