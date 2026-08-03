# Датасет

Сбор и подготовка данных на Rust. Позитивные — TTS + свой голос, негативные — готовые датасеты.

---

## Источники

| Категория | Источник | Кол-во |
|-----------|----------|--------|
| Позитивные (синтетические) | TTS генерация "гермес" разными голосами | ~500 |
| Позитивные (реальные) | Запись своим голосом через INMP441 | ~100 |
| Позитивные (аугментация) | pitch/time/noise/volume ×6 | ~3600 |
| Негативные (русская речь) | [Golos](https://github.com/sberdevices/golos) (Сбер, 1240ч) | ~2000 |
| Негативные (короткие слова) | [Google Speech Commands](https://research.google/blog/launching-the-speech-commands-dataset/) | ~2000 |
| Фон | Тишина, музыка, улица | ~500 |

---

## Позитивные: TTS генерация

```rust
// src/generate_tts.rs
// Генерируем "гермес" через Edge TTS (через subprocess) или kokoro-rs

use std::process::Command;

const VOICES: &[&str] = &[
    "ru-RU-SvetlanaNeural",
    "ru-RU-DmitriNeural",
];

const SPEEDS: &[f32] = &[0.8, 0.9, 1.0, 1.1, 1.2];

fn generate_tts(output_dir: &str) {
    let mut idx = 1;
    for voice in VOICES {
        for &speed in SPEEDS {
            let filename = format!("{}/germes_tts_{:03}.wav", output_dir, idx);
            // edge-tts через CLI
            Command::new("edge-tts")
                .args([
                    "--text", "гермес",
                    "--voice", voice,
                    "--rate", &format!("{}%", (speed * 100.0) as i32),
                    "--write-media", &filename,
                ])
                .status()
                .unwrap();
            idx += 1;
        }
    }
    println!("Generated {} TTS samples", idx - 1);
}
```

## Позитивные: запись своим голосом

```rust
// src/record.rs
// Запись с микрофона, 1 сек на каждое произношение

use hound::{WavWriter, WavSpec};

fn record_word(output_dir: &str, count: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    for i in 1..=count {
        println!("Запись {}/{} — скажи \"гермес\"", i, count);

        let filename = format!("{}/germes_real_{:03}.wav", output_dir, i);
        let mut writer = WavWriter::create(&filename, spec).unwrap();

        // cpal захват 1 сек аудио → writer.write_sample()
        // ...

        println!("Сохранено: {}", filename);
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}
```

## Аугментация

Из 600 реальных+TTS → 3600 аугментированных:

```rust
// src/augment.rs
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

## Негативные: Golos

```rust
// src/download_golos.rs
// Скачиваем небольшой кусок Golos dataset, вырезаем случайные 1-сек фрагменты

use std::process::Command;

fn download_golos(output_dir: &str, count: u32) {
    // Скачиваем test.tar (1.3 GB) с openslr.org/114/
    let url = "https://openslr.org/resources/114/test.tar";

    Command::new("wget")
        .args(["-q", url, "-O", "/tmp/golos_test.tar"])
        .status()
        .unwrap();

    Command::new("tar")
        .args(["xf", "/tmp/golos_test.tar", "-C", "/tmp/golos/"])
        .status()
        .unwrap();

    // Вырезаем случайные 1-сек фрагменты из WAV файлов
    let mut idx = 1;
    for entry in std::fs::read_dir("/tmp/golos/test/wavs/").unwrap() {
        if idx > count { break; }
        let path = entry.unwrap().path();
        if path.extension().unwrap() == "wav" {
            // Загружаем, берём случайный 1-сек кусок, сохраняем
            let filename = format!("{}/negative_golos_{:04}.wav", output_dir, idx);
            extract_random_1sec(&path, &filename);
            idx += 1;
        }
    }
    println!("Extracted {} Golos samples", idx - 1);
}
```

## Негативные: Google Speech Commands

```rust
// src/download_speech_commands.rs
// Скачиваем Speech Commands v0.02, используем все слова как негативные

fn download_speech_commands(output_dir: &str) {
    let url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz";

    Command::new("wget")
        .args(["-q", url, "-O", "/tmp/speech_commands.tar.gz"])
        .status()
        .unwrap();

    Command::new("tar")
        .args(["xzf", "/tmp/speech_commands.tar.gz", "-C", "/tmp/speech_commands/"])
        .status()
        .unwrap();

    // Копируем все .wav files (yes, no, stop, go...) как негативные
    let mut idx = 1;
    for entry in walkdir::WalkDir::new("/tmp/speech_commands/") {
        let entry = entry.unwrap();
        if entry.path().extension().map_or(false, |e| e == "wav") {
            let filename = format!("{}/negative_sc_{:05}.wav", output_dir, idx);
            std::fs::copy(entry.path(), &filename).ok();
            idx += 1;
        }
    }
    println!("Copied {} Speech Commands samples", idx - 1);
}
```

---

## Preprocessing

Все WAV → 16kHz, mono, 1 сек:

```rust
// src/preprocess.rs
use hound::{WavReader, WavSpec, WavWriter};

pub fn preprocess(input: &str, output: &str) {
    let mut reader = WavReader::open(input).unwrap();
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

## MFCC извлечение

```rust
// src/mfcc.rs
use rustfft::{FftPlanner, num_complex::Complex};

pub const N_MFCC: usize = 20;
pub const FRAME_SIZE: usize = 480;
pub const HOP_SIZE: usize = 480;
pub const FFT_SIZE: usize = 512;
pub const NUM_FRAMES: usize = 33;

pub fn wav_to_mfcc(samples: &[f32], sr: u32) -> [[f32; N_MFCC]; NUM_FRAMES] {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let hamming: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
        .collect();

    let mel_filters = compute_mel_filterbank();
    let dct_matrix = compute_dct_matrix();

    let mut mfcc = [[0.0f32; N_MFCC]; NUM_FRAMES];

    for frame in 0..NUM_FRAMES {
        let start = frame * HOP_SIZE;
        let mut windowed = vec![Complex::new(0.0, 0.0); FFT_SIZE];
        for i in 0..FRAME_SIZE.min(samples.len() - start) {
            windowed[i] = Complex::new(samples[start + i] * hamming[i], 0.0);
        }

        fft.process(&mut windowed);
        let power: Vec<f32> = windowed[..FFT_SIZE/2].iter().map(|c| c.norm_sqr()).collect();

        let mut mel_energy = [0.0f32; N_MFCC];
        for m in 0..N_MFCC {
            for k in 0..FFT_SIZE/2 {
                mel_energy[m] += mel_filters[m][k] * power[k];
            }
        }
        for m in 0..N_MFCC { mel_energy[m] = (mel_energy[m] + 1e-8).ln(); }

        for m in 0..N_MFCC {
            for k in 0..N_MFCC {
                mfcc[frame][m] += dct_matrix[m][k] * mel_energy[k];
            }
        }
    }

    // Нормализация
    let mean: f32 = mfcc.iter().flat_map(|r| r.iter()).sum::<f32>() / (NUM_FRAMES * N_MFCC) as f32;
    let std = (mfcc.iter().flat_map(|r| r.iter()).map(|v| (v - mean).powi(2)).sum::<f32>() / (NUM_FRAMES * N_MFCC) as f32).sqrt() + 1e-8;
    for frame in 0..NUM_FRAMES {
        for m in 0..N_MFCC { mfcc[frame][m] = (mfcc[frame][m] - mean) / std; }
    }

    mfcc
}
```

---

## Структура датасета

```
dataset/
├── positive/
│   ├── germes_tts_001.wav      # TTS синтетические
│   ├── germes_tts_002.wav
│   ├── germes_real_001.wav     # свой голос
│   ├── germes_aug_0001.wav     # аугментированные
│   └── ... (~3700)
├── negative/
│   ├── negative_golos_0001.wav # русская речь (Golos)
│   ├── negative_sc_00001.wav   # короткие слова (Speech Commands)
│   └── ... (~4000)
└── background/
    ├── silence_001.wav
    ├── street_001.wav
    └── ... (~500)
```

---

## Разделение

70% train / 15% val / 15% test. Стратифицированное (сохраняем пропорцию позитив/негатив).

---

## Дальше

- [03-training.md](03-training.md) — обучаем на burn