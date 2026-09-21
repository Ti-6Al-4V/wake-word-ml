//! Общие аудио-утилиты пайплайна: чтение/запись WAV, дизер, подгонка
//! длины окна. Раньше каждый бинарь носил свою копию write_wav —
//! теперь одна.
//!
//! Короткие файлы дополняются синтетическим шумом: peak −60 dBFS,
//! RMS около −64.8 dBFS. Это выбранный способ уменьшить артефакт
//! цифрового padding, а не измеренная модель микрофона. Сам padding
//! тоже может стать признаком источника; проверяй оба класса и live-test.

use std::path::Path;

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Амплитуда дизера: 10^(−60/20) = 0.001.
pub const DITHER_AMP: f32 = 0.001;

/// Читает WAV 16-bit → (сэмплы в [−1, 1], частота). Многоканальный
/// файл сводится в моно усреднением.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err("нужен WAV PCM16 (ffmpeg: -c:a pcm_s16le)".into());
    }
    let ch = spec.channels as usize;
    if ch == 0 || spec.sample_rate == 0 {
        return Err("некорректное число каналов или частота WAV".into());
    }
    let raw: Vec<f32> = reader.samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|s| s as f32 / 32768.0)
        .collect();
    if raw.len() % ch != 0 {
        return Err("неполный многоканальный кадр WAV".into());
    }
    let mono: Vec<f32> = if ch <= 1 {
        raw
    } else {
        raw.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Пишет WAV 16-bit моно. Сэмплы ожидаются в [−1, 1]; clamp — страховка.
pub fn write_wav(path: &Path, samples: &[f32], rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .unwrap_or_else(|e| panic!("не создать {}: {e}", path.display()));
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

/// Детерминированный ГПСЧ для файла: сид = хэш (имя, тег).
/// Один и тот же файл + тег → один и тот же шум при любом запуске.
pub fn rng_for(name: &str, tag: u64) -> ChaCha8Rng {
    let mut h = DefaultHasher::new();
    (name, tag).hash(&mut h);
    ChaCha8Rng::seed_from_u64(h.finish())
}

/// Один сэмпл дизера.
pub fn dither(rng: &mut impl Rng) -> f32 {
    rng.random_range(-DITHER_AMP..DITHER_AMP)
}

/// Довести сигнал до длины `len`: длинный обрезать, короткий дополнить
/// ДИЗЕРОМ (не нулями — см. шапку файла). `offset` — куда положить
/// начало сигнала внутри окна (0 = как раньше, слева).
pub fn fit_to_len(samples: &[f32], len: usize, offset: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut out: Vec<f32> = (0..len).map(|_| dither(rng)).collect();
    let offset = offset.min(len);
    let n = samples.len().min(len - offset);
    out[offset..offset + n].copy_from_slice(&samples[..n]);
    out
}

/// Пиковая нормализация: масштабировать так, чтобы |max| = `peak`.
/// Возвращает None для тишины (нормализовать нечего).
pub fn normalize_peak(samples: &[f32], peak: f32) -> Option<Vec<f32>> {
    let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max < 1e-4 {
        return None;
    }
    let gain = peak / max;
    Some(samples.iter().map(|s| s * gain).collect())
}

/// Среднеквадратичный уровень.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Границы речи в сигнале по энергии: первый и последний кадр (10мс),
/// чья RMS выше `frac` от максимальной. Нужно аугментации сдвига:
/// чтобы двигать слово по окну, надо знать, где оно.
pub fn speech_bounds(samples: &[f32], rate: u32, frac: f32) -> (usize, usize) {
    let frame = (rate / 100) as usize; // 10мс
    if frame == 0 || samples.len() < frame {
        return (0, samples.len());
    }
    let energies: Vec<f32> = samples.chunks(frame).map(rms).collect();
    let max = energies.iter().cloned().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return (0, samples.len());
    }
    let thr = max * frac;
    let first = energies.iter().position(|&e| e >= thr).unwrap_or(0);
    let last = energies.iter().rposition(|&e| e >= thr).unwrap_or(energies.len() - 1);
    (first * frame, ((last + 1) * frame).min(samples.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_reader_downmixes_pcm16_and_rejects_float() {
        let path = std::env::temp_dir().join(format!(
            "wake-word-audio-test-{}-{}.wav", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let spec = hound::WavSpec {
            channels: 2, sample_rate: 16_000, bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in [16384_i16, -16384, 8192, 8192] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let pcm = read_wav(&path);
        let mut writer = hound::WavWriter::create(&path, hound::WavSpec {
            channels: 1, bits_per_sample: 32, sample_format: hound::SampleFormat::Float,
            ..spec
        }).unwrap();
        writer.write_sample(0.5_f32).unwrap();
        writer.finalize().unwrap();
        let float = read_wav(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(pcm.unwrap(), (vec![0.0, 0.25], 16_000));
        assert!(float.unwrap_err().contains("PCM16"));
    }

    #[test]
    fn fit_pads_with_dither_not_zeros() {
        let mut rng = rng_for("x", 1);
        let out = fit_to_len(&[0.5, -0.5], 10, 3, &mut rng);
        assert_eq!(out.len(), 10);
        assert_eq!(out[3], 0.5);
        assert_eq!(out[4], -0.5);
        assert!(out[..3].iter().chain(&out[5..]).all(|v| v.abs() <= DITHER_AMP));
        assert!(out[..3].iter().chain(&out[5..]).any(|v| *v != 0.0));
    }

    #[test]
    fn fit_is_deterministic() {
        let a = fit_to_len(&[0.1], 100, 0, &mut rng_for("f", 2));
        let b = fit_to_len(&[0.1], 100, 0, &mut rng_for("f", 2));
        assert_eq!(a, b);
    }

    #[test]
    fn speech_bounds_finds_burst() {
        let rate = 1000; // кадр 10 сэмплов
        let mut s = vec![0.0f32; 300];
        for v in &mut s[100..180] { *v = 0.8; }
        let (a, b) = speech_bounds(&s, rate, 0.1);
        assert_eq!((a, b), (100, 180));
    }
}
