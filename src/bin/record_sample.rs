// Запись длинного сэмпла своего голоса для будущего voice cloning'а (Qwen TTS).
// В отличие от record.rs (окна по 1.2с под отдельные дубли), здесь —
// ОДИН непрерывный файл на всю длительность: клону нужна связная речь,
// а не отдельные слова.
//
// Запуск:  cargo run --bin record_sample -- [выход.wav] [секунды]
// Пример:  cargo run --bin record_sample -- dataset/raw/clone_sample.wav 60

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

fn main() {
    let out_path = std::env::args().nth(1)
        .unwrap_or_else(|| "dataset/raw/clone_sample.wav".to_string());
    let secs: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);

    // Папку под выходом создаём на случай нестандартного пути.
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(dir).expect("не создать папку");
    }

    let host = cpal::default_host();
    let device = host.default_input_device().expect("входное устройство не найдено");
    let config = device.default_input_config().expect("нет дефолтной конфигурации входа");
    let sample_rate = config.sample_rate();
    let channels = config.channels() as usize;
    println!("Устройство: {device}");
    println!("Частота: {sample_rate} Hz, каналов: {channels}, длительность: {secs}с");

    // Что читать. Разные фонемы и интонации дают клону больше материала;
    // читать ровным естественным голосом, не «выступать».
    println!("\nЧитай вслух, спокойно и естественно (микрофон у подбородка):");
    println!("  1. Сегодня утром я вышел из дома и поймал такси до работы.");
    println!("  2. Вечером мы готовили ужин: рыба, овощи и горячий чай с мёдом.");
    println!("  3. Погода вчера была странная — то солнце, то внезапный дождь.");
    println!("  4. В субботу планирую длинную прогулку по набережной с друзьями.");
    println!("  5. Если останется время, посмотрим старый фильм или почитаем.");
    println!("\nСтарт через 3 секунды...");
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Общий буфер: колбэк аудио-потока пишет, main ждёт окончания.
    // Тот же приём Arc<Mutex<>>, что в record.rs.
    let total_samples = sample_rate as usize * secs as usize;
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(total_samples)));
    let buf_in = buf.clone();
    let ch = channels;

    let stream = device.build_input_stream(
        config.config(),
        // Складываем всё подряд; многоканальный вход усредняем в моно.
        move |data: &[f32], _| {
            let mut b = buf_in.lock().unwrap();
            for frame in data.chunks(ch) {
                if b.len() >= total_samples { break; }
                b.push(frame.iter().sum::<f32>() / ch as f32);
            }
        },
        |err| eprintln!("ошибка потока: {err}"),
        None,
    ).expect("не удалось открыть аудиопоток (проверь доступ к микрофону!)");

    stream.play().expect("play failed");
    println!("Пишу... читай текст.");

    // Раз в 5 секунд показываем остаток, чтобы было видно, что идёт запись.
    let mut left = secs;
    while left > 0 {
        let step = left.min(5);
        std::thread::sleep(std::time::Duration::from_secs(step));
        left -= step;
        if left > 0 {
            println!("  ...осталось {left}с");
        }
    }
    drop(stream);

    let samples = buf.lock().unwrap().clone();

    // Быстрый QC на месте: тихий сэмпл клону не пригодится.
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    write_wav(&out_path, &samples, sample_rate);
    println!("\nСохранено: {out_path} ({:.1}с, пик {peak:.2})", samples.len() as f32 / sample_rate as f32);
    if peak < 0.10 {
        println!("ВНИМАНИЕ: очень тихо. Подвинь микрофон ближе и перезапиши:");
        println!("  make clone_sample");
    } else if peak > 0.99 {
        println!("ВНИМАНИЕ: клиппинг. Отодвинь микрофон и перезапиши.");
    } else {
        println!("Уровень хороший. Сэмпл готов для клонирования.");
    }
}

// Запись WAV 16-bit моно — та же функция, что в record.rs.
fn write_wav(path: &str, samples: &[f32], rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}
