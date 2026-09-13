// Препроцессинг аудио к единому формату обучения:
// WAV 16kHz, 16-bit, моно, фиксированное окно (по умолчанию 1.2с),
// пиковая нормализация до 0.9. Короткие файлы дополняются ДИЗЕРОМ
// (шум −60 дБFS), а не нулями — почему, см. src/audio.rs.
//
// Понимает и WAV, и MP3: декодирование и ресемплинг делегируем ffmpeg
// (единственная внешняя зависимость; качество ресемпла у него отличное).
// Идемпотентен: если выходной файл уже есть — пропускает (можно
// дозаписывать дубли и просто перезапускать).
//
// Запуск:  cargo run --bin preprocess -- <вход> <выход> [окно_сек]
// Примеры: cargo run --bin preprocess -- dataset/raw/real dataset/positive
//          cargo run --bin preprocess -- dataset/raw/tts  dataset/positive

use std::path::{Path, PathBuf};
use std::process::Command;

use wake_word_ml::audio;

const TARGET_RATE: u32 = 16000; // целевая частота всего датасета
const NORM_PEAK: f32 = 0.9;     // до какого пика нормализуем (запас до клиппинга)

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Использование: preprocess <вход> <выход> [окно_сек]");
        eprintln!("Пример: cargo run --bin preprocess -- dataset/raw/real dataset/positive");
        std::process::exit(1);
    }
    let in_dir = &args[1];
    let out_dir = &args[2];
    // Третий аргумент — окно в секундах. 1.2с = окно записи рекордера
    // (слово ~0.6-0.8с + запас по краям). parse: некорректное значение → дефолт.
    let window_secs: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.2);
    let window_samples = (TARGET_RATE as f32 * window_secs) as usize;

    // ffmpeg обязателен: проверяем до того, как что-то делать.
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg не найден в PATH. Установка: brew install ffmpeg");
        std::process::exit(1);
    }

    std::fs::create_dir_all(out_dir).expect("не создать выходную папку");

    // Собираем входные файлы: wav и mp3, сортируем для стабильного порядка.
    let mut files: Vec<PathBuf> = std::fs::read_dir(in_dir)
        .unwrap_or_else(|e| panic!("не открыть {in_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(
            p.extension().and_then(|x| x.to_str()).map(|x| x.to_lowercase()).as_deref(),
            Some("wav") | Some("mp3")
        ))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("В {in_dir} нет .wav/.mp3 файлов");
        std::process::exit(1);
    }

    // Один временный файл на всю сессию: файлы обрабатываются последовательно.
    let tmp = Path::new(out_dir).join(".tmp_decode.wav");

    let (mut done, mut skipped, mut silent, mut failed) = (0usize, 0usize, 0usize, 0usize);
    for path in &files {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let out_path = Path::new(out_dir).join(format!("{stem}.wav"));

        // Идемпотентность: уже обработан — пропускаем.
        if out_path.exists() {
            skipped += 1;
            continue;
        }

        // ffmpeg: любой формат → WAV 16kHz моно.
        // -loglevel error — молчит, если всё хорошо; -y — перезаписать tmp.
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(path)
            .args(["-ac", "1", "-ar", &TARGET_RATE.to_string(), "-f", "wav"])
            .arg(&tmp)
            .status();
        if !status.map_or(false, |s| s.success()) {
            eprintln!("[ошибка ffmpeg] {}", path.display());
            failed += 1;
            continue;
        }

        // Читаем декодированный WAV (16kHz моно после ffmpeg).
        let (samples, _) = audio::read_wav(&tmp).expect("tmp wav не читается");
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        // Пиковая нормализация: делим на максимум и тянем до NORM_PEAK.
        // Это убирает разницу громкостей между дублями (усталость голоса,
        // расстояние до микрофона) — модель учит слово, а не громкость.
        let Some(normalized) = audio::normalize_peak(&samples, NORM_PEAK) else {
            // Тишина: нормализовать нечего, в датасет не годится.
            eprintln!("[тишина, пропуск] {}", path.display());
            silent += 1;
            continue;
        };

        // Фиксированное окно: длинные файлы обрезаются, короткие
        // дополняются справа дизером (детерминированным: сид = имя файла).
        // Длинные записи режь заранее бинарём chop — здесь берётся начало.
        let mut rng = audio::rng_for(stem, 0);
        let window = audio::fit_to_len(&normalized, window_samples, 0, &mut rng);

        audio::write_wav(&out_path, &window, TARGET_RATE);
        done += 1;
        println!("[{done}] {stem}.wav (пик был {peak:.2})");
    }
    let _ = std::fs::remove_file(&tmp);

    println!("\nГотово: обработано {done}, пропущено готовых {skipped}, \
              тишины {silent}, ошибок {failed}. Выход: {out_dir}");
}
