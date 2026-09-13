// Нарезка длинных записей на окна 1.2с — для негативов своим голосом.
//
// Зачем: все живые позитивы записаны одним человеком в одну гарнитуру,
// а все негативы (Golos, Speech Commands) — чужими людьми в чужие
// микрофоны. Самый простой признак, разделяющий классы, — «это Жека
// в JBL», и слово при этом не нужно вовсе. Val этого не покажет: он
// устроен так же. Лечение — негативы тем же голосом в тот же микрофон:
// читаешь любой текст (make record_speech), этот бинарь режет.
//
// Что делает: каждый WAV из входной папки → окна по 1.2с без
// перекрытия на НАТИВНОЙ частоте файла (48kHz гарнитуры). Тихие окна
// (RMS ниже порога — паузы между фразами) выбрасываются. Дальше —
// обычный make preprocess IN=<выход> OUT=dataset/negative.
//
// Имена: <исходник>_c0001.wav, _c0002... Сплит группирует все куски
// одной сессии вместе (суффикс _cN, см. src/dataset.rs).
//
// Запуск: cargo run --bin chop -- <вход> <выход> [окно_сек] [rms_min]
// Пример: cargo run --bin chop -- dataset/raw/self_speech dataset/raw/self_speech_clips

use std::path::{Path, PathBuf};

use wake_word_ml::audio;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Использование: chop <вход> <выход> [окно_сек=1.2] [rms_min=0.01]");
        std::process::exit(1);
    }
    let in_dir = &args[1];
    let out_dir = &args[2];
    let window_secs: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.2);
    let rms_min: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.01);
    std::fs::create_dir_all(out_dir).expect("не создать выходную папку");

    let mut files: Vec<PathBuf> = std::fs::read_dir(in_dir)
        .unwrap_or_else(|e| panic!("не открыть {in_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "wav"))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("В {in_dir} нет .wav. Сначала: make record_speech");
        std::process::exit(1);
    }

    let (mut total, mut quiet, mut skipped) = (0usize, 0usize, 0usize);
    for path in &files {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let (samples, rate) = match audio::read_wav(path) {
            Ok(r) => r,
            Err(e) => { eprintln!("[не читается] {}: {e}", path.display()); continue; }
        };
        let window = (rate as f32 * window_secs) as usize;
        // Пиковая нормализация всей записи ДО порога RMS: иначе порог
        // зависит от того, насколько громко записалось.
        let Some(samples) = audio::normalize_peak(&samples, 0.9) else {
            eprintln!("[тишина, пропуск] {}", path.display());
            continue;
        };

        let mut n = 0usize;
        for (ci, chunk) in samples.chunks(window).enumerate() {
            // Хвост короче половины окна — выбрасываем.
            if chunk.len() < window / 2 { break; }
            if audio::rms(chunk) < rms_min { quiet += 1; continue; }
            let out_path = Path::new(out_dir).join(format!("{stem}_c{:04}.wav", ci + 1));
            if out_path.exists() { skipped += 1; continue; }
            let mut rng = audio::rng_for(stem, ci as u64);
            let full = audio::fit_to_len(chunk, window, 0, &mut rng);
            audio::write_wav(&out_path, &full, rate);
            n += 1;
        }
        total += n;
        println!("{stem}: {:.0}с → {n} окон", samples.len() as f32 / rate as f32);
    }
    println!("\nГотово: {total} окон, тихих пропущено {quiet}, уже было {skipped}. Выход: {out_dir}");
    println!("Дальше: make preprocess IN={out_dir} OUT=dataset/negative");
}
