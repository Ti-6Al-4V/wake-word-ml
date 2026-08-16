// Генерация позитивных примеров слова "Гермес" через edge-tts.
// Выход: dataset/raw/tts/*.mp3 (потом конвертируем в WAV 16kHz)

use std::fs;
use std::process::Command;

// Три интонации: точка, восклицание, вопрос — edge-tts меняет просодию
// от знаков препинания. Модели полезно видеть слово в разных интонациях.
const TEXTS: &[&str] = &["Гермес", "Гермес!", "Гермес?"];

// Скорости речи в процентах. edge-tts принимает от -50% до +50%.
const RATES: &[i32] = &[-25, -10, 0, 10, 25];

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dataset/raw/tts".to_string());
    fs::create_dir_all(&out_dir).expect("не удалось создать выходную папку");

    let voices = discover_voices();
    if voices.is_empty() {
        eprintln!("Не нашёл ни одного голоса ru-RU. Проверь: edge-tts --list-voices");
        std::process::exit(1);
    }
    println!("Голосов ru-RU: {}", voices.len());

    let (mut ok, mut skipped, mut failed) = (0u32, 0u32, 0u32);

    for (ti, text) in TEXTS.iter().enumerate() {
        for voice in &voices {
            for rate in RATES {
                // Имя файла: текст_голос_скорость, без лишних символов
                let voice_short = voice.strip_prefix("ru-RU-").unwrap_or(voice);
                let rate_tag = match rate {
                    0 => "norm".to_string(),
                    r if *r > 0 => format!("plus{r}"),
                    r => format!("minus{}", -r),
                };
                let file = format!("{}/germes_t{ti}_{voice_short}_{rate_tag}.mp3", out_dir);

                // Идемпотентность: уже есть непустой файл — пропускаем
                if fs::metadata(&file).map_or(false, |m| m.len() > 0) {
                    skipped += 1;
                    continue;
                }

                // edge-tts принимает скорость как "+25%" / "-10%" / "+0%"
                let rate_str = if *rate >= 0 { format!("+{rate}%") } else { format!("{rate}%") };

                let rate_arg = format!("--rate={rate_str}");
                let status = Command::new("edge-tts")
                    .args(["--text", text, "--voice", voice, &rate_arg, "--write-media", &file])
                    .status()
                    .expect("edge-tts не запустился — есть ли он в PATH?");

                // Проверяем не только exit code, но и что файл реально непустой:
                // edge-tts при сбоях API может создать пустой файл
                let good = status.success()
                    && fs::metadata(&file).map_or(false, |m| m.len() > 1000);

                match good {
                    true => ok += 1,
                    false => { failed += 1; eprintln!("FAIL: {file}"); }
                }
            }
        }
    }

    println!("Готово: сгенерировано {ok}, пропущено {skipped}, ошибок {failed}");
}

// Все голоса ru-RU из вывода `edge-tts --list-voices`.
// Не привязываемся к точному формату вывода: ищем любой токен,
// содержащий "ru-RU-", в любой строке.
fn discover_voices() -> Vec<String> {
    let output = Command::new("edge-tts")
        .args(["--list-voices"])
        .output()
        .expect("edge-tts не запустился");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            line.split_whitespace()
                .find(|tok| tok.contains("ru-RU-"))
                .map(|tok| tok.to_string())
        })
        .collect()
}