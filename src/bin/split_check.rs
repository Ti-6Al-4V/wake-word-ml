// Проверка разбиения датасета: размеры сплитов, баланс классов
// и главное — отсутствие утечки (один исходник не должен попасть
// в два сплита сразу вместе со своими аугментациями).
//
// Запуск: cargo run --bin split_check

use std::collections::HashSet;
use wake_word_ml::dataset;

fn main() {
    let s = dataset::load("dataset/positive", "dataset/negative");

    let count = |v: &[dataset::Sample], label: f32| v.iter().filter(|x| x.label == label).count();
    println!("train: {:5} (позитивов {}, негативов {})", s.train.len(), count(&s.train, 1.0), count(&s.train, 0.0));
    println!("val:   {:5} (позитивов {}, негативов {})", s.val.len(), count(&s.val, 1.0), count(&s.val, 0.0));
    println!("test:  {:5} (позитивов {}, негативов {})", s.test.len(), count(&s.test, 1.0), count(&s.test, 0.0));

    // Утечка: ключи групп train не должны встречаться в val/test.
    let keys = |v: &[dataset::Sample]| -> HashSet<String> {
        v.iter().map(|x| dataset::group_key(&x.path)).collect()
    };
    let (kt, kv, kte) = (keys(&s.train), keys(&s.val), keys(&s.test));
    let mut leaked = 0;
    for k in &kt {
        if kv.contains(k) || kte.contains(k) {
            println!("УТЕЧКА: группа {k} в нескольких сплитах!");
            leaked += 1;
        }
    }
    for k in &kv {
        if kte.contains(k) {
            println!("УТЕЧКА: группа {k} в val и test!");
            leaked += 1;
        }
    }
    if leaked == 0 {
        println!("\nУтечек нет: все аугментации исходника живут в одном сплите.");
    }
}
