# Обучение и оценка: справочник по коду

Актуальный код: Burn 0.21, MFCC 40×20. Порядок изучения —
[00-learning-path](00-learning-path.md), практика — [10-labs](10-labs.md).
Все команды обучения ниже работают с текущими `dataset/positive` и
`dataset/negative`; экспорт и событийная разметка ещё не реализованы.

## Файлы

| Файл | Что |
|---|---|
| `src/model.rs` | `HermesNet` — архитектура, одна для train/eval/export. `sigmoid`, `batch_tensor` |
| `src/dataset.rs` | сплит по дикторам: train/val 85/15 + holdout-test (без holdout — 70/15/15), `features_flat` |
| `src/mfcc.rs` | признаки: 40×20, log-пол, CMVN по коэффициенту |
| `src/bin/train.rs` | ручной цикл обучения, кэш признаков, чекпоинт, CSV-лог |
| `src/bin/eval.rs` | кривая порогов, score по папке, стриминг FA/час |
| `src/bin/export.rs` | заглушка (этап 8) |

## Модель

```
[b, 1, 40, 20]
 → Conv2d(1→16, 3×3, Valid) → ReLU → MaxPool 2×2   → [b, 16, 19, 9]
 → Conv2d(16→8, 3×3, Valid) → ReLU → MaxPool 2×2   → [b, 8, 8, 3]
 → reshape [b, 192] → Dropout(p) → Linear(192→16) → ReLU → Linear(16→1)
 → reshape [b]  — ЛОГИТ, сигмоида снаружи
```

Параметров 4425. Dropout с p = 0 — тождественный (по умолчанию).

```rust
let model = HermesNet::<Autodiff<Wgpu>>::new(&device, dropout);
let logits: Tensor<B, 1> = model.forward(x);   // x: Tensor<B, 4> = [b,1,40,20]
```

## Цикл обучения (`train.rs`)

```
аргументы: [эпохи=10] [батч=64] [lr=0.001] [dropout=0] [weight_decay=0]
make train E=30 BATCH=64 LR=0.001 DROPOUT=0.2 WD=0.0001

Backend::seed(&device, 42)                    — воспроизводимая инициализация
splits = dataset::load(positive, negative)    — train/val + holdout-test
train/val признаки → кэш в памяти             — MFCC один раз
for epoch:
    перемешать индексы (сид 42+epoch)
    for batch:
        logits = model.forward(x)
        loss   = BCEWithLogits(logits, y)
        grads  = GradientsParams::from_grads(loss.backward(), &model)
        model  = adam.step(lr, model, grads)
    val: loss, acc, TPR, FA при пороге 0.5 → stdout + models/train_log.csv
    если val loss лучший → save models/hermes.bin
save models/hermes_last.bin
```

Выходы:

| Файл | Что |
|---|---|
| `models/hermes.bin` | веса лучшей по val loss эпохи — их читает eval/export |
| `models/hermes_last.bin` | веса последней эпохи |
| `models/train_log.csv` | `epoch,train_loss,val_loss,val_acc,val_tpr,val_fa` |

## Оценка (`eval.rs`)

```
make eval [SPLIT=val|test]      кривая порогов 0.05..0.95: TPR, FA-доля, FN, FP;
                                самые уверенные ошибки по именам файлов
make score DIR=<папка> [T=0.5]  гистограмма score по любым wav 16kHz/1.2с
make stream FILE=<wav> [HOP=100] скользящее окно по длинной 16kHz-записи:
                                FA/час при каждом пороге (2 окна подряд, cooldown 2с)
```

Загрузка модели:

```rust
HermesNet::<Wgpu>::new(&device, 0.0)
    .load_file("models/hermes", &BinFileRecorder::<FullPrecisionSettings>::new(), &device)
```

## burn 0.21: что отличается от старых примеров в сети

Большинство туториалов написаны под 0.13–0.14. Что пришлось менять:

| Было (0.14) | Стало (0.21) |
|---|---|
| `burn::nn::conv::Conv2d<B, N>` (const N) | `Conv2d<B>` без параметра |
| `x.max_pool_2d(...)` на тензоре | модуль `MaxPool2dConfig::new([2,2]).init()` |
| `LinearConfig::new(in, out).init()` | `.init(device)` |
| `loss.init()` | `BinaryCrossEntropyLossConfig::new().with_logits(true).init::<B>(&device)` |
| `B::seed(42)` | `B::seed(&device, 42)` |
| `tensor.to_data().value` | `tensor.into_data().into_vec::<f32>()` |
| `WgpuDevice::default()` | `WgpuDevice::DefaultDevice` |
| lr как `f32` | `LearningRate = f64` |
| `AdamConfig::new().with_learning_rate(lr)` | lr передаётся в `optim.step(lr, model, grads)` |
| `model.valid()` | то же — `AutodiffModule::valid()` даёт модель на бэкенде без autodiff |

Weight decay: `AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(penalty)))`,
путь `burn::optim::decay::WeightDecayConfig`.

## Ограничения и сохранение экспериментов

Train проверяет наличие обоих классов в train/val и конечность loss.
Val BCE считается прямо из логитов без clamp, train loss усредняется
по примерам, включая неполный последний батч. Train loss накоплен по
меняющимся весам; val loss измерен на checkpoint конца эпохи.

Каждый запуск начинает обучение заново. Перед следующим запуском сохрани
оба checkpoint и CSV в отдельную папку:

```bash
mkdir -p models/runs/baseline-01
cp models/hermes.bin models/hermes_last.bin models/train_log.csv models/runs/baseline-01/
```

Выбери новое имя для следующего опыта. Сохрани рядом карточку из
[шаблона](experiment-template.md), manifests и код. Одинаковый seed
не гарантирует побитной воспроизводимости GPU или прежнего split после
добавления файлов. Размер подмножества train и manifest пока не являются
флагами CLI — это задания L5/L6.

Eval показывает кривую и диагностические ошибки при 0.5, даже если
рабочий порог другой. На test оценивай заранее выбранную рабочую точку.
Clip TPR взвешен по файлам, включая аугментации; число групп не доказывает
независимость сессий. Stream требует >=1.2с, PCM16 16kHz, шаг 1..2000мс;
гейт выключен, положительные события и задержка автоматически не оцениваются.

## Целевые метрики

Из [07-quality](07-quality.md): event recall ≥95% вместе с FA/h ≤0.1
на отдельных потоках при фиксированных параметрах детектора. Указывай
объёмы, условия и интервалы. Clip eval — промежуточная диагностика.

---

## Дальше

- [04-deploy.md](04-deploy.md) — экспорт и ручной C++ inference
