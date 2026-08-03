# Wake Word ML

Обучение custom wake word модели для ESP32-S3 на Rust.

Цель: натренировать нейросеть распознавать слово "гермес" офлайн на микроконтроллере. Понять как это работает под капотом. Внедрить в CapAI.

## Стек

- **Язык:** Rust
- **Обучение:** [burn](https://github.com/tracel-ai/burn) — Rust ML framework
- **Аудио:** `rustfft`, `hound` (WAV I/O), ручной MFCC
- **Деплой:** ручной C++ CNN inference на ESP32-S3 (модель ~5K параметров, TFLite не нужен)
- **Интеграция:** CapAI firmware

## Почему Rust

- Типобезопасность: модели и тензоры проверяются компилятором
- Скорость: нативный код, без Python overhead
- Один язык для ML + системного программирования
- `burn` — активно развивается, WGPU/CUDA backends, чистый API
- Для маленькой модели не нужен тяжёлый фреймворк — burn лёгкий

## Почему ручной C++ inference вместо TFLite

Модель крошечная: 2 conv слоя + 2 dense = ~5000 параметров, ~20 KB.
Ручной forward pass на C++ — ~200 строк. Никаких зависимостей, никаких рантаймов.
Полный контроль, полное понимание.

## Документация

| Файл | Описание |
|------|----------|
| [docs/01-theory.md](docs/01-theory.md) | Теория: звук → спектрограмма → нейросеть → классификация |
| [docs/02-dataset.md](docs/02-dataset.md) | Сбор и подготовка датасета на Rust |
| [docs/03-training.md](docs/03-training.md) | Обучение модели на burn |
| [docs/04-deploy.md](docs/04-deploy.md) | Экспорт весов, ручной C++ inference на ESP32 |
| [docs/05-integration.md](docs/05-integration.md) | Интеграция в CapAI |

## Пайплайн

```
"Гермес" (голос)
    │
    ▼
INMP441 → I2S → PCM 16kHz 16-bit
    │
    ▼
MFCC: извлечение признаков (20 коэффициентов × 33 кадров)
    │
    ▼
CNN (~5000 параметров, ~20 KB, бинарные веса)
    │
    ▼
binary: wake word / не wake word
    │
    ▼
Если wake → CapAI просыпается
```

## Связь с CapAI

Модуль wake word для [CapAI](https://github.com/Ti-6Al-4V/cap-ai).
Готовые веса (.bin) + C++ inference код → интегрируется в firmware CapAI.

MIT