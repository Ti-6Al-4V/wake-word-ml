# Wake Word ML

Обучение custom wake word модели для ESP32-S3.

Цель: натренировать нейросеть распознавать слово "гермес" офлайн на микроконтроллере. Понять как это работает под капотом. Внедрить в CapAI.

## Стек

- **Training:** Python, TensorFlow/Keras, GPU (или CPU)
- **Датасет:** записи "гермес" + негативные образцы
- **Деплой:** TensorFlow Lite Micro на ESP32-S3
- **Интеграция:** CapAI firmware

## Документация

| Файл | Описание |
|------|----------|
| [docs/01-theory.md](docs/01-theory.md) | Теория: звук → спектрограмма → нейросеть → классификация |
| [docs/02-dataset.md](docs/02-dataset.md) | Сбор и подготовка датасета |
| [docs/03-training.md](docs/03-training.md) | Обучение модели |
| [docs/04-deploy.md](docs/04-deploy.md) | Конвертация в TFLite Micro, деплой на ESP32 |
| [docs/05-integration.md](docs/05-integration.md) | Интеграция в CapAI |

## Пайплайн

```
"Гермес" (голос)
    │
    ▼
INMP441 → I2S → PCM 16kHz 16-bit
    │
    ▼
MFCC: извлечение признаков (20 коэффициентов × N кадров)
    │
    ▼
CNN модель (~200 KB, .tflite)
    │
    ▼
binary: wake word / не wake word
    │
    ▼
Если wake → CapAI просыпается
```

## Связь с CapAI

Этот репозиторий — модуль wake word для [CapAI](https://github.com/Ti-6Al-4V/cap-ai).
Готовая `.tflite` модель + ESP32 inference код → интегрируется в firmware CapAI.

MIT