# Интеграция в CapAI

Готовая модель + inference код → firmware CapAI.

---

## Что забираем из этого репо в CapAI

```
wake-word-ml/
├── esp32/mfcc.h           → cap-ai/firmware/src/mfcc.h
├── esp32/inference.h      → cap-ai/firmware/src/wake_word.h
└── models/wake_word.tflite → cap-ai/firmware/data/wake_word.tflite
```

---

## Изменения в CapAI firmware

### platformio.ini

```ini
lib_deps =
    espressif/esp32-camera
    tensorflow-lite-micro-esp32    # добавляется
    bblanchon/ArduinoJson
    linksai/ArduinoWebsockets

build_flags =
    -DCORE_DEBUG_LEVEL=3
    -DBOARD_HAS_PSRAM
    -DARDUINO_USB_CDC_ON_BOOT=1
    -DARDUINO_USB_MODE=1
```

ESP-SR (WakeNet) убирается — заменяется на свою модель.

### config.h

```cpp
#define WAKE_THRESHOLD  0.7
#define MFCC_NUM        20
#define NUM_FRAMES      33
#define FRAME_SIZE      480
#define HOP_SIZE        480
```

### main.cpp

```cpp
#include "wake_word.h"  // вместо wakenet.h

// Core 1: wake word detection (своя модель)
TaskHandle_t wake_task_handle;

void wake_word_task(void* param) {
    WakeWordDetector detector;
    detector.init();
    // ... цикл из 04-deploy.md
}

void setup() {
    // ...
    xTaskCreatePinnedToCore(
        wake_word_task, "wake", 8192, NULL, 1, &wake_task_handle, 1
    );
}
```

### State machine (без изменений)

```
LIGHT_SLEEP + wake word task (Core 1)
    │ wake word detected (score > threshold)
    ▼
WAKING_UP (WiFi, WS connect) — Core 0
    ▼
LISTENING → CAPTURING → RESPONDING → TEARDOWN
    ▼
LIGHT_SLEEP + wake word task снова
```

---

## Сравнение: ESP-SR WakeNet vs своя модель

| | WakeNet (Espressif) | Своя TFLite |
|---|---|---|
| Модель | .wn9 (проприетарный формат) | .tflite (открытый) |
| Обучение | сервис Espressif (1-2 нед) | сами (~1 день) |
| Размер | ~500 KB | ~10-30 KB |
| Inference | ~30мс | ~20-30мс |
| Потребление | ~40 mA | ~35-45 mA |
| Точность | ~95% | ~90-93% |
| Кастомизация | ограничена | полная |
| Зависимости | ESP-SR библиотека | TFLite Micro |

---

## План работы

1. Компоненты CapAI пришли → собираем, тестируем модули
2. Параллельно: собираем датасет "гермес" (этот репо)
3. Обучаем модель (~1 день на GPU/CPU)
4. Конвертируем в .tflite (int8 quantized)
5. Тест inference на ESP32 (с моделью в Flash)
6. Интегрируем в CapAI firmware
7. Тест на кепке: говорим "гермес" → просыпается