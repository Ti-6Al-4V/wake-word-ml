# Интеграция в CapAI

Готовые веса + C++ inference код → firmware CapAI.

---

## Что забираем из этого репо в CapAI

```
wake-word-ml/
├── esp32/mfcc.h               → cap-ai/firmware/src/mfcc.h
├── esp32/wake_word_inference.h → cap-ai/firmware/src/wake_word.h
└── models/wake_word.bin        → cap-ai/firmware/data/wake_word_weights.h (через xxd -i)
```

---

## Изменения в CapAI firmware

### platformio.ini

```ini
[env:xiao_esp32s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
board_build.arduino.memory_type = qspi_opi
monitor_speed = 115200

lib_deps =
    espressif/esp32-camera
    bblanchon/ArduinoJson
    linksai/ArduinoWebsockets
    ; НЕТ tensorflow — ручной inference
    ; НЕТ esp-sr — своя модель

build_flags =
    -DCORE_DEBUG_LEVEL=3
    -DBOARD_HAS_PSRAM
    -DARDUINO_USB_CDC_ON_BOOT=1
    -DARDUINO_USB_MODE=1
```

### config.h

```cpp
#define WAKE_THRESHOLD  0.7
#define MFCC_NUM        20
#define NUM_FRAMES      33
#define FRAME_SIZE      480   // 30мс при 16kHz
```

### main.cpp

```cpp
#include "mfcc.h"
#include "wake_word.h"  // ручной CNN inference

TaskHandle_t wake_task_handle;

void wake_word_task(void* param) {
    float mfcc[NUM_FRAMES][MFCC_NUM];
    int frame_idx = 0;
    int16_t pcm[FRAME_SIZE];

    while (true) {
        size_t n;
        i2s_read(I2S_MIC_PORT, pcm, FRAME_SIZE * 2, &n, portMAX_DELAY);
        compute_mfcc_frame(pcm, mfcc[frame_idx]);
        frame_idx = (frame_idx + 1) % NUM_FRAMES;

        if (frame_idx % 10 == 0) {
            float score = wake_word_forward(mfcc);
            if (score > WAKE_THRESHOLD) {
                xTaskNotifyGive(main_task_handle);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup() {
    // ...
    xTaskCreatePinnedToCore(wake_word_task, "wake", 8192, NULL, 1, &wake_task_handle, 1);
}
```

### State machine (без изменений)

```
LIGHT_SLEEP + wake_word_task (Core 1)
    │ score > 0.7
    ▼
WAKING_UP → LISTENING → CAPTURING → RESPONDING → TEARDOWN
    ▼
LIGHT_SLEEP + wake_word_task снова
```

---

## План работы

1. Компоненты CapAI пришли → собираем, тестируем модули
2. Параллельно: собираем датасет "гермес" (Rust скрипт записи)
3. Обучаем модель на burn (~30 мин на CPU, меньше на GPU)
4. Экспорт весов: `cargo run --bin export` → `wake_word.bin`
5. Конвертация: `xxd -i wake_word.bin > wake_word_weights.h`
6. Тест inference на ESP32 (forward pass, замер скорости)
7. Интеграция в CapAI firmware
8. Тест на кепке: говорим "гермес" → просыпается

Критерии готовности и метрики (false accepts/hour, тестовые наборы) —
в [07-quality.md](07-quality.md). Без них пункт 8 будет работать только дома в тишине.