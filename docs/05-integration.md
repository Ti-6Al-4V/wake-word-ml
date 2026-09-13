# Интеграция в CapAI

> Обновлено 2026-09-13: 40 кадров, стриминг с шагом 100мс и
> подтверждением (см. [04-deploy](04-deploy.md)). Прежний вариант
> с 33 кадрами и проверкой раз в 10 кадров устарел.

Готовые веса + C++ inference код → firmware CapAI.

---

## Что забираем из этого репо в CapAI

```
wake-word-ml/
├── esp32/mfcc.h                → cap-ai/firmware/src/mfcc.h
├── esp32/hermes_inference.h    → cap-ai/firmware/src/hermes_inference.h
└── models/hermes.export.bin    → cap-ai/firmware/src/hermes_weights.h (генератор в export.rs)
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
#define WAKE_THRESHOLD  0.85f   // из кривой make eval, НЕ 0.5
#define N_MFCC          20
#define NUM_FRAMES      40      // 1.2с при кадре 30мс
#define FRAME_SIZE      480     // 30мс при 16kHz
#define HOP_SAMPLES     1600    // шаг стриминга 100мс
#define WAKE_CONFIRM    2       // окон подряд выше порога
#define WAKE_COOLDOWN   20      // шагов по 100мс = 2с
#define NOISE_FLOOR     0.005f  // RMS ниже — CNN не считаем
```

### main.cpp

```cpp
#include "mfcc.h"
#include "hermes_inference.h"

TaskHandle_t wake_task_handle;

void wake_word_task(void* param) {
    static int16_t ring[NUM_FRAMES * FRAME_SIZE];   // 19200 сэмплов = 1.2с
    static float mfcc[NUM_FRAMES][N_MFCC];
    int hits = 0, cooldown = 0;

    while (true) {
        i2s_read_into_ring(ring, HOP_SAMPLES);       // блокируется на 100мс аудио
        if (cooldown > 0) { cooldown--; hits = 0; continue; }
        if (ring_rms(ring) < NOISE_FLOOR) { hits = 0; continue; }

        compute_mfcc(ring, mfcc);                    // тот же алгоритм, что src/mfcc.rs
        float p = hermes_forward(mfcc);
        hits = (p > WAKE_THRESHOLD) ? hits + 1 : 0;
        if (hits >= WAKE_CONFIRM) {
            xTaskNotifyGive(main_task_handle);
            cooldown = WAKE_COOLDOWN;
            hits = 0;
        }
    }
}

void setup() {
    // ...
    xTaskCreatePinnedToCore(wake_word_task, "wake", 16384, NULL, 1, &wake_task_handle, 1);
}
```

Стек задачи 16 KB: буферы свёрток объявлены `static` в
`hermes_inference.h`, в стек не попадают.

### State machine (без изменений)

```
LIGHT_SLEEP + wake_word_task (Core 1)
    │ 2 окна подряд > WAKE_THRESHOLD
    ▼
WAKING_UP → LISTENING → CAPTURING → RESPONDING → TEARDOWN
    ▼
LIGHT_SLEEP + wake_word_task снова
```

---

## План работы

1. Компоненты CapAI пришли → собираем, тестируем модули
2. Датасет и модель — по [08-runbook](08-runbook.md), этапы 2b–7
3. `cargo run --bin export` → `hermes.export.bin` + `hermes_weights.h`
4. **Тест совпадения**: 100 окон из `dataset/` через Rust (`make score`)
   и через C++ на ESP32 (или на Mac, скомпилировав `hermes_inference.h`
   обычным `clang`) — вероятности равны до 3 знака. До этого метрики
   на устройстве не измерять
5. Тест на кепке дома: говорим «Гермес» → просыпается; молчим/говорим
   другое → не просыпается. Порог — из `make eval`
6. Тест на улице; при провале — этап 2 фаза 2 (перезапись позитивов на
   INMP441) и негативы с улицы

Критерии готовности и метрики (false accepts/hour, тестовые наборы) —
в [07-quality.md](07-quality.md). Без них пункт 5 будет работать только дома в тишине.
