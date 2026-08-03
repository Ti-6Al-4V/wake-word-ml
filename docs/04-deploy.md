# Деплой на ESP32-S3

TFLite Micro + MFCC extraction на микроконтроллере.

---

## Архитектура inference

```
I2S → PCM (16kHz, 16-bit, mono)
    │
    ▼
Кольцевой буфер (1 сек, 16000 сэмплов = 32 KB)
    │
    ▼ каждые 480 сэмплов (30мс)
MFCC извлечение → [20 коэффициентов]
    │
    ▼ накапливаем 33 окна
MFCC матрица [33 × 20]
    │
    ▼
TFLite Micro: model->invoke()
    │
    ▼
score (int8 → float)
    │
    ▼
score > THRESHOLD → wake word!
```

---

## TFLite Micro на ESP32

### Библиотека

```ini
# platformio.ini (дополнение к CapAI)
lib_deps =
    tensorflow-lite-micro-esp32
```

### Структура

```
esp32/
├── inference.h         # TFLite Micro wrapper
├── mfcc.h              # MFCC извлечение на ESP32
├── audio_buffer.h      # кольцевой буфер PCM
└── models/
    └── wake_word.tflite  # обученная модель
```

---

## MFCC на ESP32

```cpp
// mfcc.h
#pragma once
#include <math.h>
#include <Arduino.h>

#define MFCC_NUM 20
#define FRAME_SIZE 480      // 30мс при 16kHz
#define HOP_SIZE 480        // шаг 30мс
#define FFT_SIZE 512
#define NUM_FRAMES 33       // 1 сек / 30мс

// Mel filterbank (предрассчитан)
static const float mel_filters[MFCC_NUM][FFT_SIZE/2] = { ... };

// Hamming window
static float hamming[FRAME_SIZE];
void init_hamming() {
    for (int i = 0; i < FRAME_SIZE; i++)
        hamming[i] = 0.54 - 0.46 * cos(2*PI*i / (FRAME_SIZE-1));
}

// DCT матрица (предрассчитана)
static const float dct_matrix[MFCC_NUM][MFCC_NUM] = { ... };

// FFT (простая реализация radix-2)
void fft(float *real, float *imag, int n) {
    // Cooley-Tukey FFT
    // ...
}

// Один кадр PCM → 20 MFCC коэффициентов
void compute_mfcc_frame(int16_t *pcm, float *mfcc_out) {
    float frame[FRAME_SIZE];
    
    // 1. Pre-emphasis + Hamming window
    for (int i = 0; i < FRAME_SIZE; i++) {
        frame[i] = (float)pcm[i] / 32768.0;
        frame[i] *= hamming[i];
    }
    
    // 2. FFT → спектр мощности
    float real[FFT_SIZE] = {0}, imag[FFT_SIZE] = {0};
    memcpy(real, frame, FRAME_SIZE * sizeof(float));
    fft(real, imag, FFT_SIZE);
    
    float power[FFT_SIZE/2];
    for (int i = 0; i < FFT_SIZE/2; i++)
        power[i] = real[i]*real[i] + imag[i]*imag[i];
    
    // 3. Mel filterbank → 20 полос
    float mel_energy[MFCC_NUM] = {0};
    for (int m = 0; m < MFCC_NUM; m++)
        for (int k = 0; k < FFT_SIZE/2; k++)
            mel_energy[m] += mel_filters[m][k] * power[k];
    
    // 4. Log
    for (int m = 0; m < MFCC_NUM; m++)
        mel_energy[m] = log(mel_energy[m] + 1e-8);
    
    // 5. DCT → MFCC
    for (int m = 0; m < MFCC_NUM; m++) {
        mfcc_out[m] = 0;
        for (int k = 0; k < MFCC_NUM; k++)
            mfcc_out[m] += dct_matrix[m][k] * mel_energy[k];
    }
    
    // 6. Нормализация
    float mean = 0, std = 0;
    for (int m = 0; m < MFCC_NUM; m++) mean += mfcc_out[m];
    mean /= MFCC_NUM;
    for (int m = 0; m < MFCC_NUM; m++) std += sq(mfcc_out[m] - mean);
    std = sqrt(std / MFCC_NUM) + 1e-8;
    for (int m = 0; m < MFCC_NUM; m++)
        mfcc_out[m] = (mfcc_out[m] - mean) / std;
}
```

---

## TFLite Micro Inference

```cpp
// inference.h
#pragma once
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "wake_word_model.h"  // .tflite как C массив

#define TENSOR_ARENA_SIZE 8192  // 8 KB

class WakeWordDetector {
public:
    bool init() {
        // Загружаем модель
        model = tflite::GetModel(wake_word_model_tflite);
        
        // Регистрируем нужные ops
        static tflite::MicroMutableOpResolver<6> resolver;
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddFullyConnected();
        resolver.AddReshape();
        resolver.AddSoftmax();
        resolver.AddPad();
        
        // Создаём интерпретатор
        static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
        static tflite::MicroInterpreter interpreter(
            model, resolver, tensor_arena, TENSOR_ARENA_SIZE
        );
        interpreter_ = &interpreter;
        
        interpreter_->AllocateTensors();
        input_ = interpreter_->input(0);
        output_ = interpreter_->output(0);
        
        return true;
    }
    
    // mfcc: [33 × 20] → score
    float detect(float mfcc[NUM_FRAMES][MFCC_NUM]) {
        // Заполняем input tensor (int8 quantized)
        int8_t* input_data = input_->data.int8;
        float input_scale = input_->params.scale;
        int input_zero = input_->params.zero_point;
        
        int idx = 0;
        for (int t = 0; t < NUM_FRAMES; t++) {
            for (int m = 0; m < MFCC_NUM; m++) {
                input_data[idx++] = (int8_t)(mfcc[t][m] / input_scale + input_zero);
            }
        }
        
        // Inference
        interpreter_->Invoke();
        
        // Читаем output
        float output_scale = output_->params.scale;
        int output_zero = output_->params.zero_point;
        float score = (output_->data.int8[0] - output_zero) * output_scale;
        
        return score;  // 0.0 ... 1.0
    }
    
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter_;
    TfLiteTensor* input_;
    TfLiteTensor* output_;
};
```

---

## Основной цикл

```cpp
// main loop (Core 1, не мешает WiFi на Core 0)
void wake_word_task(void* param) {
    WakeWordDetector detector;
    detector.init();
    
    float mfcc_buffer[NUM_FRAMES][MFCC_NUM];
    int frame_idx = 0;
    
    int16_t pcm[FRAME_SIZE];
    
    while (true) {
        // Читаем 30мс аудио из I2S
        size_t bytes_read;
        i2s_read(I2S_MIC_PORT, pcm, FRAME_SIZE * 2, &bytes_read, portMAX_DELAY);
        
        // Вычисляем MFCC для этого кадра
        float mfcc[MFCC_NUM];
        compute_mfcc_frame(pcm, mfcc);
        
        // Добавляем в кольцевой буфер
        memcpy(mfcc_buffer[frame_idx], mfcc, MFCC_NUM * sizeof(float));
        frame_idx = (frame_idx + 1) % NUM_FRAMES;
        
        // Каждые 10 кадров (~300мс) — проверяем
        if (frame_idx % 10 == 0) {
            float score = detector.detect(mfcc_buffer);
            
            if (score > WAKE_THRESHOLD) {
                // WAKE WORD DETECTED!
                xTaskNotifyGive(main_task_handle);
            }
        }
        
        // Light sleep между окнами (экономия энергии)
        // ~5мс свободно, остальное — vTaskDelay
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
```

---

## Память

| Компонент | Размер | Где |
|-----------|--------|-----|
| Модель .tflite | 10-30 KB | Flash (PROGMEM) |
| MFCC буфер | 2.6 KB | RAM (33×20×4) |
| TFLite arena | 8 KB | PSRAM |
| PCM буфер | 32 KB | PSRAM |
| Hamming + DCT + Mel | ~10 KB | Flash |
| **Итого** | **~80 KB** | из 8MB PSRAM + 8MB Flash |

---

## Скорость

| Операция | Время |
|----------|-------|
| I2S read (30мс аудио) | ~1мс |
| MFCC (FFT + mel + DCT) | ~5мс |
| TFLite inference | ~15-25мс |
| Полный цикл | ~20-30мс |
| Проверка каждые | 300мс (10 кадров) |

Real-time: 30мс обработка на 30мс аудио. Успеваем.

---

## Дальше

- [05-integration.md](05-integration.md) — интеграция в CapAI