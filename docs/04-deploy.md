# Деплой на ESP32-S3

Ручной C++ inference. Никаких рантаймов — модель маленькая, forward pass пишем руками.

---

## Почему без TFLite

Модель: 2 Conv2D + 2 Dense = ~5000 параметров, ~20 KB. Ручной forward pass ~200 строк C++. Полный контроль, ноль зависимостей.

---

## Экспорт из Rust

`models/wake_word.bin` — бинарный файл с весами:

```
[header: ndim + shape] [data: f32 values]
  Conv1 weights [16, 1, 3, 3]  = 144 f32 + 16 bias = 640 bytes
  Conv2 weights [8, 16, 3, 3]  = 1152 f32 + 8 bias  = 4640 bytes
  FC1   weights [320, 16]      = 5120 f32 + 16 bias = 20544 bytes
  FC2   weights [16, 1]        = 16 f32 + 1 bias    = 68 bytes
  Итого: ~26 KB
```

---

## Загрузка весов на ESP32

```cpp
// wake_word_weights.h
#pragma once
#include <pgmspace.h>

// Веса прошиваются во Flash через PROGMEM
// Сгенерировано из wake_word.bin (xxd -i)

static const float CONV1_WEIGHT[16][1][3][3] PROGMEM = { ... };
static const float CONV1_BIAS[16] PROGMEM = { ... };
static const float CONV2_WEIGHT[8][16][3][3] PROGMEM = { ... };
static const float CONV2_BIAS[8] PROGMEM = { ... };
static const float FC1_WEIGHT[16][320] PROGMEM = { ... };
static const float FC1_BIAS[16] PROGMEM = { ... };
static const float FC2_WEIGHT[1][16] PROGMEM = { ... };
static const float FC2_BIAS[1] PROGMEM = { ... };
```

---

## MFCC на C++ (тот же алгоритм что в Rust)

```cpp
// mfcc.h — см. docs/04-deploy.md в предыдущей версии
// FFT → mel filterbank → log → DCT → нормализация
// Вход: PCM 16000 сэмплов, выход: [33][20] MFCC матрица
```

---

## Ручной CNN Forward Pass

```cpp
// wake_word_inference.h
#pragma once
#include <math.h>
#include "wake_word_weights.h"

#define CONV1_OUT 16
#define CONV2_OUT 8
#define FC1_OUT   16
#define INPUT_H   33
#define INPUT_W   20

static float relu(float x) { return x > 0 ? x : 0; }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Conv2d + ReLU, padding=same, stride=1
void conv2d_relu(const float* input, int in_h, int in_w, int in_ch,
                 const float* weight, const float* bias,
                 int out_ch, int k_h, int k_w,
                 float* output) {
    int out_h = in_h;  // same padding
    int out_w = in_w;
    int pad_h = (k_h - 1) / 2;
    int pad_w = (k_w - 1) / 2;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ky = 0; ky < k_h; ky++) {
                        for (int kx = 0; kx < k_w; kx++) {
                            int iy = y + ky - pad_h;
                            int ix = x + kx - pad_w;
                            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                float in_val = input[ic * in_h * in_w + iy * in_w + ix];
                                float w_val = weight[oc * in_ch * k_h * k_w
                                                   + ic * k_h * k_w + ky * k_w + kx];
                                sum += in_val * w_val;
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + y * out_w + x] = relu(sum);
            }
        }
    }
}

// MaxPool2d 2×2, stride=2
void maxpool2d(const float* input, int in_h, int in_w, int channels,
               float* output, int* out_h, int* out_w) {
    *out_h = in_h / 2;
    *out_w = in_w / 2;
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < *out_h; y++) {
            for (int x = 0; x < *out_w; x++) {
                float max = -1e30;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y * 2 + dy;
                        int ix = x * 2 + dx;
                        float v = input[c * in_h * in_w + iy * in_w + ix];
                        if (v > max) max = v;
                    }
                }
                output[c * (*out_h) * (*out_w) + y * (*out_w) + x] = max;
            }
        }
    }
}

// Dense + ReLU
void dense_relu(const float* input, int in_size,
                const float* weight, const float* bias,
                int out_size, float* output) {
    for (int o = 0; o < out_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_size; i++) {
            sum += input[i] * weight[o * in_size + i];
        }
        output[o] = relu(sum);
    }
}

// Dense + Sigmoid
float dense_sigmoid(const float* input, int in_size,
                    const float* weight, const float* bias,
                    int out_size) {
    float sum = bias[0];
    for (int i = 0; i < in_size; i++) {
        sum += input[i] * weight[i];
    }
    return sigmoid(sum);
}

// Полный forward pass
float wake_word_forward(const float mfcc[INPUT_H][INPUT_W]) {
    // Input: [1, 33, 20] → flatten to [33*20]
    float input[INPUT_H * INPUT_W];
    for (int y = 0; y < INPUT_H; y++)
        for (int x = 0; x < INPUT_W; x++)
            input[y * INPUT_W + x] = mfcc[y][x];

    // Conv1: [1, 33, 20] → [16, 33, 20]
    static float conv1_out[CONV1_OUT * INPUT_H * INPUT_W];
    conv2d_relu(input, INPUT_H, INPUT_W, 1,
                (const float*)CONV1_WEIGHT, CONV1_BIAS,
                CONV1_OUT, 3, 3, conv1_out);

    // MaxPool1: [16, 33, 20] → [16, 16, 10]
    int p1_h, p1_w;
    static float pool1_out[CONV1_OUT * 17 * 10];
    maxpool2d(conv1_out, INPUT_H, INPUT_W, CONV1_OUT, pool1_out, &p1_h, &p1_w);

    // Conv2: [16, 16, 10] → [8, 16, 10]
    static float conv2_out[CONV2_OUT * 17 * 10];
    conv2d_relu(pool1_out, p1_h, p1_w, CONV1_OUT,
                (const float*)CONV2_WEIGHT, CONV2_BIAS,
                CONV2_OUT, 3, 3, conv2_out);

    // MaxPool2: [8, 16, 10] → [8, 8, 5] = 320
    int p2_h, p2_w;
    static float pool2_out[CONV2_OUT * 9 * 5];
    maxpool2d(conv2_out, p1_h, p1_w, CONV2_OUT, pool2_out, &p2_h, &p2_w);

    int flatten_size = CONV2_OUT * p2_h * p2_w;  // 320

    // FC1: 320 → 16
    static float fc1_out[FC1_OUT];
    dense_relu(pool2_out, flatten_size,
               (const float*)FC1_WEIGHT, FC1_BIAS,
               FC1_OUT, fc1_out);

    // FC2: 16 → 1 (sigmoid)
    float score = dense_sigmoid(fc1_out, FC1_OUT,
                                (const float*)FC2_WEIGHT, FC2_BIAS, 1);
    return score;  // 0.0 ... 1.0
}
```

---

## Основной цикл

```cpp
// в CapAI firmware, Core 1
void wake_word_task(void* param) {
    float mfcc[33][20];
    int16_t pcm[FRAME_SIZE];

    while (true) {
        // Читаем 30мс аудио
        size_t n;
        i2s_read(I2S_MIC_PORT, pcm, FRAME_SIZE * 2, &n, portMAX_DELAY);

        // MFCC для этого кадра → накапливаем 33 кадра
        compute_mfcc_frame(pcm, mfcc[frame_idx]);
        frame_idx = (frame_idx + 1) % 33;

        // Проверяем каждые 10 кадров (~300мс)
        if (frame_idx % 10 == 0) {
            float score = wake_word_forward(mfcc);
            if (score > WAKE_THRESHOLD) {
                xTaskNotifyGive(main_task_handle);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
```

---

## Память

| Компонент | Размер | Где |
|-----------|--------|-----|
| Веса (PROGMEM) | ~26 KB | Flash |
| Conv1 буфер | 10.6 KB | PSRAM |
| Pool1 буфер | 10.9 KB | PSRAM |
| Conv2 буфер | 5.4 KB | PSRAM |
| Pool2 буфер | 1.4 KB | PSRAM |
| FC1 буфер | 64 B | RAM |
| MFCC буфер | 2.6 KB | RAM |
| **Итого** | **~57 KB** | из 8MB PSRAM + 8MB Flash |

---

## Скорость

| Операция | Время |
|----------|-------|
| MFCC (FFT + mel + DCT) | ~5мс |
| Conv1 (16×3×3×1×33×20) | ~3мс |
| Conv2 (8×3×3×16×16×10) | ~2мс |
| FC1 + FC2 | ~1мс |
| **Полный forward** | **~11мс** |
| Проверка | каждые 300мс |

Real-time. Без рантаймов, без TFLite, без зависимостей.

---

## Дальше

- [05-integration.md](05-integration.md) — интеграция в CapAI