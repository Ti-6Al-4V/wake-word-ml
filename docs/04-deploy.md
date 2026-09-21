# Деплой на ESP32-S3

> Статус: проект реализации. `src/bin/export.rs` — заглушка; C++ ниже —
> образец, а не собранная и проверенная прошивка. Порядок работы — L8 в
> [лабораторных](10-labs.md). Первым реализуется float-путь.

## Почему без TFLite

Модель: 2 Conv2D + 2 Dense = 4425 параметров, 18 KB float32. Ручной
forward pass ~200 строк C++. Полный контроль, ноль зависимостей.

---

## Что должно совпасть с Rust

Первый тест — численное совпадение одинакового входа в Rust и C++.
Сохрани эталонные PCM, признаки, выходы каждого слоя и логит. Сравнивай
`abs(a-b) <= atol + rtol*abs(reference)`: начальные допуски для f32 можно
взять atol=1e-5, rtol=1e-4 и уточнить по обусловленности операций.
Не увеличивай допуск, чтобы скрыть ошибку индексации. Проверка только
sigmoid до третьего знака может скрыть большую ошибку насыщенного логита.
Побитное совпадение разных FFT/GPU/CPU не требуется.

Обязаны совпасть:

1. MFCC: pre-emphasis 0.97, кадр 480 без перекрытия, Хэмминг, FFT 512,
   20 mel-фильтров 0–8000 Гц (формула `hz_to_bin` из `src/mfcc.rs`),
   `ln(max(energy, 1e-5))`, DCT-II 20×20, CMVN по каждому коэффициенту
   (среднее/дисперсия по 40 кадрам накапливаются в double из float,
   `+1e-5` к std, результат приводится к float).
2. Порядок весов и раскладка тензоров (ниже).
3. Паддинг Valid: выход свёртки на 2 меньше входа по каждой оси.

---

## Формат экспорта, который предстоит реализовать

`models/hermes.bin` — checkpoint Burn, не плоский массив C++.
План для `models/hermes.export.bin`: magic/version, little-endian, число
тензоров, затем для каждого тензора заголовок и значения:

```
[u32 ndim][u32 dim0..dimN][f32 × prod(dims)]   ×  8 тензоров по порядку:

  conv1.weight [16, 1, 3, 3]     144 f32
  conv1.bias   [16]
  conv2.weight [8, 16, 3, 3]    1152 f32
  conv2.bias   [8]
  fc1.weight   [192, 16]        3072 f32   — burn хранит Linear как [in, out]
  fc1.bias     [16]
  fc2.weight   [16, 1]            16 f32
  fc2.bias     [1]
  Итого 4425 f32 = 17.7 KB
```

Внимание на `fc1.weight [in, out]`: burn считает `x · W`, а не `W · x`.
В C++ ниже индексация под это.

Квантование int8 (после того, как float-версия совпала с Rust):
per-tensor scale = max|w| / 127, веса → int8, активации остаются float.
18 KB → 4.4 KB. Это weight-only хранение: оно не гарантирует ускорения и не является
полностью целочисленным inference. Текущий eval читает float-checkpoint;
для проверки реализуй загрузку квантованных весов с теми же scale и
арифметикой, затем сравни события на development/final. Bias и метаданные
добавляют байты к 4425 байтам int8. Для integer-активаций потребуются
калибровка, scale/zero-point, аккумуляторы int32 и requantization.

---

## Загрузка весов на ESP32

```cpp
// hermes_weights.h — сгенерировано из hermes.export.bin
#pragma once
static const float CONV1_W[16][1][3][3] = { ... };
static const float CONV1_B[16] = { ... };
static const float CONV2_W[8][16][3][3] = { ... };
static const float CONV2_B[8] = { ... };
static const float FC1_W[192][16] = { ... };   // [in][out], как в burn
static const float FC1_B[16] = { ... };
static const float FC2_W[16][1] = { ... };
static const float FC2_B[1] = { ... };
```

На ESP32-S3 константные массивы и так лежат во flash (rodata); PROGMEM
не нужен.

---

## Ручной CNN Forward Pass

```cpp
// hermes_inference.h
#pragma once
#include <math.h>
#include "hermes_weights.h"

#define NUM_FRAMES 40
#define N_MFCC     20

static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Conv2d + ReLU, padding=VALID, stride=1: out = in - 2 по каждой оси.
// Раскладка: input[ic][y][x], weight[oc][ic][ky][kx], output[oc][y][x]
static void conv2d_valid_relu(const float* in, int in_h, int in_w, int in_ch,
                              const float* w, const float* b, int out_ch,
                              float* out) {
    const int out_h = in_h - 2, out_w = in_w - 2;
    for (int oc = 0; oc < out_ch; oc++)
        for (int y = 0; y < out_h; y++)
            for (int x = 0; x < out_w; x++) {
                float s = b[oc];
                for (int ic = 0; ic < in_ch; ic++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++)
                            s += in[(ic * in_h + y + ky) * in_w + x + kx]
                               * w[((oc * in_ch + ic) * 3 + ky) * 3 + kx];
                out[(oc * out_h + y) * out_w + x] = relu(s);
            }
}

// MaxPool 2×2, stride 2: нечётный хвост отбрасывается (19→9, 17→8, 7→3).
static void maxpool2(const float* in, int in_h, int in_w, int ch, float* out) {
    const int out_h = in_h / 2, out_w = in_w / 2;
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < out_h; y++)
            for (int x = 0; x < out_w; x++) {
                float m = -1e30f;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        float v = in[(c * in_h + 2 * y + dy) * in_w + 2 * x + dx];
                        if (v > m) m = v;
                    }
                out[(c * out_h + y) * out_w + x] = m;
            }
}

// Полный forward: MFCC [40][20] → вероятность 0..1
static float hermes_forward(const float mfcc[NUM_FRAMES][N_MFCC]) {
    static float c1[16 * 38 * 18];
    static float p1[16 * 19 * 9];
    static float c2[8 * 17 * 7];
    static float p2[8 * 8 * 3];       // 192 — вход Dense
    float h[16];

    conv2d_valid_relu(&mfcc[0][0], 40, 20, 1, &CONV1_W[0][0][0][0], CONV1_B, 16, c1);
    maxpool2(c1, 38, 18, 16, p1);
    conv2d_valid_relu(p1, 19, 9, 16, &CONV2_W[0][0][0][0], CONV2_B, 8, c2);
    maxpool2(c2, 17, 7, 8, p2);

    // Dense 192→16 + ReLU.  burn: y = x·W, W[in][out]
    for (int o = 0; o < 16; o++) {
        float s = FC1_B[o];
        for (int i = 0; i < 192; i++) s += p2[i] * FC1_W[i][o];
        h[o] = relu(s);
    }
    // Dense 16→1 → сигмоида
    float logit = FC2_B[0];
    for (int i = 0; i < 16; i++) logit += h[i] * FC2_W[i][0];
    return sigmoid(logit);
}
```

Порядок flatten: burn `reshape([b, c*h*w])` идёт по (c, h, w) — ровно
как `p2` заполнен выше. Если перепутать порядок, числа не сойдутся,
хотя размер 192 совпадёт.

---

## Стриминг (окно скользит, а не прыгает)

Базовый вариант — кольцевой буфер на 1.2с, каждые **100мс**
(1600 сэмплов) считаем MFCC всего окна и forward. Это не stateful CNN:
вычисления перекрывающихся окон повторяются. При идеальной геометрии
слово длительностью d полностью попадает хотя бы в одно окно, если
шаг <= 1.2−d; наличие двух успешных score этим не гарантируется.
Шаг 300мс не ошибочен сам по себе — сравни recall и задержку.
При startup сначала накопи полные 19200 отсчётов. Перед compute_mfcc
разверни ring в хронологический порядок и переведи PCM16 в float
делением на 32768, как в Rust. Функции I2S/ring/MFCC ниже предстоит реализовать:

```cpp
void hermes_task(void*) {
    static int16_t ring[19200];      // 1.2с при 16kHz
    static float mfcc[NUM_FRAMES][N_MFCC];
    int hits = 0, cooldown = 0;
    while (true) {
        read_i2s_into_ring(ring, 1600);          // +100мс
        if (cooldown > 0) { cooldown--; hits = 0; continue; }
        compute_mfcc(ring, mfcc);                // 40 кадров, CMVN внутри
        float p = hermes_forward(mfcc);
        hits = (p > WAKE_THRESHOLD) ? hits + 1 : 0;
        if (hits >= 2) {                         // 2 окна подряд
            xTaskNotifyGive(main_task_handle);
            cooldown = 20;                       // 2с
            hits = 0;
        }
    }
}
```

Эти K=2 и cooldown=20 шагов соответствуют `make stream HOP=100`.
Энергетический гейт пока выключен в обоих путях. `WAKE_THRESHOLD`
выбирается на development-потоке с размеченными позитивами, clip eval
служит предварительной диагностикой. Если добавляешь гейт — реализуй
его и в offline-оценке, проверь тихие обращения.

CNN требует 238 672 MAC без bias/активаций, из них conv2 — 137 088.
1152 — число весов conv2 без bias, а не число операций полного слоя.
Время MFCC/CNN и доля загрузки ESP32 не измерены. Замерь полный шаг,
p50/p95/максимум и пропуски аудио под нагрузкой; period=100мс — бюджет,
а не доказательство, что вычисления в него укладываются.

---

## Память

| Буфер | Размер |
|---|---|
| Веса (flash) | 17.7 KB float / 4.4 KB int8 |
| Кольцо PCM 1.2с | 38.4 KB |
| MFCC 40×20 | 3.2 KB |
| c1 16×38×18 | 43.8 KB |
| p1 16×19×9 | 10.9 KB |
| c2 8×17×7 | 3.8 KB |
| p2 192 | 0.8 KB |
| **Итого RAM** | **~100.9 KB** без FFT scratch, стека, драйверов и остальных задач |

c1 — самый большой буфер; после измерений можно объединить conv1→pool1
и переиспользовать память. Доступность внутренней SRAM/PSRAM зависит от
прошивки. Static-буферы в примере не допускают одновременных вызовов
из нескольких задач без синхронизации.

---

## Дальше

- [05-integration.md](05-integration.md) — интеграция в CapAI
