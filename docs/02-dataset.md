# Датасет

Сбор и подготовка данных для обучения wake word модели.

---

## Что нужно собрать

| Категория | Что | Кол-во | Длительность |
|-----------|-----|--------|--------------|
| Позитивные | слово "гермес" | 500-2000 | 1 сек каждая |
| Негативные | другие слова, фразы, шум | 2000-5000 | 1-3 сек |
| Фон | тишина, музыка, улица | 500 | 1-3 сек |

---

## Сбор позитивных ("гермес")

### Способ 1: Записать самому

```bash
# Скрипт записи (Python, на ноутбуке)
# recording/record_positives.py
python record_positives.py --word "гермес" --count 500 --output dataset/positive/
```

Скрипт:
1. Показывает "Запись 1/500"
2. Ты говоришь "гермес"
3. Запись 1 сек → сохраняет `positive/germes_001.wav`
4. Пауза 2 сек
5. Повтор

500 записей × 3 сек цикл = ~25 минут.

### Способ 2: Аугментация (из 100 → 1000)

Из 100 реальных записей生成 1000 через аугментацию:

```python
augmentations = [
    pitch_shift(±2 полутона),      # выше/ниже
    time_stretch(0.8x ... 1.2x),   # быстрее/медленнее
    add_noise(SNR 5-20 dB),        # фоновый шум
    volume(0.5x ... 1.5x),         # тише/громче
    shift(±100мс),                 # сдвиг по времени
]
```

100 записей × 10 аугментаций = 1000 позитивных.

### Способ 3: Синтетические (TTS)

Генерируем "гермес" через Edge TTS разными голосами:

```python
voices = ["ru-RU-SvetlanaNeural", "ru-RU-DmitriNeural", ...]
for voice in voices:
    for speed in [0.8, 0.9, 1.0, 1.1, 1.2]:
        tts.generate("гермес", voice=voice, speed=speed)
```

Быстро, но менее реалистично. Хорошо для дополнения к реальным записям.

---

## Сбор негативных

### Источники:

1. **Google Speech Commands Dataset** — 35 слов, ~100k записей. Бесплатно. Слова типа "yes", "no", "stop" — не "гермес", но речь.
2. **Common Voice (Mozilla)** — русская речь, тысячи фраз.
3. **Шум:** recordings of music, street, office, wind, typing.
4. **Похожие слова:** "германий", "серьёзно", "Герма", "героизм" — чтобы модель училась отличать.

### Структура папок:

```
dataset/
├── positive/          # "гермес"
│   ├── germes_001.wav
│   ├── germes_002.wav
│   └── ...
├── negative/          # другие слова + шум
│   ├── other_001.wav
│   ├── noise_001.wav
│   └── ...
└── background/        # тишина, музыка, улица
    ├── silence_001.wav
    ├── street_001.wav
    └── ...
```

---

## Подготовка (preprocessing)

Все аудио приводим к единому формату:

```python
# preprocessing.py
import librosa
import soundfile as sf

def preprocess(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Нормализация громкости
    y = y / max(abs(y)) * 0.9
    
    # Обрезка/дополнение до 1 сек (16000 сэмплов)
    if len(y) > 16000:
        y = y[:16000]
    elif len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    
    sf.write(output_path, y, 16000)
```

---

## Извлечение MFCC

Преобразуем WAV в MFCC матрицы для обучения:

```python
# extract_features.py
import librosa
import numpy as np

def wav_to_mfcc(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=20,           # 20 коэффициентов
        n_fft=512,           # размер FFT окна
        hop_length=480,      # шаг (30мс при 16kHz)
        win_length=480,      # размер окна (30мс)
    )
    
    # Транспонируем: [time × mfcc] → [33 × 20]
    mfcc = mfcc.T
    
    # Нормализация
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    
    return mfcc  # shape: (33, 20)
```

### Размер:

- 1 сек аудио → 33 окна × 20 MFCC = матрица [33 × 20]
- Это вход нейросети

---

## Разделение датасета

```
Всего: 3000 записей (1000 позитивных + 2000 негативных)

Train:      70%  (2100) — обучение
Validation: 15%  (450)  — контроль during training
Test:       15%  (450)  — финальная проверка
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    mfccs, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)
```

---

## Дальше

- [03-training.md](03-training.md) — обучаем модель на этом датасете