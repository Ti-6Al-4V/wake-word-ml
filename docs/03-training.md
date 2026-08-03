# Обучение модели

TensorFlow/Keras. CNN для бинарной классификации: wake word / не wake word.

---

## Окружение

```bash
# Python 3.12
pip install tensorflow[and-cuda] librosa scikit-learn numpy soundfile
```

Проверка GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # должен увидеть GPU
```

Без GPU тоже работает — модель маленькая, обучение ~30 мин на CPU.

---

## Модель

```python
# training/train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(33, 20, 1)):
    model = models.Sequential([
        # Conv блок 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Conv блок 2
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Классификация
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model
```

### Архитектура:

```
Вход: [33, 20, 1] (MFCC матрица, 1 канал)
    │
    ▼
Conv2D(16, 3×3) + ReLU → [33, 20, 16]
MaxPool(2×2)             → [16, 10, 16]
Dropout(0.3)
    │
    ▼
Conv2D(8, 3×3) + ReLU   → [16, 10, 8]
MaxPool(2×2)             → [8, 5, 8]
Dropout(0.3)
    │
    ▼
Flatten                  → [320]
Dense(16) + ReLU
Dropout(0.3)
    │
    ▼
Dense(1) + Sigmoid       → [0.0 ... 1.0]
```

~5000 параметров. .tflite ~25 KB.

---

## Обучение

```python
# training/train_model.py (продолжение)

def train():
    # Загружаем датасет
    X_train, y_train, X_val, y_val = load_dataset()

    # Добавляем размерность канала: [33, 20] → [33, 20, 1]
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]

    model = build_model()

    # Callbacks
    callbacks = [
        # Ранняя остановка — если val_loss не улучшается 5 эпох
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        ),
        # Сохраняем лучшую модель
        tf.keras.callbacks.ModelCheckpoint(
            'models/wake_word_best.keras',
            save_best_only=True, monitor='val_accuracy'
        ),
        # Уменьшаем learning rate если застряли
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )

    return model, history
```

### Параметры:

| Параметр | Значение | Почему |
|----------|---------|--------|
| Epochs | 50 (early stop ~15-25) | Модель маленькая, сходится быстро |
| Batch size | 32 | Стандарт, не перегружает RAM |
| Learning rate | 0.001 | Adam default, ReduceLROnPlateau уменьшает если застряли |
| Dropout | 0.3 | Защита от overfitting |
| Loss | binary_crossentropy | Бинарная классификация |

---

## Оценка

```python
def evaluate(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")

    # Confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    # [[TN  FP]
    #  [FN  TP]]

    # False positive rate (важно для wake word)
    fpr = cm[0][1] / (cm[0][0] + cm[0][1])
    print(f"False positive rate: {fpr:.4f}")
```

### Целевые метрики:

| Метрика | Цель | Почему |
|---------|------|--------|
| Accuracy | >90% | Общая точность |
| False positive rate | <3% | Кепка не должна просыпаться на другие слова |
| False negative rate | <10% | Кепка должна слышать "гермес" в 9 из 10 случаев |
| Model size | <100 KB | Помещается в PSRAM ESP32-S3 |

---

## Аугментация при обучении

```python
# training/augment.py
import librosa
import numpy as np

def augment_audio(y, sr=16000):
    augmented = []

    # Pitch shift
    for steps in [-2, 2]:
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        augmented.append(y_shift)

    # Time stretch
    for rate in [0.9, 1.1]:
        y_stretch = librosa.effects.time_stretch(y, rate=rate)
        augmented.append(y_stretch)

    # Add noise
    for snr in [10, 20]:
        noise = np.random.randn(len(y)) * 0.01 * (10 ** (-snr / 20))
        augmented.append(y + noise)

    # Volume change
    for vol in [0.7, 1.3]:
        augmented.append(y * vol)

    return augmented
```

100 реальных → 600 аугментированных (×6).

---

## Конвертация в TFLite

```python
# training/convert_tflite.py
import tensorflow as tf

def convert_to_tflite(keras_model_path, output_path):
    model = tf.keras.models.load_model(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Оптимизация для размера
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Quantization (int8) — для скорости на ESP32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model size: {len(tflite_model)} bytes")
    # ~10-30 KB после quantization
```

### Quantization (int8)

По умолчанию веса — float32 (4 байта). Quantization преобразует в int8 (1 байт) — модель в 4 раза меньше, inference в 3-5 раз быстрее.

Цена: ~1-2% точности. Для wake word — нормально.

---

## Дальше

- [04-deploy.md](04-deploy.md) — деплой на ESP32