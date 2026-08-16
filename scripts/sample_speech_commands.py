#!/usr/bin/env python3
# Детерминированная выборка негативов из Speech Commands v0.02:
# N случайных клипов (по умолчанию 2000) из всех 35 слов + фоновые шумы.
# Сид фиксирован (42) — повторный запуск даёт ту же выборку.
#
# Запуск: python3 scripts/sample_speech_commands.py [N]

import os, random, shutil, sys

SRC = "dataset/raw/speech_commands"
OUT = "dataset/raw/sc_sample"
BG_OUT = "dataset/raw/background"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

random.seed(42)
wavs = []
for word in sorted(os.listdir(SRC)):
    d = os.path.join(SRC, word)
    # _background_noise_ и служебные файлы не являются словами
    if not os.path.isdir(d) or word.startswith(("_", ".")):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".wav"):
            wavs.append(os.path.join(d, f))
print(f"Всего клипов в Speech Commands: {len(wavs)}")

sample = random.sample(wavs, N)
os.makedirs(OUT, exist_ok=True)
for i, p in enumerate(sample, 1):
    shutil.copy(p, os.path.join(OUT, f"negative_sc_{i:05}.wav"))
print(f"Скопировано {N} негативов в {OUT}")

# Фоновые шумы (записи по несколько минут) — пригодятся для ambient-теста
os.makedirs(BG_OUT, exist_ok=True)
bg = os.path.join(SRC, "_background_noise_")
count = 0
for f in sorted(os.listdir(bg)):
    if f.endswith(".wav"):
        shutil.copy(os.path.join(bg, f), os.path.join(BG_OUT, f))
        count += 1
print(f"Фоновых шумов скопировано: {count} в {BG_OUT}")
