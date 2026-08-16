#!/usr/bin/env python3
# QC дублей «Гермес»: проверка, что в WAV реально есть слово, а не тишина.
# Использование: python3 scripts/qc.py [папка]   (по умолчанию dataset/raw/real)
#
# Метрики на файл:
#   пик  — максимальная амплитуда (0..1). Норма для речи ≥0.15, идеал 0.3–0.8.
#   rms  — средняя энергия. У чистой речи обычно 0.02–0.15.
#   пик% — где внутри окна находится максимум. >93% = слово не успело в окно,
#          начало затянуто (говори сразу после секундной паузы).
#
# Код выхода: 0 — все файлы годные, 1 — есть подозрительные.

import wave, struct, glob, os, sys

folder = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw/real"
files = sorted(glob.glob(f"{folder}/*.wav"))
if not files:
    print(f"В {folder} нет WAV-файлов")
    sys.exit(1)

PEAK_MIN = 0.10   # ниже — тишина/шёпот, в датасет не годится
PEAK_MAX = 0.99   # выше — клиппинг (перегруз)
POS_MAX = 93.0    # пик позже этого процента = слово в самом конце окна

print(f'{"файл":28} {"длит":>5} {"пик":>6} {"rms":>7} {"пик на":>7}  вердикт')
bad = []
for f in files:
    w = wave.open(f)
    n, rate = w.getnframes(), w.getframerate()
    samples = struct.unpack(f"<{n}h", w.readframes(n))
    peak = max(abs(s) for s in samples) / 32767
    rms = (sum(s * s for s in samples) / n) ** 0.5 / 32767
    pos = max(range(n), key=lambda i: abs(samples[i])) / n * 100

    verdict = ""
    if peak < PEAK_MIN:
        verdict = "ТИХО/ПУСТО"
    elif peak > PEAK_MAX:
        verdict = "КЛИППИНГ"
    elif pos > POS_MAX:
        verdict = "СЛОВО В КОНЦЕ ОКНА"
    if verdict:
        bad.append((f, verdict))
        verdict = f"<-- {verdict}"
    else:
        verdict = "ок"
    print(f"{os.path.basename(f):28} {n/rate:5.2f} {peak:6.2f} {rms:7.3f} {pos:6.1f}%  {verdict}")

print(f"\nИтого: {len(files)} файлов, годных: {len(files) - len(bad)}, подозрительных: {len(bad)}")
if bad:
    print("Перезаписать бы (удалить и надиктовать заново):")
    for f, why in bad:
        print(f"  {f}  ({why})")
sys.exit(1 if bad else 0)
