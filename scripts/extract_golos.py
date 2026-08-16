#!/usr/bin/env python3
# Режет фразы Golos (parquet-зеркало bonlime/golos-test) на окна по 1.2с —
# негативы «живая русская речь». В parquet аудио уже WAV 16kHz моно.
# Детерминированно: фиксированный порядок шардов и фраз, стоп по лимиту.
# Куски-паузы (RMS ниже порога) пропускаются, чтобы не тащить тишину.
#
# Запуск: uv run --with pyarrow python3 scripts/extract_golos.py [лимит]

import io, os, struct, sys, wave

import pyarrow.parquet as pq

DL = "dataset/raw/downloads"
OUT = "dataset/raw/golos_clips"
WINDOW = 19200    # 1.2с при 16kHz — единое окно всего датасета
RMS_MIN = 0.01    # ниже — пауза/тишина, в негативы не годится
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

shards = sorted(
    f for f in os.listdir(DL)
    if f.startswith("golos_test_") and f.endswith(".parquet")
)
os.makedirs(OUT, exist_ok=True)

n_clips = 0
stop = False
for shard in shards:
    path = f"{DL}/{shard}"
    try:
        table = pq.read_table(path, columns=["audio"])
    except Exception as e:
        print(f"[пропуск шарда] {shard}: {e}")
        continue
    print(f"{shard}: {table.num_rows} фраз")
    for row in table.to_pylist():
        w = wave.open(io.BytesIO(row["audio"]["bytes"]))
        # Раскладка фиксированная в датасете, но проверим на всякий случай
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            continue
        samples = struct.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes()))
        # Режем окнами по 1.2с без перекрытия; хвост короче половины окна выбрасываем
        for start in range(0, len(samples) - WINDOW // 2, WINDOW):
            chunk = samples[start:start + WINDOW]
            if len(chunk) < WINDOW:
                chunk = chunk + (0,) * (WINDOW - len(chunk))
            rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5 / 32768
            if rms < RMS_MIN:
                continue
            n_clips += 1
            out = wave.open(f"{OUT}/golos_{n_clips:05}.wav", "wb")
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(struct.pack(f"<{WINDOW}h", *chunk))
            out.close()
            if n_clips >= LIMIT:
                stop = True
                break
        if stop:
            break
    if stop:
        break

print(f"Нарезано {n_clips} клипов в {OUT}")
