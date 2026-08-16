.PHONY: generate_tts record check preprocess augment negatives

# Число дублей за сессию записи (по умолчанию 20).
# Переопределяется: make record N=50
N ?= 20

generate_tts:
	cargo run --bin generate_tts

# Запись дублей «Гермес» через гарнитуру в dataset/raw/real.
# Идемпотентно: нумерация продолжается с уже записанных файлов.
record:
	cargo run --bin record -- dataset/raw/real $(N)

# Проверка записанных дублей: уровень, попадание слова в окно.
check:
	python3 scripts/qc.py dataset/raw/real

# Препроцессинг любой папки: make preprocess IN=dataset/raw/real OUT=dataset/positive
preprocess:
	cargo run --bin preprocess -- $(IN) $(OUT)

# Аугментация позитивов (8 вариантов на файл, идемпотентно)
augment:
	cargo run --bin augment -- dataset/positive dataset/positive

# Полный пайплайн негативов: выборка Speech Commands + нарезка Golos + preprocess
negatives:
	python3 scripts/sample_speech_commands.py 2000
	uv run --with pyarrow python3 scripts/extract_golos.py 4000
	cargo run --bin preprocess -- dataset/raw/sc_sample dataset/negative
	cargo run --bin preprocess -- dataset/raw/golos_clips dataset/negative
