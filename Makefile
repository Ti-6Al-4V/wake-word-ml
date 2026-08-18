.PHONY: generate_tts record check preprocess augment negatives clone_sample download-data

# Число дублей за сессию записи (по умолчанию 20).
# Переопределяется: make record N=50
N ?= 20

# Длительность сэмпла голоса для клонирования, секунды (по умолчанию 60).
# Переопределяется: make clone_sample SECS=90
SECS ?= 60

generate_tts:
	cargo run --bin generate_tts

# Запись дублей «Гермес» через гарнитуру в dataset/raw/real.
# Идемпотентно: нумерация продолжается с уже записанных файлов.
record:
	cargo run --bin record -- dataset/raw/real $(N)

# Непрерывная запись связной речи для сэмпла voice cloning'а (Qwen).
clone_sample:
	cargo run --bin record_sample -- dataset/raw/clone_sample.wav $(SECS)

# Проверка записанных дублей: уровень, попадание слова в окно.
check:
	python3 scripts/qc.py dataset/raw/real

# Препроцессинг любой папки: make preprocess IN=dataset/raw/real OUT=dataset/positive
preprocess:
	cargo run --bin preprocess -- $(IN) $(OUT)

# Аугментация позитивов (8 вариантов на файл, идемпотентно)
augment:
	cargo run --bin augment -- dataset/positive dataset/positive

# Скачивание исходников негативов (~3.5GB): Speech Commands + зеркало Golos.
# Возобновляемое (curl -C -): после обрыва просто перезапустить.
# Запускать ПЕРЕД make negatives.
download-data:
	mkdir -p dataset/raw/downloads dataset/raw/speech_commands
	curl -L -C - -o dataset/raw/downloads/speech_commands_v0.02.tar.gz \
		"https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
	tar -xzf dataset/raw/downloads/speech_commands_v0.02.tar.gz -C dataset/raw/speech_commands
	for i in 0 1 2; do \
		curl -sSL -C - -o dataset/raw/downloads/golos_test_$$i.parquet \
			"https://huggingface.co/datasets/bonlime/golos-test/resolve/main/crowd/test-0000$$i-of-00003.parquet"; \
	done

# Полный пайплайн негативов: выборка Speech Commands + нарезка Golos + preprocess
negatives:
	python3 scripts/sample_speech_commands.py 2000
	uv run --with pyarrow python3 scripts/extract_golos.py 4000
	cargo run --bin preprocess -- dataset/raw/sc_sample dataset/negative
	cargo run --bin preprocess -- dataset/raw/golos_clips dataset/negative
