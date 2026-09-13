.PHONY: generate_tts record record_confusables record_speech chop check preprocess augment \
        negatives clone_sample download-data mfcc-check split-check train eval score stream \
        rebuild-dataset test

# Число дублей за сессию записи (по умолчанию 20).
# Переопределяется: make record N=50
N ?= 20

# Длительность непрерывной записи, секунды (clone_sample / record_speech).
SECS ?= 60

# ---------- позитивы ----------

generate_tts:
	cargo run --bin generate_tts

# Запись дублей «Гермес» через гарнитуру в dataset/raw/real.
# Идемпотентно: нумерация продолжается с уже записанных файлов.
record:
	cargo run --bin record -- dataset/raw/real $(N)

# Проверка записанных дублей: уровень, попадание слова в окно.
check:
	python3 scripts/qc.py dataset/raw/real

# Непрерывная запись связной речи для сэмпла voice cloning'а (Qwen).
clone_sample:
	cargo run --bin record_sample -- dataset/raw/clone_sample.wav $(SECS)

# ---------- негативы своим голосом (тот же микрофон, тот же ты) ----------

# Фонематические двойники: make record_confusables WORD=термес N=30
# Файлы: dataset/raw/confusables/conf_<WORD>_0001.wav ... Потом:
#   make preprocess IN=dataset/raw/confusables OUT=dataset/negative
WORD ?= термес
record_confusables:
	cargo run --bin record -- dataset/raw/confusables $(N) "$(WORD)" "conf_$(WORD)"

# Чтение любого текста вслух одним файлом: make record_speech SECS=300
# Каждый запуск — новый файл s<номер>.wav; слово «Гермес» не произносить.
record_speech:
	@mkdir -p dataset/raw/self_speech
	cargo run --bin record_sample -- dataset/raw/self_speech/s$$(printf '%02d' $$(( $$(ls dataset/raw/self_speech/*.wav 2>/dev/null | wc -l) + 1 ))).wav $(SECS) speech

# Нарезка записей self_speech на окна 1.2с (тихие выбрасываются). Потом:
#   make preprocess IN=dataset/raw/self_speech_clips OUT=dataset/negative
chop:
	cargo run --bin chop -- dataset/raw/self_speech dataset/raw/self_speech_clips

# ---------- негативы из готовых датасетов ----------

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

# ---------- общий формат ----------

# Препроцессинг любой папки: make preprocess IN=dataset/raw/real OUT=dataset/positive
preprocess:
	cargo run --bin preprocess -- $(IN) $(OUT)

# Аугментация позитивов (8 вариантов на файл, идемпотентно; фоны из raw/background)
augment:
	cargo run --bin augment -- dataset/positive dataset/positive

# Полная пересборка производных данных (raw/ не трогается). Нужна после
# смены preprocess/augment/extract_golos — старые файлы иначе останутся
# (пайплайн только добавляет, никогда не удаляет).
rebuild-dataset:
	rm -rf dataset/positive dataset/negative dataset/raw/golos_clips
	uv run --with pyarrow python3 scripts/extract_golos.py 4000
	cargo run --bin preprocess -- dataset/raw/real dataset/positive
	cargo run --bin preprocess -- dataset/raw/tts  dataset/positive
	cargo run --bin augment    -- dataset/positive dataset/positive
	cargo run --bin preprocess -- dataset/raw/sc_sample dataset/negative
	cargo run --bin preprocess -- dataset/raw/golos_clips dataset/negative
	@test -d dataset/raw/self_speech_clips && cargo run --bin preprocess -- dataset/raw/self_speech_clips dataset/negative || true
	@test -d dataset/raw/confusables && cargo run --bin preprocess -- dataset/raw/confusables dataset/negative || true

# ---------- проверки ----------

# Посмотреть MFCC-признаки файла «глазами модели»:
# make mfcc-check FILE=dataset/positive/germes_real_0020.wav
mfcc-check:
	cargo run --bin mfcc_check -- $(FILE)

# Размеры и баланс сплитов + проверка отсутствия утечки данных
split-check:
	cargo run --bin split_check

# Юнит-тесты библиотеки (MFCC, дизер, группировка)
test:
	cargo test --lib

# ---------- обучение и оценка ----------

# Обучение CNN: make train [E=10] [BATCH=64] [LR=0.001] [DROPOUT=0] [WD=0]
# Лучшая эпоха → models/hermes.bin, кривая → models/train_log.csv
train:
	cargo run --release --bin train -- $(or $(E),10) $(or $(BATCH),64) $(or $(LR),0.001) $(or $(DROPOUT),0) $(or $(WD),0)

# Кривая порогов на val (по умолчанию) или test: make eval SPLIT=test
eval:
	cargo run --release --bin eval -- eval $(or $(SPLIT),val)

# Скоры модели на любой папке wav 16kHz/1.2с: make score DIR=dataset/negative [T=0.5]
score:
	cargo run --release --bin eval -- score $(DIR) $(or $(T),0.5)

# Ложные тревоги в час на длинной записи: make stream FILE=ambient.wav [HOP=100]
stream:
	cargo run --release --bin eval -- stream $(FILE) $(or $(HOP),100)
