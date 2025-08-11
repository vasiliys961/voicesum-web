# app.py
from flask import Flask, render_template, request, jsonify
import os
import tempfile
from pydub import AudioSegment
from openai import OpenAI
import time
import math
import logging
from httpx import Client as HttpxClient

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# === Настройки ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Загрузка модели Whisper (tiny) ===
logger.info("🎙️ Загружаю модель Whisper (tiny)...")
whisper_model = whisper.load_model("tiny", device="cpu")
logger.info("✅ Модель Whisper загружена!")

# === Клиент OpenRouter (только для генерации резюме) ===
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=HttpxClient(timeout=30.0),
)

# === Вспомогательные функции ===

def split_audio(wav_path, chunk_length_sec=300):
    try:
        audio = AudioSegment.from_wav(wav_path)
        chunk_length_ms = chunk_length_sec * 1000
        chunks = []
        total_duration_ms = len(audio)
        num_chunks = math.ceil(total_duration_ms / chunk_length_ms)

        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = min(start + chunk_length_ms, total_duration_ms)
            chunk = audio[start:end]
            chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}_{int(time.time())}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        logger.info(f"✂️ Аудио нарезано на {len(chunks)} фрагментов по ~{chunk_length_sec} сек.")
        return chunks
    except Exception as e:
        logger.error(f"❌ Ошибка нарезки аудио: {e}")
        raise

def transcribe_very_long_audio(wav_path):
    full_transcript = ""
    chunk_paths = split_audio(wav_path)

    for i, chunk_path in enumerate(chunk_paths):
        try:
            if os.path.getsize(chunk_path) == 0:
                full_transcript += f"[Фрагмент {i+1} пуст] "
                continue

            result = whisper_model.transcribe(
                chunk_path,
                language=None,
                fp16=False,
                verbose=False
            )
            text = result["text"].strip()
            if text:
                full_transcript += text + " "
            else:
                full_transcript += f"[Фрагмент {i+1} — речь не распознана] "
        except Exception as e:
            error_msg = f"[Ошибка фрагмента {i+1}: {str(e)[:50]}...] "
            full_transcript += error_msg
            logger.error(f"❌ Ошибка транскрипции: {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    return full_transcript.strip()

def generate_summary(text):
    try:
        response = llm_client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — лаконичный аналитик. Твоя задача: за 5–8 коротких строк выдать сжатый результат без лишних слов. Пиши по-русски. "
                        "Никаких эмодзи, маркетинговых фраз, вводных вроде «Итог:». Если факта нет — не выдумывай. Объединяй дубли и формулируй обобщённо."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Проанализируй разговор или доклад ниже и выдай результат строго в формате:\n"
                        f"Краткое резюме (1–2 предложения, максимум 220 символов).\n"
                        f"3–5 маркеров, каждый в одной строке, по категориям: решения / действия / договорённости / темы. Если категория не встречается — пропусти её. Если пунктов много — сгруппируй и оставь самое важное.\n"
                        f"Требования к стилю:\n"
                        f"Короткие фразы, глаголы в повелительном или инфинитиве (для действий).\n"
                        f"Без оценочных суждений и советов, только факты из разговора.\n"
                        f"Не повторяй одно и то же разными словами.\n"
                        f"Не превышай 6 строк после резюме.\n"
                        f"Формат вывода (строго):\n"
                        f"Краткое резюме: <1–2 предложения>\n"
                        f"— Решения: <кратко>\n"
                        f"— Действия: <кратко>\n"
                        f"— Договорённости: <кратко>\n"
                        f"— Темы: <кратко>\n\n"
                        f"Разговор:\n\n{text[:12000]}"
                    )
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка генерации резюме: {e}")
        return f"Ошибка генерации резюме: {e}"

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    input_path = os.path.join(TEMP_DIR, file.filename)
    try:
        file.save(input_path)
        logger.info(f"📥 Файл сохранён: {input_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения: {e}")
        return jsonify({"error": "Не удалось сохранить файл"}), 500

    # Конвертация в WAV 16kHz mono
    wav_path = os.path.join(TEMP_DIR, f"{int(time.time())}.wav")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        logger.info(f"✅ Конвертация успешна: {wav_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка конвертации: {e}")
        return jsonify({"error": f"Ошибка конвертации: {e}"}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    try:
        transcript = transcribe_very_long_audio(wav_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if not transcript:
            return jsonify({"error": "Не удалось распознать речь"}), 400

        summary = generate_summary(transcript)

        return jsonify({
            "transcript": transcript,
            "summary": summary
        })
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка обработки: {e}"}), 500

# === Запуск сервера ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)