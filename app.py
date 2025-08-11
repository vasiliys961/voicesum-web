# app.py
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import openai
from pydub import AudioSegment
import time
import logging
from httpx import Client as HttpxClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# === Настройки ===
OPENROUTER_API_KEY = "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"
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Клиент для OpenRouter (Whisper + LLM) ===
whisper_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=HttpxClient(timeout=60.0),  # аудио могут быть длинными
)

llm_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=HttpxClient(timeout=30.0),
)

# === Вспомогательные функции ===

def transcribe_audio_with_openrouter(audio_path):
    """Транскрибирует аудио через OpenRouter → OpenAI Whisper"""
    try:
        with open(audio_path, "rb") as f:
            response = whisper_client.audio.transcriptions.create(
                model="openai/whisper-v3",  # ← это настоящий Whisper от OpenAI
                file=f,
                language="ru",  # можно убрать, если хочешь автоопределение
            )
        return response.text.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка транскрипции через API: {e}")
        raise

def generate_summary(text):
    """Генерирует резюме через OpenRouter (Claude Haiku)"""
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
        transcript = transcribe_audio_with_openrouter(wav_path)
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
        return jsonify({"error": f"Ошибка транскрипции: {e}"}), 500

# === Запуск сервера ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)