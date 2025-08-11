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

# === Клиент OpenRouter (работает с аудио!) ===
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=HttpxClient(timeout=60.0),
)

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

    # Сохраняем файл
    input_path = os.path.join(TEMP_DIR, file.filename)
    try:
        file.save(input_path)
        logger.info(f"📥 Файл сохранён: {input_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения: {e}")
        return jsonify({"error": "Не удалось сохранить файл"}), 500

    try:
        # Отправляем аудио напрямую в OpenRouter — он сам транскрибирует!
        response = client.chat.completions.create(
            model="openai/whisper-v3",  # ← специальная модель для транскрипции
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Пожалуйста, транскрибируй это аудио на русском языке."
                        },
                        {
                            "type": "audio",
                            "audio": {
                                "url": f"file://{input_path}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=256
        )
        transcript = response.choices[0].message.content.strip()
        logger.info("✅ Транскрипция получена")
    except Exception as e:
        logger.error(f"❌ Ошибка транскрипции: {e}")
        return jsonify({"error": f"Ошибка транскрипции: {e}"}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    if not transcript:
        return jsonify({"error": "Не удалось распознать речь"}), 400

    try:
        # Генерация резюме через Claude
        summary_response = client.chat.completions.create(
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
                        f"Разговор:\n\n{transcript[:12000]}"
                    )
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        summary = summary_response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка генерации резюме: {e}")
        summary = f"Ошибка генерации резюме: {e}"

    return jsonify({
        "transcript": transcript,
        "summary": summary
    })

# === Запуск сервера ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)