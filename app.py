# app.py - Облегченная версия для тестирования
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import whisper
from pydub import AudioSegment
from openai import OpenAI
import time
import math
import logging
from httpx import Client as HttpxClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB

# === Настройки ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Загрузка ЛЕГКОЙ модели Whisper ===
logger.info("🎙️ Загружаю модель Whisper (tiny)...")
try:
    whisper_model = whisper.load_model("tiny", device="cpu")  # САМАЯ ЛЕГКАЯ МОДЕЛЬ
    logger.info("✅ Модель Whisper (tiny) загружена!")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки Whisper: {e}")
    whisper_model = None

# === Клиент OpenRouter ===
llm_client = None
if OPENROUTER_API_KEY:
    try:
        llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            http_client=HttpxClient(proxies=None, timeout=30.0),
        )
        logger.info("✅ OpenRouter клиент инициализирован!")
    except Exception as e:
        logger.error(f"❌ Ошибка OpenRouter: {e}")

# === Функции ===

def transcribe_audio_simple(wav_path):
    """Простая транскрипция без нарезки"""
    if not whisper_model:
        raise Exception("Модель Whisper не загружена")
    
    try:
        result = whisper_model.transcribe(wav_path, language=None, fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"❌ Ошибка транскрипции: {e}")
        raise

def generate_summary_simple(text):
    """Упрощенная генерация резюме"""
    if not llm_client:
        return "Резюме: Не удалось сгенерировать (API не настроен)"
    
    try:
        response = llm_client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {"role": "system", "content": "Создай краткое резюме на русском языке, до 3 предложений."},
                {"role": "user", "content": f"Текст: {text[:5000]}"}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка резюме: {e}")
        return f"Ошибка резюме: {str(e)[:100]}"

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "openrouter_configured": llm_client is not None,
        "model": "tiny"
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not whisper_model:
        return jsonify({"error": "Модель Whisper не загружена"}), 500
        
    if 'audio' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    input_path = os.path.join(TEMP_DIR, f"input_{int(time.time())}.tmp")
    wav_path = os.path.join(TEMP_DIR, f"converted_{int(time.time())}.wav")
    
    try:
        # Сохранение файла
        file.save(input_path)
        logger.info(f"📥 Файл сохранён: {input_path}")
        
        # Конвертация (упрощенная)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Ограничиваем длительность до 2 минут для tiny модели
        if len(audio) > 120000:  # 2 минуты в миллисекундах
            audio = audio[:120000]
            logger.info("⏱️ Аудио обрезано до 2 минут")
        
        audio.export(wav_path, format="wav")
        logger.info(f"✅ Конвертация успешна")
        
        # Транскрипция
        transcript = transcribe_audio_simple(wav_path)
        if not transcript:
            return jsonify({"error": "Не удалось распознать речь"}), 400

        # Резюме
        summary = generate_summary_simple(transcript)

        return jsonify({
            "transcript": transcript,
            "summary": summary,
            "model_used": "whisper-tiny"
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка: {e}"}), 500
    finally:
        # Очистка
        for path in [input_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)

# === Запуск ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✅ Сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
