# app.py - Ультра-легкая версия с ограничениями
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import whisper
from pydub import AudioSegment
from openai import OpenAI
import time
import logging
from httpx import Client as HttpxClient
import threading
import signal

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Уменьшено до 10 MB

# === Настройки ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Ленивая загрузка модели ===
whisper_model = None
llm_client = None

def load_whisper_model():
    """Загрузка модели только при первом использовании"""
    global whisper_model
    if whisper_model is None:
        logger.info("🎙️ Загружаю модель Whisper (tiny)...")
        try:
            whisper_model = whisper.load_model("tiny", device="cpu")
            logger.info("✅ Модель Whisper (tiny) загружена!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Whisper: {e}")
            raise
    return whisper_model

def load_llm_client():
    """Загрузка LLM клиента только при первом использовании"""
    global llm_client
    if llm_client is None and OPENROUTER_API_KEY:
        try:
            llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                http_client=HttpxClient(proxies=None, timeout=20.0),
            )
            logger.info("✅ OpenRouter клиент инициализирован!")
        except Exception as e:
            logger.error(f"❌ Ошибка OpenRouter: {e}")
            raise
    return llm_client

# === Функции с таймаутом ===

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Операция превысила таймаут")

def transcribe_with_timeout(wav_path, timeout_seconds=30):
    """Транскрипция с таймаутом"""
    def transcribe_thread():
        nonlocal result, error
        try:
            model = load_whisper_model()
            result = model.transcribe(wav_path, language=None, fp16=False, verbose=False)
        except Exception as e:
            error = e

    result = None
    error = None
    
    thread = threading.Thread(target=transcribe_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"❌ Транскрипция превысила таймаут {timeout_seconds}с")
        raise TimeoutError(f"Транскрипция превысила {timeout_seconds} секунд")
    
    if error:
        raise error
    
    if result:
        return result["text"].strip()
    
    raise Exception("Неизвестная ошибка транскрипции")

def generate_summary_simple(text):
    """Упрощенная генерация резюме"""
    try:
        client = load_llm_client()
        if not client:
            return "Резюме: API не настроен"
        
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {"role": "system", "content": "Создай краткое резюме на русском языке, максимум 2 предложения."},
                {"role": "user", "content": f"Текст: {text[:3000]}"}  # Еще меньше токенов
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка резюме: {e}")
        return f"Резюме недоступно: {str(e)[:50]}"

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "openrouter_configured": OPENROUTER_API_KEY is not None,
        "model": "tiny-lazy",
        "max_file_size": "10MB",
        "max_duration": "60s"
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()
    input_path = None
    wav_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Файл не загружен"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        # Уникальные имена файлов
        timestamp = int(time.time() * 1000)
        input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.tmp")
        wav_path = os.path.join(TEMP_DIR, f"converted_{timestamp}.wav")
        
        # Сохранение файла
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"📥 Файл сохранён: {file_size} байт")
        
        # Конвертация с жесткими ограничениями
        audio = AudioSegment.from_file(input_path)
        
        # Ограничение по длительности - максимум 60 секунд
        if len(audio) > 60000:  # 60 секунд
            audio = audio[:60000]
            logger.info("⏱️ Аудио обрезано до 60 секунд")
        
        # Агрессивное сжатие
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        
        logger.info(f"✅ Конвертация завершена за {time.time() - start_time:.1f}с")
        
        # Транскрипция с таймаутом
        transcribe_start = time.time()
        transcript = transcribe_with_timeout(wav_path, timeout_seconds=25)
        
        logger.info(f"🎙️ Транскрипция завершена за {time.time() - transcribe_start:.1f}с")
        
        if not transcript or len(transcript.strip()) < 3:
            return jsonify({"error": "Речь не распознана или слишком короткая"}), 400

        # Резюме
        summary = generate_summary_simple(transcript)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Обработка завершена за {total_time:.1f}с")

        return jsonify({
            "transcript": transcript,
            "summary": summary,
            "model_used": "whisper-tiny",
            "processing_time": f"{total_time:.1f}s",
            "audio_duration": f"{len(audio)/1000:.1f}s"
        })
        
    except TimeoutError as e:
        logger.error(f"⏰ Таймаут: {e}")
        return jsonify({"error": f"Обработка превысила лимит времени: {e}"}), 408
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка: {str(e)[:100]}"}), 500
    finally:
        # Очистка файлов
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# === Запуск ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✅ Сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
