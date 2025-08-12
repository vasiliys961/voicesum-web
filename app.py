# app.py - Оптимизированная версия для Koyeb
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
import gc

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Уменьшено до 50 MB

# === Настройки ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
    
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Глобальные переменные для моделей ===
whisper_model = None
llm_client = None

def initialize_models():
    """Инициализация моделей при первом запросе"""
    global whisper_model, llm_client
    
    if whisper_model is None:
        logger.info("🎙️ Загружаю модель Whisper (small)...")
        try:
            whisper_model = whisper.load_model("small", device="cpu")
            logger.info("✅ Модель Whisper загружена!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Whisper: {e}")
            return False
    
    if llm_client is None and OPENROUTER_API_KEY:
        try:
            llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                http_client=HttpxClient(
                    proxies=None,
                    timeout=30.0,
                ),
            )
            logger.info("✅ OpenRouter клиент инициализирован!")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации OpenRouter: {e}")
            return False
    
    return True

# === Вспомогательные функции ===

def split_audio(wav_path, chunk_length_sec=240):  # Уменьшено до 4 минут
    """Нарезка аудио на более короткие фрагменты"""
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
    """Транскрибирует длинное аудио с освобождением памяти"""
    if not whisper_model:
        raise Exception("Модель Whisper не загружена")
        
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
                
            # Принудительная очистка памяти
            del result
            gc.collect()
            
        except Exception as e:
            error_msg = f"[Ошибка фрагмента {i+1}: {str(e)[:50]}...] "
            full_transcript += error_msg
            logger.error(f"❌ Ошибка транскрипции: {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    
    return full_transcript.strip()

def generate_summary(text):
    """Генерирует резюме с обработкой ошибок"""
    if not llm_client:
        return "Ошибка: API ключ OpenRouter не настроен"
        
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
                        f"Разговор:\n\n{text[:10000]}"  # Ограничено для экономии токенов
                    )
                }
            ],
            max_tokens=400,
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

@app.route("/health")
def health():
    """Health check для Koyeb"""
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "openrouter_configured": llm_client is not None
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Инициализация моделей при первом запросе
    if not initialize_models():
        return jsonify({"error": "Ошибка инициализации моделей"}), 500
        
    if 'audio' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    input_path = os.path.join(TEMP_DIR, f"input_{int(time.time())}_{file.filename}")
    try:
        file.save(input_path)
        logger.info(f"📥 Файл сохранён: {input_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения: {e}")
        return jsonify({"error": "Не удалось сохранить файл"}), 500

    # Конвертация в WAV 16kHz mono
    wav_path = os.path.join(TEMP_DIR, f"converted_{int(time.time())}.wav")
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
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({"error": f"Ошибка обработки: {e}"}), 500

# === Запуск сервера ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✅ Сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
