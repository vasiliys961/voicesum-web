# app.py - Версия для длинных аудиофайлов
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
import math

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# === Настройки ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Ленивая загрузка модели ===
whisper_model = None
llm_client = None

def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        logger.info("🎙️ Загружаю модель Whisper (tiny для длинных файлов)...")
        try:
            whisper_model = whisper.load_model("tiny", device="cpu")  # tiny для скорости
            logger.info("✅ Модель Whisper (tiny) загружена!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Whisper: {e}")
            raise
    return whisper_model

def load_llm_client():
    global llm_client
    if llm_client is None and OPENROUTER_API_KEY:
        try:
            llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                http_client=HttpxClient(proxies=None, timeout=60.0),
            )
            logger.info("✅ OpenRouter клиент инициализирован!")
        except Exception as e:
            logger.error(f"❌ Ошибка OpenRouter: {e}")
            raise
    return llm_client

def split_audio_into_chunks(wav_path, chunk_duration_minutes=4):
    """Разбивает аудио на куски по N минут"""
    try:
        audio = AudioSegment.from_wav(wav_path)
        chunk_length_ms = chunk_duration_minutes * 60 * 1000  # в миллисекундах
        chunks = []
        total_duration_ms = len(audio)
        num_chunks = math.ceil(total_duration_ms / chunk_length_ms)

        logger.info(f"📁 Разбиваю аудио на {num_chunks} частей по {chunk_duration_minutes} мин")

        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = min(start + chunk_length_ms, total_duration_ms)
            chunk = audio[start:end]
            
            chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}_{int(time.time())}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append({
                'path': chunk_path,
                'index': i,
                'start_time': start / 1000,  # в секундах
                'duration': len(chunk) / 1000
            })
        
        return chunks, total_duration_ms / 1000  # возвращаем общую длительность в секундах
    except Exception as e:
        logger.error(f"❌ Ошибка разбивки аудио: {e}")
        raise

def transcribe_chunk_with_timeout(chunk_info, timeout_seconds=60):
    """Транскрибирует один кусок с таймаутом"""
    def transcribe_thread():
        nonlocal result, error
        try:
            model = load_whisper_model()
            transcript_result = model.transcribe(
                chunk_info['path'], 
                language=None, 
                fp16=False, 
                verbose=False
            )
            result = transcript_result["text"].strip()
        except Exception as e:
            error = e

    result = None
    error = None
    
    thread = threading.Thread(target=transcribe_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"❌ Транскрипция куска {chunk_info['index']} превысила таймаут")
        return f"[Кусок {chunk_info['index']+1}: таймаут транскрипции]"
    
    if error:
        logger.error(f"❌ Ошибка транскрипции куска {chunk_info['index']}: {error}")
        return f"[Кусок {chunk_info['index']+1}: ошибка транскрипции]"
    
    return result or f"[Кусок {chunk_info['index']+1}: пустой результат]"

def transcribe_long_audio(wav_path):
    """Транскрибирует длинное аудио по частям"""
    chunks, total_duration = split_audio_into_chunks(wav_path, chunk_duration_minutes=4)
    full_transcript = ""
    successful_chunks = 0
    
    logger.info(f"🎙️ Начинаю транскрипцию {len(chunks)} частей...")
    
    for i, chunk_info in enumerate(chunks):
        try:
            logger.info(f"🔄 Обрабатываю часть {i+1}/{len(chunks)} ({chunk_info['duration']:.1f}с)")
            
            transcript = transcribe_chunk_with_timeout(chunk_info, timeout_seconds=45)
            
            if transcript and not transcript.startswith("[Кусок"):
                # Добавляем временную метку
                start_min = int(chunk_info['start_time'] // 60)
                start_sec = int(chunk_info['start_time'] % 60)
                full_transcript += f"\n\n[{start_min:02d}:{start_sec:02d}] {transcript}"
                successful_chunks += 1
            else:
                full_transcript += f"\n\n{transcript}"
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки части {i+1}: {e}")
            full_transcript += f"\n\n[Часть {i+1}: ошибка обработки]"
        finally:
            # Удаляем временный файл куска
            if os.path.exists(chunk_info['path']):
                os.remove(chunk_info['path'])
    
    logger.info(f"✅ Транскрипция завершена: {successful_chunks}/{len(chunks)} частей успешно")
    return full_transcript.strip(), successful_chunks, len(chunks), total_duration

def generate_long_summary(text):
    """Генерирует резюме для длинного текста"""
    try:
        client = load_llm_client()
        if not client:
            return "Резюме: API не настроен"
        
        # Для очень длинного текста берем первые 15000 символов
        text_for_summary = text[:15000]
        if len(text) > 15000:
            text_for_summary += "\n\n[...текст сокращен для анализа...]"
        
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Создай подробное структурированное резюме длинной аудиозаписи на русском языке. "
                        "Выдели основные темы, ключевые моменты, решения и выводы. "
                        "Организуй информацию по разделам."
                    )
                },
                {"role": "user", "content": f"Транскрипт длинной записи:\n\n{text_for_summary}"}
            ],
            max_tokens=800,  # Больше токенов для длинного резюме
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка резюме: {e}")
        return f"Резюме недоступно: {str(e)[:100]}"

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "model": "tiny-chunks",
        "openrouter_configured": OPENROUTER_API_KEY is not None,
        "max_file_size": "100MB",
        "max_duration": "unlimited",
        "chunk_size": "4min"
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

        timestamp = int(time.time() * 1000)
        input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.tmp")
        wav_path = os.path.join(TEMP_DIR, f"converted_{timestamp}.wav")
        
        # Сохранение файла
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"📥 Файл сохранён: {file_size / 1024 / 1024:.1f} MB")
        
        # Конвертация
        logger.info("🔄 Конвертирую аудио...")
        audio = AudioSegment.from_file(input_path)
        original_duration = len(audio) / 1000 / 60  # в минутах
        
        logger.info(f"📊 Исходная длительность: {original_duration:.1f} минут")
        
        # Конвертируем в стандартный формат
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        
        logger.info("✅ Конвертация завершена, начинаю транскрипцию...")
        
        # Транскрипция длинного аудио
        transcript, successful_chunks, total_chunks, duration = transcribe_long_audio(wav_path)
        
        if not transcript or len(transcript.strip()) < 10:
            return jsonify({"error": "Не удалось получить транскрипцию"}), 400

        # Генерируем резюме
        logger.info("📝 Генерирую резюме...")
        summary = generate_long_summary(transcript)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Полная обработка завершена за {total_time/60:.1f} минут")

        return jsonify({
            "transcript": transcript,
            "summary": summary,
            "statistics": {
                "processing_time": f"{total_time:.1f}s",
                "processing_time_min": f"{total_time/60:.1f}min",
                "audio_duration": f"{duration/60:.1f}min",
                "file_size": f"{file_size / 1024 / 1024:.1f}MB",
                "successful_chunks": successful_chunks,
                "total_chunks": total_chunks,
                "success_rate": f"{successful_chunks/total_chunks*100:.1f}%"
            },
            "model_used": "whisper-tiny-chunks"
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка: {str(e)[:150]}"}), 500
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
