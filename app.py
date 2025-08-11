# app.py
from flask import Flask, render_template_string, request, jsonify
import os
import tempfile
import time
import logging
from pathlib import Path
import zipfile
import shutil
import wave
import json

from pydub import AudioSegment
import httpx
from openai import OpenAI

# ===== ЛОГИ =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicesum")

# ===== FLASK =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# ===== ВРЕМЕННАЯ ПАПКА =====
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# ===== КЛЮЧИ / НАСТРОЙКИ =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning("⚠️ OPENROUTER_API_KEY не задан — суммаризация работать не будет.")

# ===== OpenRouter (через OpenAI SDK) =====
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=30.0,
)

# ===== Vosk: скачивание small-модели и инициализация =====
VOSK_MODEL_URL = os.getenv(
    "VOSK_MODEL_URL",
    "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
)
VOSK_DIR = Path(TEMP_DIR) / "vosk_model"
_vosk_ready = False
_asr_model = None

def ensure_vosk_model():
    """Скачивает и распаковывает Vosk-модель в TEMP_DIR при отсутствии."""
    if VOSK_DIR.exists() and any(VOSK_DIR.iterdir()):
        logger.info("✅ Модель Vosk уже распакована")
        return
    zip_path = Path(TEMP_DIR) / "vosk_model.zip"
    logger.info("⬇️ Скачивание Vosk-модели…")
    with httpx.stream("GET", VOSK_MODEL_URL, timeout=300.0) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    logger.info("📦 Распаковка Vosk-модели…")
    with zipfile.ZipFile(zip_path, "r") as z:
        root = z.namelist()[0].split("/")[0]
        z.extractall(TEMP_DIR)
    extracted = Path(TEMP_DIR) / root
    if VOSK_DIR.exists():
        shutil.rmtree(VOSK_DIR)
    extracted.rename(VOSK_DIR)
    zip_path.unlink(missing_ok=True)
    logger.info(f"✅ Модель готова: {VOSK_DIR}")

def init_vosk():
    """Ленивая инициализация Vosk-модели."""
    global _vosk_ready, _asr_model
    if _vosk_ready:
        return
    ensure_vosk_model()
    from vosk import Model
    _asr_model = Model(str(VOSK_DIR))
    _vosk_ready = True
    logger.info("🧠 Vosk инициализирован")

# ===== УТИЛИТЫ АУДИО =====
def convert_to_wav(input_path: str) -> str:
    """Конвертация любого аудио → WAV 16 kHz mono."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = os.path.join(TEMP_DIR, f"{int(time.time()*1000)}.wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_vosk(wav_path: str) -> str:
    """Оффлайн ASR через Vosk small."""
    init_vosk()
    from vosk import KaldiRecognizer
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000):
        raise RuntimeError("WAV должен быть PCM 16-bit mono 8/16 kHz (это делает convert_to_wav)")
    rec = KaldiRecognizer(_asr_model, wf.getframerate())
    rec.SetWords(True)
    parts = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "text" in res:
                parts.append(res["text"])
    final = json.loads(rec.FinalResult())
    if "text" in final:
        parts.append(final["text"])
    return " ".join(t for t in parts if t).strip()

# ===== РЕЗЮМЕ (LLM) =====
from textwrap import dedent

def generate_summary(text: str) -> str:
    if not OPENROUTER_API_KEY:
        return "Суммаризация недоступна: не задан OPENROUTER_API_KEY."

    prompt = dedent(f"""\
        Ты — лаконичный аналитик. Выдай строго в формате:
        Краткое резюме: <1–2 предложения>
        — Решения: <кратко>
        — Действия: <кратко>
        — Договорённости: <кратко>
        — Темы: <кратко>

        Разговор:

        {text[:12000]}
    """)

    r = llm_client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "system", "content": "Ты — лаконичный аналитик."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

# ===== UI =====
INDEX_HTML = """
<!doctype html><html lang="ru"><meta charset="utf-8">
<title>VoiceSum</title>
<h1>Загрузите аудио → транскрипция (Vosk) → резюме (OpenRouter)</h1>
<form method="POST" action="/transcribe" enctype="multipart/form-data">
  <input type="file" name="audio" accept="audio/*" required>
  <button type="submit">Отправить</button>
</form>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400
    f = request.files['audio']
    if not f.filename:
        return jsonify({"error": "Файл не выбран"}), 400

    src = os.path.join(TEMP_DIR, f.filename)
    try:
        f.save(src)
        wav = convert_to_wav(src)
    except Exception as e:
        logger.exception("Ошибка конвертации")
        return jsonify({"error": f"Ошибка конвертации: {e}"}), 500
    finally:
        try:
            if os.path.exists(src):
                os.remove(src)
        except:
            pass

    try:
        transcript = transcribe_vosk(wav)
    except Exception as e:
        logger.exception("Ошибка транскрипции")
        transcript = f"Транскрипция недоступна: {e}"
    finally:
        try:
            if os.path.exists(wav):
                os.remove(wav)
        except:
            pass

    try:
        summary = generate_summary(transcript if transcript else "Пустая транскрипция.")
    except Exception as e:
        logger.exception("Ошибка суммаризации")
        summary = f"Ошибка генерации резюме: {e}"

    return jsonify({"transcript": transcript, "summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

