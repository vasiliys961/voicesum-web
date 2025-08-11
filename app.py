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

# —— Логирование ——
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicesum")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# —— Настройки ——
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning("⚠️ OPENROUTER_API_KEY не задан — суммаризация работать не будет.")

# —— LLM клиент (OpenRouter) ——
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# —— Vosk: скачивание и инициализация small-модели ——
VOSK_MODEL_URL = os.getenv(
    "VOSK_MODEL_URL",
    "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
)
VOSK_DIR = Path(TEMP_DIR) / "vosk_model"
_vosk_ready = False
_asr_model = None

def ensure_vosk_model():
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
    global _vosk_ready, _asr_model
    if _vosk_ready:
        return
    ensure_vosk_model()
    from vosk import Model
    _asr_model = Model(str(VOSK_DIR))
    _vosk_ready = True
    logger.info("🧠 Vosk инициализирован")

def convert_to_wav(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = os.path.join(TEMP_DIR, f"{int(time.time()*1000)}.wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_vosk(wav_path: str) -> str:
    init_vosk()
    from vosk import KaldiRecognizer
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000,16000):
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

def generate_summary(text: str) -> str:
    if not OPENROUTER_API_KEY:
        return "Суммаризация недоступна: не задан OPENROUTER_API_KEY."
    prompt = (
        "Ты — лаконичный аналитик. Выдай:\n"
        "Краткое резюме: <1–2 предложения>\n"
        "— Решения: <кратко>\n"
        "— Действия: <кратко>\n"
        "— Договорённости: <кратко>\n"
        "— Темы: <кратко>\n\n"
        f"Разговор:\n\n{text
