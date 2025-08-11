# app.py
import streamlit as st
import openai
import os
import tempfile
import base64
from pydub import AudioSegment
from httpx import Client as HttpxClient

# Настройка страницы
st.set_page_config(page_title="VoiceSum AI", layout="centered")
st.title("🎙️ VoiceSum AI")

# === Настройки ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENAI_API_KEY or not OPENROUTER_API_KEY:
    st.error("🔑 Не задан API-ключ. Проверь переменные окружения.")
    st.stop()

# Клиенты
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=HttpxClient(timeout=60.0))
llm_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=HttpxClient(timeout=30.0),
)

# Временная папка
temp_dir = tempfile.mkdtemp(prefix="voicesum_")

# === Функции ===

def transcribe_audio(file_path):
    """Транскрибирует аудио через OpenAI Whisper"""
    try:
        with open(file_path, "rb") as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru"
            )
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Ошибка транскрипции: {e}")
        return None

def generate_summary(text):
    """Генерирует резюме через Claude через OpenRouter"""
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
        st.error(f"❌ Ошибка генерации: {e}")
        return None

# === Интерфейс ===

st.write("Загрузите аудиофайл или запишите голос с микрофона.")

# Выбор: загрузка или запись
mode = st.radio("Выберите способ", ["📁 Загрузить файл", "🎤 Запись с микрофона"])

audio_file = None

if mode == "📁 Загрузить файл":
    audio_file = st.file_uploader("Загрузите аудиофайл", type=["wav", "mp3", "webm", "m4a"])
elif mode == "🎤 Запись с микрофона":
    audio_file = st.audio_input("Запишите голос")

if audio_file:
    # Сохраняем файл
    input_path = os.path.join(temp_dir, audio_file.name)
    with open(input_path, "wb") as f:
        f.write(audio_file.read())

    # Конвертируем в WAV 16kHz mono
    wav_path = os.path.join(temp_dir, "input.wav")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
    except Exception as e:
        st.error(f"❌ Ошибка конвертации: {e}")
        st.stop()

    if st.button("🔄 Начать обработку"):
        with st.spinner("Транскрибирую..."):
            transcript = transcribe_audio(wav_path)
        
        if transcript:
            st.subheader("📝 Транскрипция")
            st.write(transcript)

            with st.spinner("Генерирую резюме..."):
                summary = generate_summary(transcript)
            
            if summary:
                st.subheader("📋 Резюме (на русском)")
                st.write(summary)