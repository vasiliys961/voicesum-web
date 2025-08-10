import whisper

print("Загружаю модель whisper...")
model = whisper.load_model("small")
print("✅ Модель загружена!")