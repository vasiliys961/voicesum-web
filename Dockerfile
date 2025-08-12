FROM python:3.10-slim

# Установка ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Настройка рабочей директории
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Создаём временную папку
RUN mkdir -p /tmp/voicesum

# Запуск
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
