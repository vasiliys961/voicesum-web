FROM python:3.10-slim

# Установка ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание временной директории
RUN mkdir -p /tmp/voicesum

# Переменная окружения для порта
ENV PORT=8000

# Запуск
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
