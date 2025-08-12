# app.py - Версия без зависимости от ffmpeg
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import assemblyai as aai
from openai import OpenAI
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# === API ключи ===
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "fb277d535ab94838bc14cc2f687b30be")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_aai_")

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Настройка AssemblyAI ===
aai.settings.api_key = ASSEMBLYAI_API_KEY
logger.info("🔑 AssemblyAI API ключ настроен")

# === Конфигурация транскрипции ===
def get_transcription_config():
    """Возвращает конфигурацию с максимальными возможностями"""
    return aai.TranscriptionConfig(
        # === Основные настройки ===
        speech_model=aai.SpeechModel.best,  # Лучшая модель
        language_code="ru",  # Русский язык
        
        # === AI функции (бесплатные!) ===
        speaker_labels=True,  # Определение спикеров
        speakers_expected=2,  # Ожидаемое количество
        
        auto_highlights=True,  # Ключевые моменты
        auto_chapters=True,   # Автоматические главы
        
        sentiment_analysis=True,  # Анализ эмоций
        entity_detection=True,    # Определение сущностей
        
        content_safety=True,      # Модерация контента
        iab_categories=True,      # Категоризация по темам
        
        summarization=True,       # Встроенное резюме
        summary_model=aai.SummarizationModel.informative,
        summary_type=aai.SummarizationType.bullets,
        
        # === Настройки качества ===
        punctuate=True,          # Пунктуация
        format_text=True,        # Форматирование текста
        dual_channel=False,      # Моно обработка
        
        # === Дополнительные функции ===
        disfluencies=False,      # Убираем "эм", "ах"
        filter_profanity=False,  # Не фильтруем мат
        redact_pii=False,        # Не скрываем персональные данные
    )

# === Умный генератор резюме ===
class AdvancedSummarizer:
    def __init__(self, openrouter_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        ) if openrouter_key else None
    
    def create_smart_summary(self, transcript_result):
        """Создает умное резюме на основе всех данных AssemblyAI"""
        if not self.client:
            return self._create_basic_summary(transcript_result)
        
        try:
            # Собираем все данные
            transcript = transcript_result.text
            chapters = getattr(transcript_result, 'chapters', None) or []
            highlights = getattr(transcript_result, 'auto_highlights', None)
            sentiment = getattr(transcript_result, 'sentiment_analysis_results', None) or []
            entities = getattr(transcript_result, 'entities', None) or []
            builtin_summary = getattr(transcript_result, 'summary', None)
            
            # Формируем богатый контекст
            context = f"ПОЛНЫЙ ТРАНСКРИПТ:\n{transcript[:25000]}\n\n"
            
            if chapters:
                context += "📚 АВТОМАТИЧЕСКИЕ ГЛАВЫ:\n"
                for i, chapter in enumerate(chapters[:10], 1):
                    headline = getattr(chapter, 'headline', f'Глава {i}')
                    start_time = getattr(chapter, 'start', 0) / 1000 / 60  # в минутах
                    context += f"{i}. {headline} ({start_time:.1f}мин)\n"
                context += "\n"
            
            if highlights and hasattr(highlights, 'results'):
                context += "💡 КЛЮЧЕВЫЕ МОМЕНТЫ:\n"
                for highlight in highlights.results[:15]:
                    text = getattr(highlight, 'text', '')
                    rank = getattr(highlight, 'rank', 0)
                    if text:
                        context += f"• {text} (важность: {rank:.2f})\n"
                context += "\n"
            
            if entities:
                context += "🏷️ УПОМЯНУТЫЕ СУЩНОСТИ:\n"
                entity_groups = {}
                for entity in entities[:25]:
                    entity_type = getattr(entity, 'entity_type', 'other')
                    entity_text = getattr(entity, 'text', '')
                    if entity_type not in entity_groups:
                        entity_groups[entity_type] = []
                    if entity_text not in entity_groups[entity_type]:
                        entity_groups[entity_type].append(entity_text)
                
                for entity_type, texts in entity_groups.items():
                    context += f"  {entity_type}: {', '.join(texts[:5])}\n"
                context += "\n"
            
            if builtin_summary:
                context += f"🤖 БАЗОВОЕ РЕЗЮМЕ ASSEMBLYAI:\n{builtin_summary}\n\n"
            
            # Анализ тональности
            if sentiment:
                positive_count = sum(1 for s in sentiment if getattr(s, 'sentiment', '') == 'POSITIVE')
                negative_count = sum(1 for s in sentiment if getattr(s, 'sentiment', '') == 'NEGATIVE')
                total_sentiment = len(sentiment)
                if total_sentiment > 0:
                    context += f"🎭 ОБЩАЯ ТОНАЛЬНОСТЬ: {positive_count}/{total_sentiment} позитивных, {negative_count}/{total_sentiment} негативных\n\n"
            
            # Генерируем умное резюме
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты - профессиональный аналитик аудиоконтента с опытом работы с деловыми встречами, "
                            "интервью, лекциями и подкастами. На основе предоставленного богатого контекста создай "
                            "подробное структурированное резюме на русском языке.\n\n"
                            
                            "СТРУКТУРА РЕЗЮМЕ:\n"
                            "📋 КРАТКОЕ РЕЗЮМЕ (2-3 предложения - суть записи)\n"
                            "🎯 ОСНОВНЫЕ ТЕМЫ И РАЗДЕЛЫ (с временными метками если есть)\n"
                            "👥 УЧАСТНИКИ И РОЛИ (если определены спикеры)\n"
                            "💡 КЛЮЧЕВЫЕ ИНСАЙТЫ И ВЫВОДЫ\n"
                            "📊 ВАЖНЫЕ ФАКТЫ, ЦИФРЫ, ДАТЫ\n"
                            "🎭 ЭМОЦИОНАЛЬНАЯ ТОНАЛЬНОСТЬ\n"
                            "✅ РЕШЕНИЯ, ДЕЙСТВИЯ, NEXT STEPS\n"
                            "🏷️ КЛЮЧЕВЫЕ ПЕРСОНЫ И ОРГАНИЗАЦИИ\n\n"
                            
                            "ТРЕБОВАНИЯ:\n"
                            "- Используй эмодзи для структуры\n"
                            "- Будь конкретным и информативным\n"
                            "- Выдели самое важное\n"
                            "- Сохраняй профессиональный тон\n"
                            "- Не повторяй информацию\n"
                            "- Используй данные от AssemblyAI как основу"
                        )
                    },
                    {"role": "user", "content": context}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Ошибка умного резюме: {e}")
            return self._create_basic_summary(transcript_result)
    
    def _create_basic_summary(self, transcript_result):
        """Создает базовое резюме из данных AssemblyAI"""
        summary_parts = []
        
        # Встроенное резюме
        builtin_summary = getattr(transcript_result, 'summary', None)
        if builtin_summary:
            summary_parts.append(f"📋 РЕЗЮМЕ:\n{builtin_summary}")
        
        # Главы
        chapters = getattr(transcript_result, 'chapters', None)
        if chapters:
            summary_parts.append("\n🎯 ОСНОВНЫЕ РАЗДЕЛЫ:")
            for i, chapter in enumerate(chapters[:8], 1):
                headline = getattr(chapter, 'headline', f'Раздел {i}')
                start_time = getattr(chapter, 'start', 0) / 1000 / 60
                summary_parts.append(f"{i}. {headline} ({start_time:.1f}мин)")
        
        # Ключевые моменты
        highlights = getattr(transcript_result, 'auto_highlights', None)
        if highlights and hasattr(highlights, 'results'):
            summary_parts.append("\n💡 КЛЮЧЕВЫЕ МОМЕНТЫ:")
            for highlight in highlights.results[:10]:
                text = getattr(highlight, 'text', '').strip()
                if text:
                    summary_parts.append(f"• {text}")
        
        return "\n".join(summary_parts) if summary_parts else "📋 Базовое резюме недоступно"

# === Инициализация ===
summarizer = AdvancedSummarizer(OPENROUTER_API_KEY)

# === Вспомогательные функции ===
def analyze_sentiment_overall(sentiment_results):
    """Анализирует общую тональность"""
    if not sentiment_results:
        return "neutral", 0, 0, 0
    
    positive_count = sum(1 for s in sentiment_results if getattr(s, 'sentiment', '') == 'POSITIVE')
    negative_count = sum(1 for s in sentiment_results if getattr(s, 'sentiment', '') == 'NEGATIVE')
    neutral_count = len(sentiment_results) - positive_count - negative_count
    
    if positive_count > negative_count and positive_count > neutral_count:
        return "positive", positive_count, negative_count, neutral_count
    elif negative_count > positive_count and negative_count > neutral_count:
        return "negative", positive_count, negative_count, neutral_count
    else:
        return "neutral", positive_count, negative_count, neutral_count

def format_entities_by_type(entities):
    """Группирует сущности по типам"""
    if not entities:
        return {}
    
    entity_groups = {}
    for entity in entities:
        entity_type = getattr(entity, 'entity_type', 'other')
        entity_text = getattr(entity, 'text', '')
        
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        
        if entity_text and entity_text not in entity_groups[entity_type]:
            entity_groups[entity_type].append(entity_text)
    
    return entity_groups

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "AssemblyAI Official SDK (No ffmpeg dependency)",
        "api_configured": bool(ASSEMBLYAI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "features": [
            "🎙️ Лучшая модель транскрипции",
            "👥 Определение спикеров", 
            "📚 Автоматические главы",
            "💡 Ключевые моменты",
            "🎭 Анализ тональности",
            "🏷️ Определение сущностей",
            "🤖 Встроенное резюме",
            "🧠 Умное резюме Claude",
            "🛡️ Модерация контента",
            "📊 Категоризация тем"
        ],
        "limits": {
            "free_credits": "$50 (185 hours)",
            "max_file_size": "500MB",
            "max_duration": "unlimited",
            "parallel_processing": "5 files"
        }
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()
    input_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Файл не загружен"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        timestamp = int(time.time() * 1000)
        input_path = os.path.join(TEMP_DIR, f"aai_{timestamp}.{file.filename.split('.')[-1]}")
        
        # Сохранение файла
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"📥 Файл сохранён: {file_size / 1024 / 1024:.1f} MB")
        
        # Примерная оценка длительности без pydub
        # Для MP3: ~1MB = ~1 минута при среднем качестве
        estimated_duration = file_size / 1024 / 1024  # грубая оценка в минутах
        logger.info(f"📊 Примерная длительность: ~{estimated_duration:.1f} минут")
        
        # Создаем транскрайбер с нашей конфигурацией
        config = get_transcription_config()
        transcriber = aai.Transcriber(config=config)
        
        # Запускаем транскрипцию
        logger.info("🚀 Запускаю транскрипцию AssemblyAI...")
        transcript = transcriber.transcribe(input_path)
        
        # Проверяем результат
        if transcript.status == aai.TranscriptStatus.error:
            error_msg = getattr(transcript, 'error', 'Неизвестная ошибка')
            raise RuntimeError(f"Ошибка транскрипции: {error_msg}")
        
        if not transcript.text:
            return jsonify({"error": "Не удалось получить транскрипцию"}), 400
        
        logger.info("✅ Транскрипция завершена успешно!")
        
        # Получаем точную длительность из результата AssemblyAI
        audio_duration_ms = getattr(transcript, 'audio_duration', None)
        actual_duration = audio_duration_ms / 1000 / 60 if audio_duration_ms else estimated_duration
        
        # Создаем умное резюме
        logger.info("🧠 Генерирую умное резюме...")
        summary = summarizer.create_smart_summary(transcript)
        
        # Анализируем результаты
        sentiment_analysis = getattr(transcript, 'sentiment_analysis_results', None) or []
        overall_sentiment, pos_count, neg_count, neu_count = analyze_sentiment_overall(sentiment_analysis)
        
        entities = getattr(transcript, 'entities', None) or []
        entities_by_type = format_entities_by_type(entities)
        
        chapters = getattr(transcript, 'chapters', None) or []
        highlights = getattr(transcript, 'auto_highlights', None)
        
        # Подсчет использованных кредитов
        credits_used = actual_duration / 60 * 0.37  # примерно $0.37 за час
        
        total_time = time.time() - start_time
        logger.info(f"✅ Полная обработка завершена за {total_time:.1f}с")

        # Формируем детальный ответ
        response_data = {
            "transcript": transcript.text,
            "summary": summary,
            "service_used": "AssemblyAI Official SDK",
            
            # Статистика
            "statistics": {
                "processing_time": f"{total_time:.1f}s",
                "audio_duration": f"{actual_duration:.1f}min", 
                "file_size": f"{file_size / 1024 / 1024:.1f}MB",
                "confidence": getattr(transcript, 'confidence', 0),
                "credits_used": f"${credits_used:.3f}",
                "words_count": len(transcript.text.split()) if transcript.text else 0
            },
            
            # AI анализ
            "ai_analysis": {
                "speakers_detected": len(set([utterance.speaker for utterance in getattr(transcript, 'utterances', []) if utterance.speaker])),
                "chapters_found": len(chapters),
                "highlights_found": len(highlights.results) if highlights and hasattr(highlights, 'results') else 0,
                "entities_found": len(entities),
                "sentiment_breakdown": {
                    "overall": overall_sentiment,
                    "positive_segments": pos_count,
                    "negative_segments": neg_count,
                    "neutral_segments": neu_count
                }
            }
        }
        
        # Добавляем детальные результаты если есть
        if chapters:
            response_data["chapters"] = [
                {
                    "headline": getattr(ch, 'headline', ''),
                    "start_time": f"{getattr(ch, 'start', 0)/1000/60:.1f}min",
                    "end_time": f"{getattr(ch, 'end', 0)/1000/60:.1f}min",
                    "summary": getattr(ch, 'summary', '')
                }
                for ch in chapters[:15]
            ]
        
        if highlights and hasattr(highlights, 'results'):
            response_data["key_highlights"] = [
                {
                    "text": getattr(h, 'text', ''),
                    "rank": getattr(h, 'rank', 0),
                    "start_time": f"{getattr(h, 'start', 0)/1000/60:.1f}min"
                }
                for h in highlights.results[:20]
            ]
        
        if entities_by_type:
            # Ограничиваем количество сущностей каждого типа
            limited_entities = {}
            for entity_type, entity_list in entities_by_type.items():
                limited_entities[entity_type] = entity_list[:10]
            response_data["entities_by_type"] = limited_entities
        
        # Встроенное резюме от AssemblyAI
        builtin_summary = getattr(transcript, 'summary', None)
        if builtin_summary:
            response_data["assemblyai_summary"] = builtin_summary

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка: {str(e)[:300]}"}), 500
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✅ AssemblyAI без ffmpeg сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
