# app.py - Полная версия с гибридным подходом AssemblyAI
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
TEMP_DIR = tempfile.mkdtemp(prefix="voicesum_hybrid_")

logger.info(f"📁 Используется временная папка: {TEMP_DIR}")

# === Настройка AssemblyAI ===
aai.settings.api_key = ASSEMBLYAI_API_KEY
logger.info("🔑 AssemblyAI API ключ настроен")

# === Конфигурации транскрипции ===
def get_transcription_config_auto():
    """Конфигурация с автообнаружением языка для максимальных возможностей"""
    return aai.TranscriptionConfig(
        # === Основные настройки ===
        speech_model=aai.SpeechModel.best,  # Лучшая модель
        # language_code НЕ указываем - автообнаружение
        
        # === AI функции (все включены для автообнаружения) ===
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

def get_transcription_config_russian():
    """Конфигурация специально для русского языка"""
    return aai.TranscriptionConfig(
        # === Основные настройки ===
        speech_model=aai.SpeechModel.best,  # Лучшая модель
        language_code="ru",  # Русский язык
        
        # === Доступные для русского языка функции ===
        speaker_labels=True,  # Определение спикеров ✅
        speakers_expected=2,  # Ожидаемое количество
        
        entity_detection=True,    # Определение сущностей ✅
        
        # === Настройки качества ===
        punctuate=True,          # Пунктуация
        format_text=True,        # Форматирование текста
        dual_channel=False,      # Моно обработка
        
        # === Дополнительные функции ===
        disfluencies=False,      # Убираем "эм", "ах"
        filter_profanity=False,  # Не фильтруем мат
        redact_pii=False,        # Не скрываем персональные данные
    )

def transcribe_with_fallback(file_path):
    """Транскрипция с fallback: сначала авто, потом русский"""
    
    # Пробуем с автообнаружением (все функции)
    try:
        logger.info("🌍 Пробую транскрипцию с автообнаружением языка (все AI функции)...")
        config_auto = get_transcription_config_auto()
        transcriber = aai.Transcriber(config=config_auto)
        transcript = transcriber.transcribe(file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Ошибка автообнаружения: {transcript.error}")
        
        logger.info("✅ Транскрипция с автообнаружением успешна!")
        return transcript, "auto_detection_full_features"
        
    except Exception as e:
        logger.warning(f"⚠️ Автообнаружение не сработало: {e}")
        
        # Fallback: русский язык (ограниченные функции)
        try:
            logger.info("🇷🇺 Переключаюсь на русский язык (ограниченные функции)...")
            config_ru = get_transcription_config_russian()
            transcriber = aai.Transcriber(config=config_ru)
            transcript = transcriber.transcribe(file_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"Ошибка русского: {transcript.error}")
            
            logger.info("✅ Транскрипция на русском успешна!")
            return transcript, "russian_limited_features"
            
        except Exception as e2:
            logger.error(f"❌ Оба метода не сработали: {e2}")
            raise RuntimeError(f"Не удалось транскрибировать: авто({e}), русский({e2})")

# === Умный генератор резюме ===
class AdvancedSummarizer:
    def __init__(self, openrouter_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        ) if openrouter_key else None
    
    def create_smart_summary(self, transcript_result, transcription_method):
        """Создает умное резюме на основе всех данных AssemblyAI"""
        if not self.client:
            return self._create_basic_summary(transcript_result, transcription_method)
        
        try:
            # Собираем все данные
            transcript = transcript_result.text
            chapters = getattr(transcript_result, 'chapters', None) or []
            highlights = getattr(transcript_result, 'auto_highlights', None)
            sentiment = getattr(transcript_result, 'sentiment_analysis_results', None) or []
            entities = getattr(transcript_result, 'entities', None) or []
            builtin_summary = getattr(transcript_result, 'summary', None)
            detected_language = getattr(transcript_result, 'language_code', 'unknown')
            
            # Формируем богатый контекст
            context = f"ПОЛНЫЙ ТРАНСКРИПТ:\n{transcript[:25000]}\n\n"
            context += f"МЕТОД ТРАНСКРИПЦИИ: {transcription_method}\n"
            context += f"ОПРЕДЕЛЕННЫЙ ЯЗЫК: {detected_language}\n\n"
            
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
            
            # Определяем язык контента для адаптации промпта
            content_language = detected_language if detected_language != 'unknown' else 'неизвестный'
            
            # Адаптируем промпт в зависимости от метода транскрипции
            if transcription_method == "auto_detection_full_features":
                system_prompt = (
                    f"Ты - профессиональный аналитик аудиоконтента с доступом к полному набору AI-анализа. "
                    f"Транскрипт выполнен на языке: {content_language}. "
                    f"Используй все предоставленные данные: главы, ключевые моменты, тональность, сущности. "
                    f"ВАЖНО: Создай подробное структурированное резюме СТРОГО НА РУССКОМ ЯЗЫКЕ, "
                    f"даже если исходный контент на другом языке. Переведи все ключевые моменты и выводы."
                )
            else:
                system_prompt = (
                    f"Ты - профессиональный аналитик аудиоконтента. Транскрипт выполнен на русском языке. "
                    f"У тебя есть транскрипт и базовая информация о сущностях и спикерах. "
                    f"Создай подробное структурированное резюме НА РУССКОМ ЯЗЫКЕ, "
                    f"максимально используя доступную информацию."
                )
            
            system_prompt += f"""

ВАЖНЫЕ УКАЗАНИЯ:
- РЕЗЮМЕ ДОЛЖНО БЫТЬ СТРОГО НА РУССКОМ ЯЗЫКЕ
- Если исходный контент на {content_language}, переведи ключевые моменты
- Имена собственные можно оставлять в оригинале, но с пояснениями на русском
- Технические термины переводи или поясняй на русском

СТРУКТУРА РЕЗЮМЕ:
📋 КРАТКОЕ РЕЗЮМЕ (2-3 предложения - суть записи)
🌍 ЯЗЫК КОНТЕНТА: {content_language}
🎯 ОСНОВНЫЕ ТЕМЫ И РАЗДЕЛЫ (с временными метками если есть)
👥 УЧАСТНИКИ И РОЛИ (если определены спикеры)
💡 КЛЮЧЕВЫЕ ИНСАЙТЫ И ВЫВОДЫ
📊 ВАЖНЫЕ ФАКТЫ, ЦИФРЫ, ДАТЫ
🎭 ЭМОЦИОНАЛЬНАЯ ТОНАЛЬНОСТЬ (если доступна)
✅ РЕШЕНИЯ, ДЕЙСТВИЯ, NEXT STEPS
🏷️ КЛЮЧЕВЫЕ ПЕРСОНЫ И ОРГАНИЗАЦИИ

ТРЕБОВАНИЯ:
- ВСЕ РЕЗЮМЕ НА РУССКОМ ЯЗЫКЕ
- Используй эмодзи для структуры
- Будь конкретным и информативным
- Выдели самое важное
- Сохраняй профессиональный тон
- Переводи иностранные термины и понятия
- Не повторяй информацию
- Используй все доступные данные от AssemblyAI
"""
            
            # Генерируем умное резюме
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Ошибка умного резюме: {e}")
            return self._create_basic_summary(transcript_result, transcription_method)
    
    def _create_basic_summary(self, transcript_result, transcription_method):
        """Создает базовое резюме из данных AssemblyAI НА РУССКОМ ЯЗЫКЕ"""
        summary_parts = []
        detected_language = getattr(transcript_result, 'language_code', 'unknown')
        
        # Определяем язык контента
        language_names = {
            'en': 'английский',
            'ru': 'русский', 
            'es': 'испанский',
            'fr': 'французский',
            'de': 'немецкий',
            'it': 'итальянский',
            'pt': 'португальский',
            'zh': 'китайский',
            'ja': 'японский',
            'ko': 'корейский',
            'unknown': 'не определен'
        }
        
        language_display = language_names.get(detected_language, detected_language)
        
        summary_parts.append(f"🔧 МЕТОД ОБРАБОТКИ: {transcription_method}")
        summary_parts.append(f"🌍 ЯЗЫК КОНТЕНТА: {language_display}")
        
        # Встроенное резюме (переводим если нужно)
        builtin_summary = getattr(transcript_result, 'summary', None)
        if builtin_summary:
            if detected_language != 'ru' and detected_language != 'unknown':
                summary_parts.append(f"\n📋 БАЗОВОЕ РЕЗЮМЕ (автоперевод с {language_display}):")
                summary_parts.append(f"[ОРИГИНАЛ] {builtin_summary}")
                summary_parts.append("📝 Для детального анализа на русском см. умное резюме выше")
            else:
                summary_parts.append(f"\n📋 БАЗОВОЕ РЕЗЮМЕ:\n{builtin_summary}")
        
        # Главы (переводим заголовки если нужно)
        chapters = getattr(transcript_result, 'chapters', None)
        if chapters:
            summary_parts.append("\n🎯 ОСНОВНЫЕ РАЗДЕЛЫ:")
            for i, chapter in enumerate(chapters[:8], 1):
                headline = getattr(chapter, 'headline', f'Раздел {i}')
                start_time = getattr(chapter, 'start', 0) / 1000 / 60
                
                if detected_language != 'ru' and detected_language != 'unknown':
                    summary_parts.append(f"{i}. [{headline}] ({start_time:.1f}мин)")
                else:
                    summary_parts.append(f"{i}. {headline} ({start_time:.1f}мин)")
        
        # Ключевые моменты (показываем с пометкой о языке)
        highlights = getattr(transcript_result, 'auto_highlights', None)
        if highlights and hasattr(highlights, 'results'):
            summary_parts.append(f"\n💡 КЛЮЧЕВЫЕ МОМЕНТЫ (на {language_display}):")
            for highlight in highlights.results[:10]:
                text = getattr(highlight, 'text', '').strip()
                if text:
                    summary_parts.append(f"• {text}")
        
        # Сущности (всегда доступны, группируем и переводим названия типов)
        entities = getattr(transcript_result, 'entities', None)
        if entities:
            summary_parts.append("\n🏷️ УПОМЯНУТЫЕ СУЩНОСТИ:")
            entity_groups = {}
            
            # Перевод типов сущностей
            entity_type_translation = {
                'person': 'Люди',
                'organization': 'Организации', 
                'location': 'Места',
                'date': 'Даты',
                'money': 'Деньги',
                'phone_number': 'Телефоны',
                'email': 'Email',
                'product': 'Продукты',
                'event': 'События',
                'other': 'Прочее'
            }
            
            for entity in entities[:15]:
                entity_type = getattr(entity, 'entity_type', 'other')
                entity_text = getattr(entity, 'text', '')
                translated_type = entity_type_translation.get(entity_type, entity_type)
                
                if translated_type not in entity_groups:
                    entity_groups[translated_type] = []
                if entity_text not in entity_groups[translated_type]:
                    entity_groups[translated_type].append(entity_text)
            
            for entity_type, texts in entity_groups.items():
                if texts:
                    summary_parts.append(f"  {entity_type}: {', '.join(texts[:3])}")
        
        # Добавляем примечание о языке
        if detected_language != 'ru' and detected_language != 'unknown':
            summary_parts.append(f"\n📝 ПРИМЕЧАНИЕ: Исходный контент на {language_display}. Для полного анализа на русском языке используется умное резюме выше.")
        
        if not summary_parts:
            summary_parts.append("📋 Базовое резюме недоступно")
        
        return "\n".join(summary_parts)

# === Инициализация ===
summarizer = AdvancedSummarizer(OPENROUTER_API_KEY)

# === Вспомогательные функции ===
def analyze_sentiment_overall(sentiment_results):
    """Анализирует общую тональность"""
    if not sentiment_results:
        return "not_analyzed", 0, 0, 0
    
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

def get_transcription_features(transcription_method):
    """Возвращает список доступных функций в зависимости от метода"""
    if transcription_method == "auto_detection_full_features":
        return [
            "🎙️ Лучшая модель транскрипции",
            "🌍 Автообнаружение языка",
            "👥 Определение спикеров", 
            "📚 Автоматические главы",
            "💡 Ключевые моменты",
            "🎭 Анализ тональности",
            "🏷️ Определение сущностей",
            "🤖 Встроенное резюме",
            "🧠 Умное резюме Claude",
            "🛡️ Модерация контента",
            "📊 Категоризация тем"
        ]
    else:
        return [
            "🎙️ Лучшая модель транскрипции",
            "🇷🇺 Русский язык",
            "👥 Определение спикеров",
            "🏷️ Определение сущностей",
            "🧠 Умное резюме Claude"
        ]

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "AssemblyAI Hybrid Approach",
        "api_configured": bool(ASSEMBLYAI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "transcription_methods": [
            "1. Auto-detection (full features)",
            "2. Russian language (fallback)"
        ],
        "auto_features": get_transcription_features("auto_detection_full_features"),
        "russian_features": get_transcription_features("russian_limited_features"),
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
        input_path = os.path.join(TEMP_DIR, f"hybrid_{timestamp}.{file.filename.split('.')[-1]}")
        
        # Сохранение файла
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"📥 Файл сохранён: {file_size / 1024 / 1024:.1f} MB")
        
        # Примерная оценка длительности
        estimated_duration = file_size / 1024 / 1024  # грубая оценка в минутах
        logger.info(f"📊 Примерная длительность: ~{estimated_duration:.1f} минут")
        
        # Гибридная транскрипция с fallback
        transcript, transcription_method = transcribe_with_fallback(input_path)
        
        if not transcript.text:
            return jsonify({"error": "Не удалось получить транскрипцию"}), 400
        
        logger.info(f"✅ Транскрипция завершена методом: {transcription_method}")
        
        # Получаем точную длительность из результата AssemblyAI
        audio_duration_ms = getattr(transcript, 'audio_duration', None)
        actual_duration = audio_duration_ms / 1000 / 60 if audio_duration_ms else estimated_duration
        detected_language = getattr(transcript, 'language_code', 'unknown')
        
        # Создаем умное резюме
        logger.info("🧠 Генерирую умное резюме...")
        summary = summarizer.create_smart_summary(transcript, transcription_method)
        
        # Анализируем результаты в зависимости от метода
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
            "service_used": "AssemblyAI Hybrid Approach",
            "transcription_method": transcription_method,
            "detected_language": detected_language,
            
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
                "method_used": transcription_method,
                "speakers_detected": len(set([utterance.speaker for utterance in getattr(transcript, 'utterances', []) if utterance.speaker])),
                "chapters_found": len(chapters),
                "highlights_found": len(highlights.results) if highlights and hasattr(highlights, 'results') else 0,
                "entities_found": len(entities),
                "sentiment_breakdown": {
                    "overall": overall_sentiment,
                    "positive_segments": pos_count,
                    "negative_segments": neg_count,
                    "neutral_segments": neu_count
                },
                "features_available": get_transcription_features(transcription_method)
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
        
        # Встроенное резюме от AssemblyAI (если доступно)
        builtin_summary = getattr(transcript, 'summary', None)
        if builtin_summary:
            response_data["assemblyai_summary"] = builtin_summary
        
        # Информация о методе транскрипции
        if transcription_method == "auto_detection_full_features":
            response_data["method_info"] = {
                "name": "Автообнаружение языка",
                "features": "Все AI функции доступны",
                "language_detected": detected_language,
                "fallback_used": False
            }
        else:
            response_data["method_info"] = {
                "name": "Русский язык (fallback)",
                "features": "Ограниченный набор функций",
                "language_forced": "ru",
                "fallback_used": True
            }

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка: {str(e)[:300]}"}), 500
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✅ AssemblyAI Hybrid сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
