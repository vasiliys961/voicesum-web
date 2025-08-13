# app.py - Исправленная версия с гибридным подходом AssemblyAI
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import assemblyai as aai
from openai import OpenAI
import time
import logging
import signal
import sys

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
        
        # === AI функции (ИСПРАВЛЕНО: убран конфликт) ===
        speaker_labels=True,  # Определение спикеров
        speakers_expected=2,  # Ожидаемое количество
        
        auto_highlights=True,  # Ключевые моменты
        # ИСПРАВЛЕНИЕ: Выбираем ЛИБО auto_chapters, ЛИБО summarization
        summarization=True,       # Встроенное резюме
        summary_model=aai.SummarizationModel.informative,
        summary_type=aai.SummarizationType.bullets,
        # auto_chapters=True,   # ОТКЛЮЧЕНО из-за конфликта с summarization
        
        sentiment_analysis=True,  # Анализ эмоций
        entity_detection=True,    # Определение сущностей
        
        content_safety=True,      # Модерация контента
        iab_categories=True,      # Категоризация по темам
        
        # === Настройки качества ===
        punctuate=True,          # Пунктуация
        format_text=True,        # Форматирование текста
        dual_channel=False,      # Моно обработка
        
        # === Дополнительные функции ===
        disfluencies=False,      # Убираем "эм", "ах"
        filter_profanity=False,  # Не фильтруем мат
        redact_pii=False,        # Не скрываем персональные данные
    )

def get_transcription_config_auto_chapters():
    """Альтернативная конфигурация с главами вместо резюме"""
    return aai.TranscriptionConfig(
        # === Основные настройки ===
        speech_model=aai.SpeechModel.best,
        
        # === AI функции с главами ===
        speaker_labels=True,
        speakers_expected=2,
        
        auto_highlights=True,
        auto_chapters=True,       # Главы вместо резюме
        # summarization НЕ включаем
        
        sentiment_analysis=True,
        entity_detection=True,
        content_safety=True,
        iab_categories=True,
        
        # === Настройки качества ===
        punctuate=True,
        format_text=True,
        dual_channel=False,
        
        # === Дополнительные функции ===
        disfluencies=False,
        filter_profanity=False,
        redact_pii=False,
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
    """Улучшенная транскрипция с множественным fallback"""
    
    # Стратегия 1: Автообнаружение с резюме
    try:
        logger.info("🌍 Пробую автообнаружение языка с резюме...")
        config_auto = get_transcription_config_auto()
        transcriber = aai.Transcriber(config=config_auto)
        transcript = transcriber.transcribe(file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Ошибка автообнаружения: {transcript.error}")
        
        logger.info("✅ Транскрипция с автообнаружением (резюме) успешна!")
        return transcript, "auto_detection_summary"
        
    except Exception as e:
        logger.warning(f"⚠️ Автообнаружение с резюме не сработало: {e}")
        
        # Стратегия 2: Автообнаружение с главами
        try:
            logger.info("🌍 Пробую автообнаружение языка с главами...")
            config_chapters = get_transcription_config_auto_chapters()
            transcriber = aai.Transcriber(config=config_chapters)
            transcript = transcriber.transcribe(file_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"Ошибка автообнаружения с главами: {transcript.error}")
            
            logger.info("✅ Транскрипция с автообнаружением (главы) успешна!")
            return transcript, "auto_detection_chapters"
            
        except Exception as e2:
            logger.warning(f"⚠️ Автообнаружение с главами не сработало: {e2}")
            
            # Стратегия 3: Русский язык (fallback)
            try:
                logger.info("🇷🇺 Переключаюсь на русский язык (ограниченные функции)...")
                config_ru = get_transcription_config_russian()
                transcriber = aai.Transcriber(config=config_ru)
                transcript = transcriber.transcribe(file_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise RuntimeError(f"Ошибка русского: {transcript.error}")
                
                logger.info("✅ Транскрипция на русском успешна!")
                return transcript, "russian_limited_features"
                
            except Exception as e3:
                logger.error(f"❌ Все методы не сработали: авто-резюме({e}), авто-главы({e2}), русский({e3})")
                raise RuntimeError(f"Не удалось транскрибировать файл всеми доступными методами")

# === Улучшенный генератор резюме ===
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
            # Таймаут для генерации резюме
            timeout_seconds = 60
            
            # Собираем все данные
            transcript = transcript_result.text
            chapters = getattr(transcript_result, 'chapters', None) or []
            highlights = getattr(transcript_result, 'auto_highlights', None)
            sentiment = getattr(transcript_result, 'sentiment_analysis_results', None) or []
            entities = getattr(transcript_result, 'entities', None) or []
            builtin_summary = getattr(transcript_result, 'summary', None)
            detected_language = getattr(transcript_result, 'language_code', 'unknown')
            
            # Формируем богатый контекст (ограничиваем размер)
            context = f"ПОЛНЫЙ ТРАНСКРИПТ:\n{transcript[:20000]}\n\n"
            context += f"МЕТОД ТРАНСКРИПЦИИ: {transcription_method}\n"
            context += f"ОПРЕДЕЛЕННЫЙ ЯЗЫК: {detected_language}\n\n"
            
            if chapters:
                context += "📚 АВТОМАТИЧЕСКИЕ ГЛАВЫ:\n"
                for i, chapter in enumerate(chapters[:8], 1):
                    headline = getattr(chapter, 'headline', f'Глава {i}')
                    start_time = getattr(chapter, 'start', 0) / 1000 / 60  # в минутах
                    context += f"{i}. {headline} ({start_time:.1f}мин)\n"
                context += "\n"
            
            if highlights and hasattr(highlights, 'results'):
                context += "💡 КЛЮЧЕВЫЕ МОМЕНТЫ:\n"
                for highlight in highlights.results[:10]:
                    text = getattr(highlight, 'text', '')
                    if text:
                        context += f"• {text}\n"
                context += "\n"
            
            if entities:
                context += "🏷️ УПОМЯНУТЫЕ СУЩНОСТИ:\n"
                entity_groups = {}
                for entity in entities[:20]:
                    entity_type = getattr(entity, 'entity_type', 'other')
                    entity_text = getattr(entity, 'text', '')
                    if entity_type not in entity_groups:
                        entity_groups[entity_type] = []
                    if entity_text not in entity_groups[entity_type]:
                        entity_groups[entity_type].append(entity_text)
                
                for entity_type, texts in entity_groups.items():
                    context += f"  {entity_type}: {', '.join(texts[:3])}\n"
                context += "\n"
            
            if builtin_summary:
                context += f"🤖 БАЗОВОЕ РЕЗЮМЕ ASSEMBLYAI:\n{builtin_summary}\n\n"
            
            # Определяем язык контента для адаптации промпта
            content_language = detected_language if detected_language != 'unknown' else 'неизвестный'
            
            system_prompt = f"""Ты - профессиональный аналитик аудиоконтента. 
Транскрипт выполнен на языке: {content_language}. Метод: {transcription_method}.

ВАЖНО: Создай подробное структурированное резюме СТРОГО НА РУССКОМ ЯЗЫКЕ.

СТРУКТУРА РЕЗЮМЕ:
📋 КРАТКОЕ РЕЗЮМЕ (2-3 предложения)
🌍 ЯЗЫК КОНТЕНТА: {content_language}
🎯 ОСНОВНЫЕ ТЕМЫ (с временными метками если есть)
👥 УЧАСТНИКИ (если определены спикеры)
💡 КЛЮЧЕВЫЕ ИНСАЙТЫ
📊 ВАЖНЫЕ ФАКТЫ И ЦИФРЫ
🎭 ЭМОЦИОНАЛЬНАЯ ТОНАЛЬНОСТЬ
✅ РЕШЕНИЯ И ДЕЙСТВИЯ
🏷️ КЛЮЧЕВЫЕ ПЕРСОНЫ

Используй эмодзи, будь конкретным, переводи на русский язык."""
            
            # Генерируем умное резюме с таймаутом
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=1200,
                temperature=0.3,
                timeout=timeout_seconds
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Ошибка умного резюме: {e}")
            return self._create_basic_summary(transcript_result, transcription_method)
    
    def _create_basic_summary(self, transcript_result, transcription_method):
        """Создает базовое резюме из данных AssemblyAI НА РУССКОМ ЯЗЫКЕ"""
        summary_parts = []
        detected_language = getattr(transcript_result, 'language_code', 'unknown')
        
        language_names = {
            'en': 'английский', 'ru': 'русский', 'es': 'испанский',
            'fr': 'французский', 'de': 'немецкий', 'it': 'итальянский',
            'pt': 'португальский', 'zh': 'китайский', 'ja': 'японский',
            'ko': 'корейский', 'unknown': 'не определен'
        }
        
        language_display = language_names.get(detected_language, detected_language)
        
        summary_parts.append(f"🔧 МЕТОД ОБРАБОТКИ: {transcription_method}")
        summary_parts.append(f"🌍 ЯЗЫК КОНТЕНТА: {language_display}")
        
        # Встроенное резюме
        builtin_summary = getattr(transcript_result, 'summary', None)
        if builtin_summary:
            summary_parts.append(f"\n📋 БАЗОВОЕ РЕЗЮМЕ:\n{builtin_summary}")
        
        # Главы
        chapters = getattr(transcript_result, 'chapters', None)
        if chapters:
            summary_parts.append("\n🎯 ОСНОВНЫЕ РАЗДЕЛЫ:")
            for i, chapter in enumerate(chapters[:6], 1):
                headline = getattr(chapter, 'headline', f'Раздел {i}')
                start_time = getattr(chapter, 'start', 0) / 1000 / 60
                summary_parts.append(f"{i}. {headline} ({start_time:.1f}мин)")
        
        # Ключевые моменты
        highlights = getattr(transcript_result, 'auto_highlights', None)
        if highlights and hasattr(highlights, 'results'):
            summary_parts.append(f"\n💡 КЛЮЧЕВЫЕ МОМЕНТЫ:")
            for highlight in highlights.results[:8]:
                text = getattr(highlight, 'text', '').strip()
                if text:
                    summary_parts.append(f"• {text}")
        
        # Сущности
        entities = getattr(transcript_result, 'entities', None)
        if entities:
            summary_parts.append("\n🏷️ УПОМЯНУТЫЕ СУЩНОСТИ:")
            entity_groups = {}
            
            entity_type_translation = {
                'person': 'Люди', 'organization': 'Организации', 
                'location': 'Места', 'date': 'Даты', 'money': 'Деньги',
                'phone_number': 'Телефоны', 'email': 'Email',
                'product': 'Продукты', 'event': 'События', 'other': 'Прочее'
            }
            
            for entity in entities[:12]:
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
        
        if not summary_parts:
            summary_parts.append("📋 Базовое резюме недоступно")
        
        return "\n".join(summary_parts)

# === Инициализация ===
summarizer = AdvancedSummarizer(OPENROUTER_API_KEY)

# === Улучшенные вспомогательные функции ===
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
    if transcription_method == "auto_detection_summary":
        return [
            "🎙️ Лучшая модель транскрипции",
            "🌍 Автообнаружение языка",
            "👥 Определение спикеров", 
            "💡 Ключевые моменты",
            "🎭 Анализ тональности",
            "🏷️ Определение сущностей",
            "🤖 Встроенное резюме",
            "🧠 Умное резюме Claude",
            "🛡️ Модерация контента",
            "📊 Категоризация тем"
        ]
    elif transcription_method == "auto_detection_chapters":
        return [
            "🎙️ Лучшая модель транскрипции",
            "🌍 Автообнаружение языка",
            "👥 Определение спикеров", 
            "📚 Автоматические главы",
            "💡 Ключевые моменты",
            "🎭 Анализ тональности",
            "🏷️ Определение сущностей",
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

# === Обработчик сигналов ===
def signal_handler(signum, frame):
    logger.info("🛑 Получен сигнал завершения, очищаем ресурсы...")
    # Очистка временных файлов
    import shutil
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# === Маршруты ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "AssemblyAI Hybrid Approach (Fixed)",
        "api_configured": bool(ASSEMBLYAI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "transcription_strategies": [
            "1. Auto-detection with summary",
            "2. Auto-detection with chapters", 
            "3. Russian language (fallback)"
        ],
        "fixes_applied": [
            "✅ Убран конфликт auto_chapters + summarization",
            "✅ Добавлена множественная стратегия fallback",
            "✅ Улучшена обработка ошибок",
            "✅ Добавлены таймауты",
            "✅ Оптимизирован размер контекста"
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
        input_path = os.path.join(TEMP_DIR, f"hybrid_{timestamp}.{file.filename.split('.')[-1]}")
        
        # Сохранение файла
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"📥 Файл сохранён: {file_size / 1024 / 1024:.1f} MB")
        
        # Примерная оценка длительности
        estimated_duration = file_size / 1024 / 1024  # грубая оценка в минутах
        logger.info(f"📊 Примерная длительность: ~{estimated_duration:.1f} минут")
        
        # Улучшенная гибридная транскрипция с множественным fallback
        transcript, transcription_method = transcribe_with_fallback(input_path)
        
        if not transcript.text:
            return jsonify({"error": "Не удалось получить транскрипцию"}), 400
        
        logger.info(f"✅ Транскрипция завершена методом: {transcription_method}")
        
        # Получаем точную длительность из результата AssemblyAI
        audio_duration_ms = getattr(transcript, 'audio_duration', None)
        actual_duration = audio_duration_ms / 1000 / 60 if audio_duration_ms else estimated_duration
        detected_language = getattr(transcript, 'language_code', 'unknown')
        
        # Создаем умное резюме с обработкой ошибок
        try:
            logger.info("🧠 Генерирую умное резюме...")
            summary = summarizer.create_smart_summary(transcript, transcription_method)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка генерации умного резюме: {e}")
            summary = summarizer._create_basic_summary(transcript, transcription_method)
        
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
            "service_used": "AssemblyAI Hybrid Approach (Fixed)",
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
                for ch in chapters[:12]
            ]
        
        if highlights and hasattr(highlights, 'results'):
            response_data["key_highlights"] = [
                {
                    "text": getattr(h, 'text', ''),
                    "rank": getattr(h, 'rank', 0),
                    "start_time": f"{getattr(h, 'start', 0)/1000/60:.1f}min"
                }
                for h in highlights.results[:15]
            ]
        
        if entities_by_type:
            # Ограничиваем количество сущностей каждого типа
            limited_entities = {}
            for entity_type, entity_list in entities_by_type.items():
                limited_entities[entity_type] = entity_list[:8]
            response_data["entities_by_type"] = limited_entities
        
        # Встроенное резюме от AssemblyAI (если доступно)
        builtin_summary = getattr(transcript, 'summary', None)
        if builtin_summary:
            response_data["assemblyai_summary"] = builtin_summary
        
        # Информация о методе транскрипции
        method_names = {
            "auto_detection_summary": "Автообнаружение языка с резюме",
            "auto_detection_chapters": "Автообнаружение языка с главами", 
            "russian_limited_features": "Русский язык (fallback)"
        }
        
        response_data["method_info"] = {
            "name": method_names.get(transcription_method, transcription_method),
            "features": "Все AI функции" if "auto_detection" in transcription_method else "Ограниченный набор",
            "language_detected": detected_language,
            "fallback_used": transcription_method == "russian_limited_features"
        }

        return jsonify(response_data)
