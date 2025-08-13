<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiceSum - AssemblyAI Fixed</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .upload-section {
            padding: 40px;
            text-align: center;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            margin-bottom: 30px;
        }
        
        .file-input {
            display: none;
        }
        
        .file-input-label {
            display: inline-block;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
        }
        
        .file-input-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .upload-btn {
            display: none;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(17, 153, 142, 0.3);
        }
        
        .upload-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .progress {
            margin: 30px 0;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            width: 0%;
            transition: width 0.3s ease;
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .results {
            padding: 40px;
            background: #f8f9fa;
            display: none;
        }
        
        .result-section {
            margin-bottom: 30px;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        
        .result-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .transcript {
            line-height: 1.6;
            color: #444;
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4facfe;
        }
        
        .error {
            background: #ffe6e6;
            color: #d8000c;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #d8000c;
            margin: 20px 0;
            display: none;
        }
        
        .file-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #4facfe;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }

        .processing-status {
            text-align: center;
            padding: 20px;
            display: none;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ VoiceSum</h1>
            <p>Умная транскрипция с AI анализом - AssemblyAI (Fixed)</p>
        </div>
        
        <div class="upload-section">
            <div class="file-input-wrapper">
                <input type="file" id="audioFile" class="file-input" accept="audio/*,video/*">
                <label for="audioFile" class="file-input-label">
                    📁 Выберите аудио файл
                </label>
            </div>
            
            <div class="file-info" id="fileInfo"></div>
            
            <button class="upload-btn" id="uploadBtn" onclick="uploadFile()">
                🚀 Начать обработку
            </button>
            
            <div class="progress" id="progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="processing-status" id="processingStatus">
                    <div class="spinner"></div>
                    <p id="statusText">Обрабатываем ваш файл...</p>
                </div>
            </div>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <div class="result-section">
                <h3>📊 Статистика обработки</h3>
                <div class="stats" id="stats"></div>
            </div>
            
            <div class="result-section">
                <h3>🧠 Умное резюме</h3>
                <div id="summary"></div>
            </div>
            
            <div class="result-section">
                <h3>📝 Полная транскрипция</h3>
                <div class="transcript" id="transcript"></div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('audioFile');
        const uploadBtn = document.getElementById('uploadBtn');
        const fileInfo = document.getElementById('fileInfo');
        const progress = document.getElementById('progress');
        const progressFill = document.getElementById('progressFill');
        const processingStatus = document.getElementById('processingStatus');
        const statusText = document.getElementById('statusText');
        const results = document.getElementById('results');
        const error = document.getElementById('error');

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                const estimatedMinutes = (sizeMB * 1).toFixed(1);
                
                fileInfo.innerHTML = `
                    <strong>📄 Файл:</strong> ${file.name}<br>
                    <strong>📏 Размер:</strong> ${sizeMB} MB<br>
                    <strong>⏱️ Примерное время:</strong> ~${estimatedMinutes} минут<br>
                    <strong>🔧 Метод:</strong> Гибридный подход AssemblyAI
                `;
                fileInfo.style.display = 'block';
                uploadBtn.style.display = 'inline-block';
                hideError();
                hideResults();
            }
        });

        async function uploadFile() {
            const file = fileInput.files[0];
            if (!file) {
                showError('Пожалуйста, выберите файл');
                return;
            }

            // Скрываем предыдущие результаты
            hideError();
            hideResults();
            
            // Показываем прогресс
            progress.style.display = 'block';
            processingStatus.style.display = 'block';
            uploadBtn.disabled = true;
            uploadBtn.textContent = '⏳ Обрабатываем...';

            // Анимация прогресса
            let progressValue = 0;
            const progressInterval = setInterval(() => {
                progressValue += Math.random() * 15;
                if (progressValue > 90) progressValue = 90;
                progressFill.style.width = progressValue + '%';
            }, 1000);

            // Обновляем статус
            const statusMessages = [
                'Загружаем файл на сервер...',
                'Пробуем автообнаружение языка...',
                'Выполняем транскрипцию...',
                'Анализируем содержимое...',
                'Генерируем умное резюме...',
                'Финализируем результаты...'
            ];
            
            let statusIndex = 0;
            const statusInterval = setInterval(() => {
                if (statusIndex < statusMessages.length) {
                    statusText.textContent = statusMessages[statusIndex];
                    statusIndex++;
                }
            }, 8000);

            try {
                const formData = new FormData();
                formData.append('audio', file);

                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData,
                    // Увеличиваем таймаут для больших файлов
                    signal: AbortSignal.timeout(300000) // 5 минут
                });

                clearInterval(progressInterval);
                clearInterval(statusInterval);
                progressFill.style.width = '100%';

                if (!response.ok) {
                    let errorMessage = `Ошибка ${response.status}: ${response.statusText}`;
                    try {
                        const errorData = await response.json();
                        if (errorData.error) {
                            errorMessage = errorData.error;
                        }
                    } catch (e) {
                        // Если не можем парсить JSON ошибки, используем статус
                    }
                    throw new Error(errorMessage);
                }

                let data;
                try {
                    const responseText = await response.text();
                    console.log('Raw response:', responseText.substring(0, 1000) + '...');
                    data = JSON.parse(responseText);
                } catch (parseError) {
                    console.error('JSON Parse Error:', parseError);
                    throw new Error('Сервер вернул некорректные данные. Попробуйте еще раз или используйте файл меньшего размера.');
                }

                displayResults(data);
                
            } catch (error) {
                console.error('Upload error:', error);
                clearInterval(progressInterval);
                clearInterval(statusInterval);
                
                let userMessage = 'Произошла ошибка при обработке файла.';
                
                if (error.name === 'TimeoutError') {
                    userMessage = 'Время ожидания истекло. Попробуйте файл меньшего размера.';
                } else if (error.message) {
                    userMessage = error.message;
                }
                
                showError(userMessage);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = '🚀 Начать обработку';
                progress.style.display = 'none';
                processingStatus.style.display = 'none';
            }
        }

        function displayResults(data) {
            // Статистика
            const stats = document.getElementById('stats');
            const statistics = data.statistics || {};
            
            stats.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${statistics.processing_time || 'N/A'}</div>
                    <div class="stat-label">⏱️ Время обработки</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${statistics
