// static/script.js
let mediaRecorder;
let audioChunks = [];

document.getElementById('recordButton').addEventListener('click', async () => {
    const button = document.getElementById('recordButton');
    const status = document.getElementById('recordingStatus');

    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        // Начать запись
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();

                const fileInput = document.getElementById('audioFile');
                fileInput.files = createFileList([new File([audioBlob], "recorded.webm", { type: 'audio/webm' })]);
            };

            mediaRecorder.start();
            button.textContent = '⏹️ Стоп';
            status.classList.remove('hidden');
        } catch (err) {
            alert('Ошибка доступа к микрофону: ' + err.message);
        }
    } else if (mediaRecorder.state === 'recording') {
        // Остановить запись
        mediaRecorder.stop();
        button.textContent = '● Запись';
        status.classList.add('hidden');
    }
});

document.getElementById('transcribeButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    if (!file) {
        alert("Выберите или запишите аудиофайл");
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (result.error) {
            alert("Ошибка: " + result.error);
        } else {
            document.getElementById('transcript').value = result.transcript;
            document.getElementById('summary').value = result.summary;
        }
    } catch (err) {
        alert("Ошибка соединения: " + err.message);
    }
});

// Вспомогательная функция для установки файла в input
function createFileList(arr) {
    const dataTransfer = new DataTransfer();
    arr.forEach(file => dataTransfer.items.add(file));
    return dataTransfer.files;
}