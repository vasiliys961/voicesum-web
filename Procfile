web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 1 --worker-class sync --max-requests 10 --max-requests-jitter 5 --preload --worker-tmp-dir /dev/shm


