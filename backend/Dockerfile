# Build context must be the repository root (backend/ needs sibling ml/).
FROM python:3.11-slim

# librosa/soundfile need libsndfile; parselmouth/ffmpeg for audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/

ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend
EXPOSE 5000

# Flask application factory served by gunicorn.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "-t", "120", "app:create_app()"]
