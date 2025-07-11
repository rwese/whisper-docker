FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    torch \
    torchaudio

# Create working directory
WORKDIR /app

# Copy transcription script
COPY transcribe.py .

# Create volume mount points for audio files and model cache
VOLUME ["/audio"]
VOLUME ["/root/.cache/whisper"]

# Set default command
ENTRYPOINT ["python", "transcribe.py"]