FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    faster-whisper \
    torch \
    torchaudio

# Create working directory
WORKDIR /app

# Copy transcription script
COPY transcribe.py .

# Create volume mount point for model cache
VOLUME ["/root/.cache/whisper"]

# Set default command - expect audio file from stdin
ENTRYPOINT ["python", "transcribe.py", "/dev/stdin"]