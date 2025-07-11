# Multi-stage build for better caching
FROM python:3.11-slim as base

# Install system dependencies in a separate layer for better caching
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create requirements file for better dependency caching
FROM base as deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM base as final
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Create working directory
WORKDIR /app

# Copy transcription script (this changes frequently, so put it last)
COPY transcribe.py .

# Create volume mount point for model cache
VOLUME ["/root/.cache/whisper"]

# Set default command - expect audio file from stdin
ENTRYPOINT ["python", "transcribe.py", "/dev/stdin"]