# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Docker-based Whisper transcription service that converts audio files to text using OpenAI's Whisper model. It supports both CLI and web service modes controlled by the `MODE` environment variable.

## Core Architecture

### Dual-Mode Design
- **CLI Mode** (`MODE=cli`): Traditional command-line interface for file-based transcription
- **Web Mode** (`MODE=web`): FastAPI-based REST API with HTML frontend

### Key Components
- **transcription_core.py**: Centralized transcription logic using `faster-whisper`, shared between both modes. Handles automatic language detection from filename patterns (`_en.m4a`, `_de.m4a`) and supports multiple output formats (txt, json, srt, vtt, tsv)
- **transcribe.py**: CLI entry point with stdin support and argument parsing
- **web_service.py**: FastAPI web service with endpoints for transcription, health checks, and API documentation  
- **static/index.html**: Modern HTML5 frontend with drag-and-drop file upload
- **entrypoint.sh**: Dynamic mode switching between CLI and web service based on environment
- **generate_client.py**: Auto-generates Python client from OpenAPI schema

### Language Detection Strategy
1. Filename patterns (`_en`, `_de`, etc.)
2. Manual override via `--language` parameter  
3. Default fallback to German and English auto-detection

### Containerization
- **Multi-stage Dockerfile** with FFmpeg for audio processing
- **Docker Compose** with separate services for CLI and web modes
- **Model cache persistence** at `/root/.cache/whisper` to avoid re-downloading
- **Multi-architecture builds** (linux/amd64, linux/arm64) via GitHub Actions

## Usage

### CLI Mode (Original)

#### Wrapper Script (Recommended)
```bash
./transcribe audio.m4a                    # Basic usage
./transcribe audio.wav --model base       # Different model
./transcribe audio.mp3 > transcript.txt   # Save to file
```

#### Direct Docker
```bash
cat audio.m4a | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli
cat audio.wav | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli --model base
```

### Web Service Mode

#### Start Web Service
```bash
# Using docker-compose (recommended)
docker-compose up whisper-web

# Using direct Docker
docker run -p 8000:8000 -e MODE=web -v ~/.cache/whisper:/root/.cache/whisper whisper-cli
```

#### Web Interface
Open http://localhost:8000 in your browser for the HTML frontend.

#### API Endpoints
- `POST /transcribe` - Upload and transcribe audio files
- `GET /health` - Health check
- `GET /models` - List available models  
- `GET /docs` - Interactive API documentation
- `GET /openapi.json` - OpenAPI schema

#### API Usage Examples
```bash
# Transcribe to JSON
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "model=small" \
  -F "output_format=json"

# Transcribe to plain text
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model=base" \
  -F "output_format=txt"

# Transcribe with language hint
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.m4a" \
  -F "model=small" \
  -F "language=en" \
  -F "output_format=srt"
```

### Python Client

#### Generate Python Client
```bash
# Generate client from running API
python generate_client.py --url http://localhost:8000 --output whisper_client.py
```

#### Use Generated Client
```python
from whisper_client import WhisperClient

# Initialize client
client = WhisperClient("http://localhost:8000")

# Simple transcription
text = client.transcribe_to_text("audio.mp3", model="small")
print(text)

# Detailed transcription with segments
result = client.transcribe_with_segments("audio.wav", model="base")
print(f"Language: {result['language']}")
for segment in result['segments']:
    print(f"[{segment['start']:.1f}s] {segment['text']}")

# SRT subtitles
srt = client.transcribe_to_srt("video.mp4", model="medium")
with open("subtitles.srt", "w") as f:
    f.write(srt)
```

## Testing

### API Tests
```bash
# Run API tests (requires running web service)
python test_api.py

# Or using pytest
pytest test_api.py -v
```

### Manual Testing
```bash
# Start web service
docker-compose up whisper-web

# Test health endpoint
curl http://localhost:8000/health

# Test transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test.m4a" \
  -F "model=tiny" \
  -F "output_format=json"
```

## Available Models
- `tiny` - ~72MB, fastest, least accurate
- `base` - ~142MB, good balance
- `small` - ~461MB, better accuracy (default)
- `medium` - ~1.5GB, high accuracy
- `large` - ~2.9GB, highest accuracy

## Development Commands

### Building and Testing
```bash
# Build Docker image locally
./build.sh

# Start web service
docker-compose up whisper-web

# Run CLI transcription  
./transcribe audio.m4a

# Run API tests (requires running web service)
python test_api.py
pytest test_api.py -v

# Generate Python client
python generate_client.py --url http://localhost:8000
```

### CI/CD Pipeline
- **GitHub Actions** workflow at `.github/workflows/docker-build.yml` 
- **Multi-architecture builds** for linux/amd64 and linux/arm64
- **Container Registry** pushes to GitHub Container Registry (ghcr.io)
- **Triggers**: Push to main, tags, and pull requests
- **Monitoring**: Use GitHub Actions dashboard and GitHub tools to monitor pipeline results

### Development Environment
- Uses `devbox` to manage software dependencies
- Use the devbox CLI to install/remove/update software