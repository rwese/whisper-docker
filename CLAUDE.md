# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Docker-based Whisper transcription service that converts audio files to text using OpenAI's Whisper model. It supports both CLI and web service modes controlled by the `MODE` environment variable.

## Core Architecture

### Dual-Mode Design
- **CLI Mode** (`MODE=cli`): Traditional command-line interface for file-based transcription
- **Web Mode** (`MODE=web`): FastAPI-based REST API with async-only endpoints and HTML frontend

### Threading and Concurrency Model
- **Thread-Safe Model Loading**: Whisper models are loaded lazily with proper locking mechanisms
- **Async Processing**: Background processing with expanded ThreadPoolExecutor (8 workers)
- **Multi-Worker Support**: Configurable uvicorn workers for horizontal scaling within a single process
- **Request Timeouts**: Built-in task timeout protection
- **Resource Monitoring**: Enhanced health checks with memory, CPU, and threading metrics

### Logging and Monitoring Architecture
- **Structured Logging**: Uses `structlog` for JSON-formatted logs with contextual information
- **Stdout/Stderr Output**: All operations output to stdout/stderr for easy host-level monitoring
- **Prometheus Metrics**: Comprehensive metrics collection including request counts, durations, file sizes, and error rates
- **Request Tracing**: Every request is logged with unique identifiers, duration, and metadata
- **Model Loading Tracking**: Model loading events are logged with timing information
- **Error Handling**: All errors are logged with context and stack traces

### Key Components
- **transcription_core.py**: Centralized transcription logic using `faster-whisper`, shared between both modes. Handles automatic language detection from filename patterns (`_en.m4a`, `_de.m4a`) and supports multiple output formats (txt, json, srt, vtt, tsv). Enhanced with comprehensive logging and stdout output
- **transcribe.py**: CLI entry point with stdin support and argument parsing
- **web_service.py**: FastAPI web service with async endpoints for transcription, health checks, API documentation, and Prometheus metrics. Includes comprehensive logging middleware and request tracing
- **static/index.html**: Modern HTML5 frontend with drag-and-drop file upload
- **entrypoint.sh**: Dynamic mode switching between CLI and web service based on environment
- **generate_client.py**: Auto-generates Python client from OpenAPI schema
- **async_storage.py**: Async task storage and processing with background workers

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

##### Docker Compose (Recommended)
```bash
# Multithreaded web service (4 workers) - Recommended
docker-compose up -d whisper-web

# Single-threaded mode for comparison
docker-compose --profile single up -d whisper-web-single

# Production deployment (8 workers)
docker-compose -f docker-compose.prod.yml up -d whisper-web
```

##### Direct Docker
```bash
# Single-threaded mode
docker run -p 8000:8000 -e MODE=web -v ~/.cache/whisper:/root/.cache/whisper whisper-cli

# Multi-threaded mode
docker run -p 8000:8000 -e MODE=web -e WORKERS=4 -v ~/.cache/whisper:/root/.cache/whisper whisper-cli
```

##### Direct Python (Development)
```bash
# Single-threaded mode
python web_service.py

# Multi-threaded mode
WORKERS=4 python web_service.py
```

#### Web Interface
Open http://localhost:8000 in your browser for the HTML frontend.

#### API Endpoints
- `POST /transcribe/async` - Submit async transcription tasks
- `GET /tasks/{task_id}` - Check task status
- `GET /tasks/{task_id}/result` - Get transcription result
- `GET /health` - Enhanced health check with system metrics
- `GET /stats` - Service statistics and monitoring
- `GET /models` - List available models
- `GET /metrics` - Prometheus metrics endpoint for monitoring
- `GET /docs` - Interactive API documentation
- `GET /openapi.json` - OpenAPI schema

#### API Usage Examples
```bash
# Submit async transcription task
curl -X POST http://localhost:8000/transcribe/async \
  -F "file=@audio.mp3" \
  -F "model=small" \
  -F "output_format=json"

# Check task status
curl http://localhost:8000/tasks/TASK_ID

# Get transcription result
curl http://localhost:8000/tasks/TASK_ID/result

# Get result in specific format
curl http://localhost:8000/tasks/TASK_ID/result?format=txt

# Transcribe with language hint
curl -X POST http://localhost:8000/transcribe/async \
  -F "file=@audio.m4a" \
  -F "model=small" \
  -F "language=en" \
  -F "output_format=srt"

# Check service health and performance
curl http://localhost:8000/health | jq .
curl http://localhost:8000/stats | jq .
```

#### Monitoring and Performance

The service provides comprehensive monitoring capabilities:

```bash
# Enhanced health check with system metrics
curl http://localhost:8000/health

# Service statistics and threading information
curl http://localhost:8000/stats

# Prometheus metrics for external monitoring
curl http://localhost:8000/metrics

# Example health response
{
  "status": "healthy",
  "message": "Whisper transcription service is running",
  "uptime_seconds": 3600,
  "memory_usage_mb": 245.5,
  "cpu_usage_percent": 15.2,
  "active_threads": 8,
  "thread_pool_active": 2
}
```

#### Available Prometheus Metrics
- `whisper_requests_total` - Total number of requests by method, endpoint, and status code
- `whisper_request_duration_seconds` - Request duration histogram by method and endpoint
- `whisper_transcription_duration_seconds` - Transcription duration histogram by model
- `whisper_file_size_bytes` - File size histogram for uploaded files
- `whisper_active_requests` - Current number of active requests
- `whisper_loaded_models` - Number of loaded Whisper models
- `whisper_thread_pool_active` - Active threads in the transcription thread pool
- `whisper_task_queue_size` - Size of the async task queue
- `whisper_model_load_duration_seconds` - Model loading duration by model
- `whisper_errors_total` - Total errors by type and endpoint

#### Logging Configuration
The service uses structured logging with JSON output. Set the `LOG_LEVEL` environment variable to control verbosity:
- `DEBUG` - Detailed debugging information
- `INFO` - Standard operational information (default)
- `WARNING` - Warning messages only
- `ERROR` - Error messages only

#### Host-Level Monitoring
All operations output timestamped messages to stdout/stderr for easy integration with log management systems:
- Transcription start/completion messages to stdout
- Error messages to stderr
- Model loading events to stdout
- Task creation/completion messages to stdout

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

# Simple async transcription (blocks until complete)
text = client.transcribe_to_text("audio.mp3", model="small", timeout=60)
print(text)

# Detailed async transcription with segments
result = client.transcribe_with_segments("audio.wav", model="base", timeout=60)
print(f"Language: {result['language']}")
for segment in result['segments']:
    print(f"[{segment['start']:.1f}s] {segment['text']}")

# SRT subtitles
srt = client.transcribe_to_srt("video.mp4", model="medium", timeout=120)
with open("subtitles.srt", "w") as f:
    f.write(srt)

# Manual async workflow
task_id = client.submit_async_transcription("audio.mp3", model="small")
print(f"Task submitted: {task_id}")

# Check status
status = client.get_task_status(task_id)
print(f"Status: {status['status']}")

# Get result when ready
if status['status'] == 'completed':
    result = client.get_task_result(task_id)
    print(f"Result: {result['text']}")
```

## Testing

### API Tests
```bash
# Run API tests (requires running web service)
python test_api.py

# Or using pytest
pytest test_api.py -v
```

### Concurrent Testing
The service includes tools for testing multithreaded performance:

```bash
# Test concurrent requests (requires test_audio.wav)
python test_concurrent.py

# Or use the shell script version
./test_concurrent.sh

# Test with multiple workers
WORKERS=4 python web_service.py &
python test_concurrent.py
```

### Manual Testing
```bash
# Start web service (single-threaded)
docker-compose up whisper-web

# Start web service (multi-threaded)
WORKERS=4 python web_service.py

# Test health endpoint with metrics
curl http://localhost:8000/health | jq .

# Test service statistics
curl http://localhost:8000/stats | jq .

# Test async transcription
curl -X POST http://localhost:8000/transcribe/async \
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