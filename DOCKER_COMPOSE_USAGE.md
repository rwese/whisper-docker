# Docker Compose Usage Guide

## Quick Start

### Multithreaded Web Service (Recommended)
```bash
# Start the multithreaded web service with 4 workers
docker-compose up whisper-web

# Or run in background
docker-compose up -d whisper-web
```

### Single-threaded Web Service (for comparison)
```bash
# Start single-threaded service on port 8001
docker-compose --profile single up whisper-web-single

# Or run in background
docker-compose --profile single up -d whisper-web-single
```

## Available Services

### Default Service: `whisper-web`
- **Port**: 8000
- **Workers**: 4 (for background processing)
- **Features**: Async processing, monitoring endpoints

### Comparison Service: `whisper-web-single`
- **Port**: 8001
- **Workers**: 1 (single-threaded)
- **Profile**: `single` (only starts with --profile single)

## Usage Examples

### Basic Operations
```bash
# Start multithreaded service
docker-compose up -d whisper-web

# Check service health
curl http://localhost:8000/health

# Check service statistics
curl http://localhost:8000/stats

# Test transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "model=tiny" \
  -F "output_format=json"

# Stop services
docker-compose down
```

### Concurrent Testing
```bash
# Start both services for comparison
docker-compose up -d whisper-web
docker-compose --profile single up -d whisper-web-single

# Test multithreaded service (port 8000)
curl http://localhost:8000/health

# Test single-threaded service (port 8001)
curl http://localhost:8001/health

# Compare threading metrics
curl http://localhost:8000/stats | jq '.thread_pool'
curl http://localhost:8001/stats | jq '.thread_pool'
```

## Production Deployment

### High-Performance Configuration
```bash
# Use production compose file with 8 workers
docker-compose -f docker-compose.prod.yml up -d whisper-web

# With load balancer
docker-compose -f docker-compose.prod.yml --profile loadbalancer up -d

# With monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

### Configuration Options

#### Environment Variables
- `WORKERS`: Number of uvicorn workers (default: 4)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `MODE`: Service mode (web)

#### Volume Mounts
- `whisper_cache`: Persistent model cache
- `./storage`: Async task storage
- `./logs`: Application logs (prod only)

#### Health Checks
- **Endpoint**: `/health`
- **Interval**: 30s (15s in prod)
- **Timeout**: 10s
- **Retries**: 3
- **Start Period**: 30s (45s in prod)

## Monitoring

### Service Statistics
```bash
# Get detailed service metrics
curl http://localhost:8000/stats | jq .

# Monitor thread pool usage
curl http://localhost:8000/stats | jq '.thread_pool'

# Check memory and CPU usage
curl http://localhost:8000/health | jq '{memory_usage_mb, cpu_usage_percent}'
```

### Docker Logs
```bash
# View service logs
docker-compose logs -f whisper-web

# View logs with timestamps
docker-compose logs -f -t whisper-web
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8080:8000"  # Use 8080 instead of 8000
   ```

2. **Out of Memory**
   ```bash
   # Reduce worker count
   environment:
     - WORKERS=2
   ```

3. **Slow Model Loading**
   ```bash
   # Ensure model cache is persisted
   volumes:
     - whisper_cache:/root/.cache/whisper
   ```

### Performance Tuning

#### For High Load
```yaml
environment:
  - WORKERS=8  # Increase workers
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

#### For Low Memory
```yaml
environment:
  - WORKERS=2  # Reduce workers
deploy:
  resources:
    limits:
      memory: 4G
```

## API Endpoints

- `POST /transcribe` - Synchronous transcription (multithreaded)
- `POST /transcribe/async` - Asynchronous transcription
- `GET /tasks/{task_id}` - Check async task status
- `GET /tasks/{task_id}/result` - Get async task result
- `GET /health` - Health check with system metrics
- `GET /stats` - Service statistics
- `GET /models` - Available models
- `GET /` - Web interface
- `GET /docs` - API documentation