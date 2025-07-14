# Custom Transcription API Implementation

This document describes the implementation of the custom transcription API endpoint that follows the specification defined in `spec/custom-transcription-api.md`.

## Overview

The `/transcribe` endpoint provides synchronous transcription that matches the custom API specification for integration with Obsidian Post-Processor V2 and other external tools.

## Implementation Details

### Endpoint

- **URL**: `POST /transcribe`
- **Content-Type**: `multipart/form-data`
- **Authentication**: Bearer token (optional, controlled by `TRANSCRIPTION_API_KEY` environment variable)

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | File | Yes | Audio file to transcribe |
| `model` | String | No | Model name (tiny, base, small, medium, large) - defaults to "small" |
| `language` | String | No | Language code (e.g., "en", "de") or "auto" - defaults to auto-detection |
| `prompt` | String | No | Context prompt for better transcription |
| `temperature` | Float | No | Temperature for transcription (0.0-1.0) - defaults to 0.0 |

### Response Format

#### Success Response (200 OK)
```json
{
  "status": "success",
  "transcription": "This is the transcribed text",
  "language": "en",
  "confidence": 0.95,
  "duration": 30.5,
  "model": "small",
  "metadata": {
    "segments": [...],
    "transcription_duration": 2.5,
    "prompt": "...",
    "temperature": 0.0
  }
}
```

#### Error Response (4xx/5xx)
```json
{
  "status": "error",
  "error": "Brief error description",
  "code": "ERROR_CODE",
  "message": "Detailed error message"
}
```

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_REQUEST | Missing or invalid parameters |
| 400 | INVALID_MODEL | Unsupported model name |
| 400 | INVALID_LANGUAGE | Invalid language code |
| 400 | INVALID_TEMPERATURE | Temperature out of range |
| 401 | UNAUTHORIZED | Missing or invalid API key |
| 413 | FILE_TOO_LARGE | File exceeds 25MB limit |
| 415 | INVALID_FORMAT | Unsupported audio format |
| 500 | TRANSCRIPTION_FAILED | Transcription process failed |

### Supported Audio Formats

- MP3 (`.mp3`)
- M4A (`.m4a`)
- WAV (`.wav`)
- WEBM (`.webm`)
- OGG (`.ogg`)
- FLAC (`.flac`)

## Configuration

### Environment Variables

- `TRANSCRIPTION_API_KEY`: Optional API key for authentication
- `HOST`: Service host (default: 0.0.0.0)
- `PORT`: Service port (default: 8000)

### Authentication

If `TRANSCRIPTION_API_KEY` is set, all requests must include:
```
Authorization: Bearer <API_KEY>
```

## Usage Examples

### Basic Usage (No Authentication)
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.m4a" \
  -F "model=small" \
  -F "language=en"
```

### With Authentication
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your-api-key" \
  -F "audio=@recording.m4a" \
  -F "model=medium" \
  -F "language=auto" \
  -F "prompt=Meeting recording" \
  -F "temperature=0.2"
```

### Python Example
```python
import requests

# With authentication
headers = {"Authorization": "Bearer your-api-key"}
files = {"audio": open("recording.m4a", "rb")}
data = {
    "model": "small",
    "language": "en",
    "prompt": "Voice memo",
    "temperature": 0.1
}

response = requests.post(
    "http://localhost:8000/transcribe",
    files=files,
    data=data,
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"Transcription: {result['transcription']}")
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['confidence']}")
else:
    error = response.json()
    print(f"Error: {error['error']}")
```

## Testing

Use the provided test script to validate the implementation:

```bash
# Set API key if authentication is enabled
export TRANSCRIPTION_API_KEY="your-test-key"

# Run the test script
python test_custom_api.py
```

## Integration with Obsidian Post-Processor V2

The endpoint is designed to work seamlessly with Obsidian Post-Processor V2. Configuration example:

```yaml
processors:
  transcribe:
    type: "custom_api"
    config:
      api_url: "http://localhost:8000/transcribe"
      api_key: "${TRANSCRIPTION_API_KEY}"
      timeout: 300
      model: "small"
      language: "auto"
      temperature: 0.0
```

## Monitoring and Logging

The endpoint includes comprehensive logging and metrics:

- Request/response logging with structured JSON output
- Prometheus metrics for monitoring
- Error tracking with detailed context
- Performance metrics (duration, file size, etc.)

## Development Notes

- The endpoint maintains backward compatibility with existing async endpoints
- Temperature parameter is passed through to the underlying faster-whisper model
- Language detection from filename patterns is preserved
- All existing monitoring and metrics are maintained
- Thread-safe model loading and caching
