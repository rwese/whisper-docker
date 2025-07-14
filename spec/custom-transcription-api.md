# Custom Transcription Service API

This document defines the expected API interface for custom transcription services that work with Obsidian Post-Processor V2.

## API Endpoint

The transcription service should expose a REST API endpoint that accepts audio files and returns transcriptions.

### Base Configuration

```yaml
processors:
  transcribe:
    type: "custom_api"
    config:
      api_url: "http://localhost:8080/transcribe"
      api_key: "${TRANSCRIPTION_API_KEY}"  # Optional
      timeout: 300
      model: "whisper-base"  # Optional
      language: "auto"  # Optional
```

## Request Format

### HTTP Method
`POST`

### Content-Type
`multipart/form-data`

### Request Headers
```
Content-Type: multipart/form-data
Authorization: Bearer ${API_KEY}  # If api_key is configured
```

### Request Body (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio file to transcribe |
| `model` | String | No | Transcription model to use |
| `language` | String | No | Language code (e.g., "en", "de", "auto") |
| `prompt` | String | No | Context prompt for better transcription |
| `temperature` | Float | No | Temperature for transcription (0.0-1.0) |

### Example Request

```bash
curl -X POST http://localhost:8080/transcribe \
  -H "Authorization: Bearer your-api-key" \
  -F "audio=@recording.m4a" \
  -F "model=whisper-base" \
  -F "language=en" \
  -F "prompt=This is a meeting recording"
```

## Response Format

### Success Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "status": "success",
  "transcription": "This is the transcribed text from the audio file.",
  "language": "en",
  "confidence": 0.95,
  "duration": 30.5,
  "model": "whisper-base",
  "metadata": {
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "This is the transcribed text"
      }
    ]
  }
}
```

### Error Response

**Status Code:** `4xx` or `5xx`

**Content-Type:** `application/json`

```json
{
  "status": "error",
  "error": "Invalid audio format",
  "code": "INVALID_FORMAT",
  "message": "The uploaded audio file format is not supported"
}
```

## Response Fields

### Success Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | String | Yes | Always "success" for successful requests |
| `transcription` | String | Yes | The transcribed text |
| `language` | String | No | Detected or specified language code |
| `confidence` | Float | No | Confidence score (0.0-1.0) |
| `duration` | Float | No | Audio duration in seconds |
| `model` | String | No | Model used for transcription |
| `metadata` | Object | No | Additional metadata (segments, etc.) |

### Error Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | String | Yes | Always "error" for error responses |
| `error` | String | Yes | Brief error description |
| `code` | String | No | Error code for programmatic handling |
| `message` | String | No | Detailed error message |

## Supported Audio Formats

The service should support common audio formats:
- MP3 (`.mp3`)
- M4A (`.m4a`)
- WAV (`.wav`)
- WEBM (`.webm`)
- OGG (`.ogg`)
- FLAC (`.flac`)

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad Request (invalid parameters) |
| `401` | Unauthorized (invalid API key) |
| `413` | Payload Too Large (file too big) |
| `415` | Unsupported Media Type (invalid audio format) |
| `429` | Too Many Requests (rate limiting) |
| `500` | Internal Server Error |
| `503` | Service Unavailable |

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_FORMAT` | Unsupported audio format |
| `FILE_TOO_LARGE` | Audio file exceeds size limit |
| `INVALID_LANGUAGE` | Unsupported language code |
| `INVALID_MODEL` | Unsupported model name |
| `TRANSCRIPTION_FAILED` | Transcription process failed |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |

## Configuration Options

### Complete Configuration Example

```yaml
processors:
  transcribe:
    type: "custom_api"
    config:
      api_url: "http://localhost:8080/transcribe"
      api_key: "${TRANSCRIPTION_API_KEY}"
      timeout: 300
      retry_attempts: 3
      retry_delay: 1.0
      model: "whisper-medium"
      language: "auto"
      temperature: 0.0
      prompt: "Voice memo transcription:"
      max_file_size: 25000000  # 25MB
      supported_formats: ["mp3", "m4a", "wav", "webm"]
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_url` | String | Required | Base URL for transcription API |
| `api_key` | String | None | API key for authentication |
| `timeout` | Integer | 300 | Request timeout in seconds |
| `retry_attempts` | Integer | 3 | Number of retry attempts |
| `retry_delay` | Float | 1.0 | Delay between retries in seconds |
| `model` | String | None | Default model to use |
| `language` | String | "auto" | Default language code |
| `temperature` | Float | 0.0 | Default temperature |
| `prompt` | String | None | Default context prompt |
| `max_file_size` | Integer | 25MB | Maximum file size in bytes |

## Integration Example

### Processor Implementation

The custom API processor would make requests like this:

```python
import aiohttp
import aiofiles

async def transcribe_audio(audio_file_path: str, config: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=config.get('timeout', 300))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        data = aiohttp.FormData()

        # Add audio file
        async with aiofiles.open(audio_file_path, 'rb') as f:
            audio_data = await f.read()
            data.add_field('audio', audio_data, filename='audio.m4a')

        # Add optional parameters
        if config.get('model'):
            data.add_field('model', config['model'])
        if config.get('language'):
            data.add_field('language', config['language'])
        if config.get('prompt'):
            data.add_field('prompt', config['prompt'])

        # Set headers
        headers = {}
        if config.get('api_key'):
            headers['Authorization'] = f"Bearer {config['api_key']}"

        # Make request
        async with session.post(config['api_url'], data=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    'status': 'success',
                    'transcription': result['transcription'],
                    'metadata': result.get('metadata', {})
                }
            else:
                error_data = await response.json()
                return {
                    'status': 'error',
                    'error': error_data.get('error', 'Unknown error'),
                    'code': error_data.get('code', 'UNKNOWN')
                }
```

## Testing Your API

### Test with curl

```bash
# Test successful transcription
curl -X POST http://localhost:8080/transcribe \
  -H "Authorization: Bearer test-key" \
  -F "audio=@test.m4a" \
  -F "language=en"

# Expected response:
{
  "status": "success",
  "transcription": "This is a test transcription.",
  "language": "en",
  "duration": 5.2
}
```

### Test Configuration

```yaml
# Test configuration
processors:
  test_transcribe:
    type: "custom_api"
    config:
      api_url: "http://localhost:8080/transcribe"
      api_key: "test-key"
      timeout: 30
      model: "whisper-base"
      language: "en"
```

## Migration from V1

If you're migrating from V1, the main changes are:

1. **API Format**: Use JSON responses instead of plain text
2. **Authentication**: Use Bearer token instead of custom headers
3. **Error Handling**: Use structured error responses
4. **Configuration**: Use YAML configuration instead of environment variables

## Best Practices

1. **Async Support**: Ensure your API can handle concurrent requests
2. **Rate Limiting**: Implement proper rate limiting
3. **File Size Limits**: Set reasonable file size limits
4. **Error Messages**: Provide clear, actionable error messages
5. **Logging**: Log requests for debugging
6. **Health Checks**: Provide a health check endpoint
7. **Documentation**: Document your specific API extensions

## Example Implementation

See `examples/custom-transcription-server.py` for a complete Flask/FastAPI implementation example that follows this API specification.
