# Whisper CLI - Docker Audio Transcription

A portable Docker-based audio transcription tool using OpenAI's Whisper with true streaming output.

## Quick Start

No build required! Just use the pre-built Docker images:

```bash
# Transcribe an audio file (auto-detects language)
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest

# With language detection from filename
cat meeting_en.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest

# Force a specific language
cat audio.wav | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --language de

# Use a different model size
cat audio.mp3 | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --model medium

# Save output to file formats
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --format srt > subtitles.srt
```

## Features

- **True Streaming**: See transcription results as they're processed
- **Automatic Language Detection**: Detects language from filename patterns (`_en.m4a`, `_de.m4a`)
- **Multiple Output Formats**: txt, json, srt, vtt, tsv
- **Multi-Architecture**: Supports both Intel/AMD64 and ARM64 (Apple Silicon)
- **No Local Dependencies**: Everything runs in Docker
- **Model Caching**: Whisper models are cached for faster subsequent runs

## Usage

### Basic Transcription

```bash
# Stream transcription to stdout
cat your-audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest
```

### Language Detection

The tool automatically detects language from filename patterns:

```bash
# Auto-detects English
cat presentation_en.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest

# Auto-detects German  
cat meeting_de.wav | docker run --rm -i ghcr.io/rwese/whisper-docker:latest

# Force specific language
cat audio.mp3 | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --language fr
```

### Model Sizes

Choose the right balance of speed vs accuracy:

```bash
# Fastest (tiny model)
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --model tiny

# Balanced (default)
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --model small

# Highest accuracy (large model)
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --model large
```

### Output Formats

```bash
# Plain text (default)
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest

# JSON with timestamps
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --format json

# SRT subtitles
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --format srt

# WebVTT subtitles
cat audio.m4a | docker run --rm -i ghcr.io/rwese/whisper-docker:latest --format vtt
```

## Convenience Script

For easier usage, clone the repository and use the wrapper script:

```bash
git clone https://github.com/rwese/whisper-docker.git
cd whisper-docker
./transcribe audio.m4a
```

The wrapper script handles the Docker command for you and supports all the same options.

## Supported Architectures

- `linux/amd64` - Intel/AMD 64-bit processors
- `linux/arm64` - ARM 64-bit processors (Apple Silicon, ARM servers)

## Model Information

| Model  | Size   | Speed      | Accuracy |
|--------|--------|------------|----------|
| tiny   | ~39 MB | Fastest    | Basic    |
| base   | ~142 MB| Fast       | Good     |
| small  | ~466 MB| Balanced   | Better   |
| medium | ~1.5 GB| Slower     | High     |
| large  | ~2.9 GB| Slowest    | Highest  |

## Performance Tips

1. **Model Caching**: Models are downloaded on first use and cached. Mount a volume to persist cache:
   ```bash
   cat audio.m4a | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper ghcr.io/rwese/whisper-docker:latest
   ```

2. **Batch Processing**: For multiple files, the cache will make subsequent transcriptions much faster.

3. **Model Selection**: Start with `small` model for good balance of speed and accuracy.

## Options

```
--model, -m          Model size (tiny, base, small, medium, large)
--language, -l       Force language (e.g., en, de, fr)
--format, -f         Output format (txt, json, srt, vtt, tsv)
--no-stream          Disable streaming output
--help               Show help message
```

## Development

To build locally:

```bash
docker build -t whisper-cli .
cat audio.m4a | docker run --rm -i whisper-cli
```

## License

This project uses OpenAI's Whisper model. Please refer to OpenAI's usage policies.