## Project Overview

This is a Docker-based Whisper CLI transcription tool that converts audio files to text using OpenAI's Whisper model.

## Architecture

- **Docker Container**: Runs Python script with Whisper dependencies
- **Stdin Input**: Audio files are piped directly to the container
- **Output Separation**: Progress messages → stderr, transcription → stdout
- **Cache Persistence**: Model cache mounted to avoid re-downloading

## Usage

### Wrapper Script (Recommended)
```bash
./transcribe audio.m4a                    # Basic usage
./transcribe audio.wav --model base       # Different model
./transcribe audio.mp3 > transcript.txt   # Save to file
```

### Direct Docker
```bash
cat audio.m4a | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli
cat audio.wav | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli --model base
```

## Available Models
- `tiny` - ~72MB, fastest, least accurate
- `base` - ~142MB, good balance
- `small` - ~461MB, better accuracy (default)
- `medium` - ~1.5GB, high accuracy
- `large` - ~2.9GB, highest accuracy

## Development Environment

- I use `devbox` to manage software
- Use the devbox cli to install/remove/update software!
- Build with: `./build.sh`
- Test with: `./transcribe <audio_file>`