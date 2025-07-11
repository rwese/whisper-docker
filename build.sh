#!/bin/bash

# Build the Docker image
echo "Building whisper-cli Docker image..."
docker build -t whisper-cli .

echo "Build complete!"
echo ""
echo "Usage examples:"
echo "  # Using the wrapper script (recommended):"
echo "  ./transcribe Waldertgasse.m4a"
echo "  ./transcribe audio.wav --model base --format srt"
echo ""
echo "  # Using Docker directly:"
echo "  docker run --rm -v \$(pwd):/audio -v ~/.cache/whisper:/root/.cache/whisper whisper-cli /audio/Waldertgasse.m4a"
echo "  docker run --rm -v \$(pwd):/audio -v ~/.cache/whisper:/root/.cache/whisper whisper-cli /audio/Waldertgasse.m4a --model small --format srt"