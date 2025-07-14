#!/bin/bash

# Build the Docker image
echo "Building whisper-cli Docker image..."
docker build -t whisper-cli .

echo "Build complete!"
echo ""
echo "Usage examples:"
echo "  # Using the wrapper script (recommended):"
echo "  ./transcribe Waldertgasse.m4a"
echo "  ./transcribe audio.wav --model base"
echo "  ./transcribe audio.mp3 --model medium > transcript.txt"
echo ""
echo "  # Using Docker directly:"
echo "  cat audio.m4a | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli"
echo "  cat audio.wav | docker run --rm -i -v ~/.cache/whisper:/root/.cache/whisper whisper-cli --model base"
