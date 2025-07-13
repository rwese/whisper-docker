#!/bin/bash

# Entrypoint script to switch between CLI and web modes

# Set default values if not provided
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

if [ "$MODE" = "web" ]; then
    echo "Starting Whisper web service on ${HOST}:${PORT}..."
    exec python web_service.py
else
    echo "Starting Whisper CLI mode..."
    # If no arguments passed, default to reading from stdin
    if [ $# -eq 0 ]; then
        exec python transcribe.py /dev/stdin
    else
        exec python transcribe.py "$@"
    fi
fi

# Force rebuild Sat Jul 12 13:03:30 CEST 2025
