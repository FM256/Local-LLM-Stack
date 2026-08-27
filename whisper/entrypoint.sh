#!/bin/bash
set -e

cd /app

MODEL_SIZE=${MODEL_SIZE:-small}
MODEL_PATH="/app/models/ggml-${MODEL_SIZE}.bin"

# Set threads (default to 4 if not set, or you could use `nproc` to auto-detect)
THREADS=${WHISPER_THREADS:-4}

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $MODEL_SIZE model..."
    /app/models/download-ggml-model.sh "$MODEL_SIZE" /app/models
fi

# Start whisper-server with auto language detection and thread count
/app/build/bin/whisper-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 --port 8080 \
    --language auto \
    --threads "$THREADS" &

# Start the proxy
uvicorn openai_proxy:app --host 0.0.0.0 --port 9091
