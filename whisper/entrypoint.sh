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

VAD_MODEL_PATH="/app/models/ggml-silero-v5.1.2.bin"
if [ ! -f "$VAD_MODEL_PATH" ]; then
    echo "Downloading Silero VAD model..."
    /app/models/download-vad-model.sh silero-v5.1.2 /app/models
fi

# Start whisper-server with auto language detection and thread count.
# Anti-repetition flags mirror WHISPER_ANTI_REPETITION in openai_proxy.py
# so that both CLI and server paths behave identically on repetition.
/app/build/bin/whisper-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 --port 8080 \
    --language auto \
    --threads "$THREADS" \
    --max-context 0 \
    --suppress-nst \
    --entropy-thold 2.4 \
    --logprob-thold -1.0 \
    --no-speech-thold 0.6 &

# Start the proxy
uvicorn openai_proxy:app --host 0.0.0.0 --port 9091
