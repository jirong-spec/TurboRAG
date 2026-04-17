#!/usr/bin/env bash
# Start Ollama with NVIDIA GPU support.
# Required because libggml-base.so.0 lives in ~/.local/lib/ollama
# and libnvidia-ml.so.1 needs the nvidia/current path.
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

pkill -f "ollama serve" 2>/dev/null
sleep 1

ollama serve "$@"
