#!/usr/bin/env bash
# Stop and remove the glm53-spark-tp2 container (GLM-5.3-Flash-NVFP4-Spark, Karmic Kraken beta).
# Keeps the host HF cache (/models/hf-cache) and the JIT/LMCache cache dir.
set -euo pipefail

NAME="glm53-spark-tp2"

if ! docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container '$NAME' not found — nothing to stop."
    exit 0
fi

state=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || true)

if [ "$state" = "running" ]; then
    echo "Stopping '$NAME' (grace period 60s)..."
    docker stop -t 60 "$NAME"
    echo "Stopped."
else
    echo "Container '$NAME' is '$state' — removing."
fi

docker rm "$NAME" >/dev/null 2>&1 || true
echo "Removed '$NAME'."
