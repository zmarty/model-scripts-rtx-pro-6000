#!/usr/bin/env bash
# Stop and remove the qwen38-flash-next-radixark-vllm container.
set -euo pipefail

NAME="qwen38-flash-next-radixark-vllm"

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
