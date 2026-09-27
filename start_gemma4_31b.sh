#!/usr/bin/env bash
# Start Gemma 4 31B IT (BF16) + MTP assistant drafter on 2x RTX PRO 6000 (TP2)
# via the official vLLM CUDA 12.9 nightly image.
# Uses Gemma 4's native ~0.5B MTP drafter for speculative decoding (4 tokens).
#
# Based on:
#   https://github.com/theogravity/dual-rtx-6000-blackwell-Gemma-4-31B-IT-NVFP4
# Key differences: BF16 checkpoint (not NVFP4), local model paths (offline).
#
# Low-concurrency / interactive-coding tuned. Runs detached; re-run to (re)start.
set -euo pipefail

IMAGE="vllm/vllm-openai:cu129-nightly"
NAME="gemma4-31b-it"
MODEL="/models/original/google-gemma-4-31B-it"
DRAFT_MODEL="/models/original/google-gemma-4-31B-it-assistant"
CHAT_TEMPLATE="${MODEL}/tool_chat_template_gemma4.jinja"
PORT="8000"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"
  --init
  --restart no                   # manual only
  --gpus '"device=0,1"'
  --ipc host
  --network host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v /models:/models
  -e CUDA_VISIBLE_DEVICES=0,1
  -e HF_HUB_OFFLINE=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

# ---- vLLM server arguments ------------------------------------------------
# ENTRYPOINT is already ["vllm", "serve"], so we pass args directly.
VLLM_ARGS=(
  "$MODEL"

  --served-model-name gemma-4-31B
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"

  --tensor-parallel-size 2
  --gpu-memory-utilization 0.90
  --max-model-len 262144
  --max-num-seqs 8
  --max-num-batched-tokens 8192

  --enable-chunked-prefill
  --enable-prefix-caching

  --tool-call-parser gemma4
  --reasoning-parser gemma4
  --enable-auto-tool-choice
  --chat-template "$CHAT_TEMPLATE"
  --default-chat-template-kwargs '{"enable_thinking": true}'

  --disable-custom-all-reduce                 # Blackwell sm_120: custom all-reduce not supported

  --speculative-config "{\"model\": \"${DRAFT_MODEL}\", \"num_speculative_tokens\": 4}"
)

# ---- Launch ---------------------------------------------------------------
docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
