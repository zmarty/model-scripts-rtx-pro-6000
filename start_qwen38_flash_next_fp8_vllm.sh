#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next (official FP8 flavor) on 2x RTX PRO 6000 Blackwell
# (TP2) via the dedicated vLLM Qwen3.8 Flash-Next nightly image.
#
# Checkpoint: /models/fp8/Qwen-Qwen3.8-Flash-Next-FP8   (185.6 GB)
#   arch:  Qwen4ExpForConditionalGeneration / model_type=qwen4_exp (same as the
#          NVFP4 checkpoints, so the same dedicated image + flags apply)
#   quant: whole-model fine-grained FP8 W8A8 (dynamic activations, 128x128
#          weight blocks); lm_head/embeddings/hyper-connection/conv1d excluded.
#
# NO PATCH NEEDED (unlike the RadixArk NVFP4 flavor): the outer quant config
# IS Fp8Config, so stock vLLM's FP8 PLE gate fires natively and the FP8 PLE
# n-gram table (with its ngram_embedding.weight_scale) loads correctly.
# Recipe cross-validated by local-inference-lab/rtx6kpro PR #87 on the same
# image build (v0.1.dev20073+g8e685d198) and hardware class.
#
# PLE CPU OFFLOAD IS MANDATORY for this checkpoint: 185.6 GB total weights
# cannot fit in 2x 97.9 GB (0.95 util ~= 186 GB) with any KV cache left, so
# the ~48 GB FP8 PLE table lives in pinned host RAM (needs >= ~50 GB free)
# and is prefetched per step over PCIe. GPU-resident weights ~= 137 GB.
# (The NVFP4 flavor avoids this because its FP8 PLE table halves the size;
#  see start_qwen38_flash_next_radixark_vllm.sh for the full-residency option.)
#
# Compilation: full inductor compile + FlashInfer autotune left ON (validated
# by PR #87 for this FP8 path). The inductor-autotune profiling clone of the
# PLE table that OOMs the full-residency NVFP4 setup is a non-issue here
# because the table is offloaded to host RAM.
#
# Expected KV pool (derived from our NVFP4 measurements, same BF16 KV/token):
#   ~64 GiB/GPU weights at 0.95 util -> ~1.9-2.1M tokens (~7.5x @ 262K ctx).
#
# Machine notes (same box profile as the other qwen38_flash_next scripts):
#   - 2x RTX PRO 6000 Blackwell -> TP2; sm_120 cannot use custom all-reduce.
#   - Dedicated image is required; pulled on first run.
#   - Only one model served at a time on this box (port 8000, both GPUs).
#
# Runs detached; re-run this script to (re)start. Stop with stop_qwen38_flash_next_fp8_vllm.sh
set -euo pipefail

IMAGE="vllm/vllm-openai:qwen38-flash-next"   # dedicated recipe image (nightly)
NAME="qwen38-flash-next-fp8-vllm"
MODEL="/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8"

[ -d "$MODEL" ] || { echo "ERROR: missing $MODEL (download: hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir $MODEL)" >&2; exit 1; }
PORT="8000"                                   # only one model served at a time on this box
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/qwen38-flash-next-fp8-vllm"  # warm JIT/autotune cache
EAGER="${EAGER:-0}"              # 0 = compiled/graphs (fast); 1 = --enforce-eager (debug)
SPEC="${SPEC:-1}"                # 1 = MTP speculative decoding (3 draft tokens)
SPEC_TOKENS="${SPEC_TOKENS:-3}"  # draft tokens per step
# vLLM default (full inductor compile) validated for this FP8 path by PR #87;
# set to graphs-only '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' to compare:
COMPILATION_CONFIG="${COMPILATION_CONFIG:-off}"
# FP8 path: no backend pin needed (validated unpinned with MTP by PR #87);
# set MOE_BACKEND explicitly to pin one:
MOE_BACKEND="${MOE_BACKEND:-auto}"

# PLE offload wants >= ~50 GB free host RAM for the FP8 table.
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$avail_gb" -lt 50 ]; then
  echo "ERROR: FP8 PLE CPU offload wants >= ~50 GB free host RAM, only ${avail_gb} GB available." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
# Make sure we have the dedicated image (no-op if already present).
echo "Pulling $IMAGE (if needed)..."
docker pull "$IMAGE" >/dev/null 2>&1 || true

docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"
  --init
  --restart no                   # manual only
  --gpus '"device=0,1"'
  --ipc host
  --network host                 # host networking; PORT is a host port
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v /models:/models
  -v "$CACHE_DIR":/cache
  -e CUDA_VISIBLE_DEVICES=0,1
  -e HF_HUB_OFFLINE=1            # local weights only
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # PLE table in pinned host RAM (mandatory for this checkpoint, see header):
  --cap-add=SYS_PTRACE           # pidfd_getfd for CUDA IPC (PLE offload worker)
  -e VLLM_PLE_CPU_OFFLOAD=1
  # rtx6kpro PR #87: pin container RAM so the ~48 GB offloaded table can
  # never be swapped out (memswap == mem => no swap for the container).
  --memory 62g --memory-swap 62g
)

# ---- vLLM server arguments ------------------------------------------------
# ENTRYPOINT is already ["vllm", "serve"], so we pass args directly.
VLLM_ARGS=(
  "$MODEL"

  --served-model-name qwen3.8-flash-next
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"

  # no --quantization pin: auto-detected Fp8Config (also enables the native
  # FP8 PLE path; do NOT pin modelopt_fp4 here)
  --tensor-parallel-size 2      # 2x RTX PRO 6000 Blackwell
)

[ "$EAGER" = "1" ] && VLLM_ARGS+=(--enforce-eager)   # debug only
# compilation config (default "off" = vLLM default full inductor compile;
# validated for this FP8+offload path by PR #87):
if [ "$EAGER" != "1" ] && [ -n "$COMPILATION_CONFIG" ] && [ "$COMPILATION_CONFIG" != "off" ]; then
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi
# MTP speculative decoding (built-in MTP head):
[ "$SPEC" = "1" ] && VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
# MoE backend (default "auto" = unpinned; validated with MTP by PR #87):
[ "$MOE_BACKEND" != "auto" ] && VLLM_ARGS+=(--moe-backend "$MOE_BACKEND")

VLLM_ARGS+=(
  --gpu-memory-utilization 0.90 # rtx_pro_6000 profile (PR #87 used 0.97)
  --max-num-seqs 16             # rtx_pro_6000 profile
  --max-num-batched-tokens 8192 # rtx_pro_6000 profile
  --enable-prefix-caching
  # NOTE: FlashInfer autotune intentionally left ON for this FP8 path
  # (validated by PR #87); the NVFP4 scripts disable it.

  --disable-custom-all-reduce   # Blackwell sm_120: custom all-reduce not supported

  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml  # Qwen3 XML tool-calling
  --reasoning-parser qwen3      # Qwen3 reasoning extraction

  # ---- Optional (opt-in) features ---------------------------------------
  # -- Skip the vision encoder for text-only workloads (saves KV-cache memory):
  #   --language-model-only
  # -- Extend to 1M tokens via static YaRN:
  #   --rope-scaling "{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":262144}"
  #   --max-model-len 1000000     (NOTE: requires VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 env)
)

# ---- Launch ---------------------------------------------------------------
docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"

echo "Started '$NAME' on port $PORT. Follow startup with:  docker logs -f $NAME"
echo "Verify with: curl -s http://localhost:$PORT/v1/models"
