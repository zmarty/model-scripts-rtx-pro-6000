#!/usr/bin/env bash
# Start GLM-5.3-Flash (local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) on 2x RTX PRO 6000
# Blackwell Workstation (TP2/DCP2, MTP3 speculation, B12X attention/MoE/linear backends,
# FP8 KV) via the shared Karmic Kraken beta image.
# Recipe: local-inference-lab GLM Spark preset (glm53-spark-tp2). "Spark" names the
# checkpoint, not DGX Spark hardware.
#
# The model is pre-downloaded on the host (with /models/.venv/bin/hf download) into
# $HF_CACHE_DIR, which is bind-mounted as the container's HF cache — so the stock
# preset resolves the checkpoint locally instead of downloading inside the container:
#   HF_HOME=/models/hf-cache /models/.venv/bin/hf download \
#       local-inference-lab/GLM-5.3-Flash-NVFP4-Spark
#
# Runs detached in the background; you start/stop it manually (no auto-restart).
# Re-run this script to (re)start.
set -euo pipefail

IMAGE="ghcr.io/local-inference-lab/vllm:karmic-kraken-beta"
NAME="glm53-spark-tp2"
MODEL_ID="local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
HF_CACHE_DIR="/models/hf-cache"   # host HF cache (hub layout); bind-mounted into the container
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/glm53-spark-tp2"  # persistent /cache: JIT + LMCache tiers
PORT="8000"   # 8000 is also used by qwen38-flash-next-fp8-vllm on this host — do not run both at once
MODEL_CACHE="$HF_CACHE_DIR/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark"

# ---- Download guard ---------------------------------------------------------
# The checkpoint is ~58 files; refuse to start on a missing/partial download.
if [ ! -d "$MODEL_CACHE/snapshots" ] || \
   ! ls -d "$MODEL_CACHE/snapshots"/*/ >/dev/null 2>&1 || \
   find "$MODEL_CACHE" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: '$MODEL_ID' looks like a missing or incomplete download" >&2
    echo "       (no snapshot under $MODEL_CACHE, or .incomplete files present)." >&2
    echo "       Download it first:" >&2
    echo "         HF_HOME=$HF_CACHE_DIR /models/.venv/bin/hf download $MODEL_ID" >&2
    echo "       Then re-run this script." >&2
    exit 1
fi

mkdir -p "$HF_CACHE_DIR" "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags ---------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"                 # container name (used to stop / inspect / log)
  --init                         # real init as PID 1 for clean signal handling
  --restart no                   # never auto-restart; you launch this manually
  --gpus all                     # expose GPUs to the container (both RTX PRO 6000 cards)
  --runtime nvidia               # via the NVIDIA container runtime (--gpus "device=0,1" from
                                 # the recipe errors on this host: "cannot set both Count and DeviceIDs")
  -e NVIDIA_VISIBLE_DEVICES=0,1  # the two RTX PRO 6000 cards (change IDs as needed)
  --network host                 # host networking; PORT is a host port
  --ipc host                     # share host IPC namespace (large shared-mem tensors)
  --shm-size 32g                 # /dev/shm size for NCCL and worker comms
  --ulimit memlock=-1            # unlimited locked memory (pinned / RDMA buffers)
  --ulimit stack=67108864:67108864  # 64 MiB thread stack (recipe value)
  --security-opt seccomp=unconfined  # recipe default

  # ---- Volumes --------------------------------------------------------------
  -v "$HF_CACHE_DIR":/root/.cache/huggingface  # host HF cache with the pre-downloaded checkpoint
  -v "$CACHE_DIR":/cache         # persist JIT/B12X compile caches (+ LMCache disk tier) across runs

  # ---- Preset & serving ------------------------------------------------------
  -e PRESET=glm53-spark-tp2      # TP2/DCP2, MTP3, B12X backends, 2x96GB memory profile
  -e PORT="$PORT"                # API port; served model name is GLM-5.3-Flash
  -e HF_HUB_OFFLINE=1            # resolve the checkpoint only from the mounted cache;
                                 # drop this line if startup complains about offline mode
  # MTP3 enabled (no MTP_DEPTH override) since the 615.71.09 driver upgrade.
  # History: on driver 610.43.02 the NVFP4 draft head OOMed during load even with
  # KV at 3.5 GiB (5 MiB free of 95 GiB); if that regresses, re-add:
  #   -e MTP_DEPTH=0
  # and/or lower KV:
  #   -e KV_CACHE_MEMORY_BYTES=3758096384  # 3.5 GiB KV per GPU (recipe knob)

  # ---- Optional: LMCache RAM+disk prefix cache (recipe opt-in; default is GPU-only).
  # ---- Reserves PORT+10000..10002; keep API ports >= 3 apart if you run more servers.
  # -e CACHE_MODE=lmcache
  # -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2
  # -e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_GB=64   # set L2_ENABLED=0 for RAM-only
  # -e LMCACHE_MAX_CPU_WORKERS=4 -e LMCACHE_MAX_GPU_WORKERS=2
)

docker run "${DOCKER_ARGS[@]}" "$IMAGE"

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
echo "First startup loads the model, prepares kernels and captures graphs (several minutes)."
echo "Health:  curl -fsS http://127.0.0.1:$PORT/health"
echo "Models:  curl -fsS http://127.0.0.1:$PORT/v1/models   (API name: GLM-5.3-Flash)"
echo "NOTE: port $PORT is also used by qwen38-flash-next-fp8-vllm — do not run both at once."
echo "Keep the API on a trusted network or add authentication."
