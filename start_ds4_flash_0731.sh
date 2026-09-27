#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 Blackwell (TP2) via the Gilded Gnosis r24 image.
# Uses DSpark K5 (fixed depth, 5 draft tokens) speculative decoding.
# Runs detached in the background; you start it manually (no auto-restart).
# Re-run this script to (re)start.
set -euo pipefail

IMAGE="voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si2b9bf2a-fi801d57a-cu132-20260803-r24"
NAME="ds4-0731-r24"
MODEL_DIR="/models/DeepSeek-V4-Flash-0731"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/ds4-0731-r24"   # warm JIT/autotune cache; keep this
PORT="8000"

mkdir -p "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"                 # container name (used to stop / inspect / log)
  --init                         # real init as PID 1 for clean signal handling
  --restart no                   # never auto-restart; you launch this manually
  --gpus all                     # expose GPUs to the container
  --runtime nvidia               # via the NVIDIA container runtime
  --privileged                   # required by the Gilded Gnosis image (GPU/NUMA access)
  --ipc host                     # share host IPC namespace (large shared-mem tensors)
  --shm-size 32g                 # /dev/shm size for NCCL and worker comms
  --network host                 # host networking; PORT is a host port
  --ulimit memlock=-1            # unlimited locked memory (pinned / RDMA buffers)
  --ulimit nofile=1048576:1048576  # high fd limit for model files
  --ulimit stack=67108864        # 64 MiB thread stack
  -v /models:/models:ro          # mount the model tree (read-only)
  -v "$CACHE_DIR":/cache         # persist JIT / autotune cache across runs

  # ---- GPU & NCCL ----------------------------------------------------------
  -e CUDA_VISIBLE_DEVICES=0,1    # use the first two GPUs (TP2)
  -e CUTE_DSL_ARCH=sm_120a       # target Blackwell sm_120a kernels
  -e NCCL_P2P_LEVEL=SYS          # permit GPU-to-GPU P2P across the PCIe system
  -e NCCL_PROTO=LL,LL128,Simple  # NCCL protocols allowed
  -e NCCL_IB_DISABLE=1           # no InfiniBand on this host

  # ---- HuggingFace ---------------------------------------------------------
  -e HF_HUB_OFFLINE=1            # never contact Hugging Face; local weights only
  -e HF_HUB_ENABLE_FILE_HASHING=0 # skip hash checks on local files

  # ---- Model & serving config (DSpark K5 release defaults) -----------------
  -e MODEL_PATH="$MODEL_DIR"     # local path to the 0731 checkpoint (bypasses HF Hub)
  -e SERVED_MODEL_NAME=DeepSeek-V4-Flash  # match the old model name for client compatibility
  -e PORT="$PORT"                # server port
  -e MODE=dspark                 # native DSpark serving for the 0731 checkpoint
  -e BACKEND=b12x-a8             # SparkInfer/B12X W4A8 target path
  -e TP_SIZE=2                   # tensor-parallel across 2 GPUs
  -e DCP_SIZE=1                  # no data-copy parallelism
  -e DSPARK_DEPTH_MODE=fixed     # fixed draft depth (dynamic confidence control is opt-in)
  -e DSPARK_TOKENS=5             # K5 profile (13.3% faster decode than K7)
  -e MAX_NUM_SEQS=8              # scheduler concurrency
  -e MAX_MODEL_LEN=262144        # extended to 256K
  -e MAX_NUM_BATCHED_TOKENS=2048 # prefill scheduler budget (reduced for memory)
  -e GPU_MEMORY_UTILIZATION=0.98  # GPU memory target (leave ~1 GiB headroom for prefill activation)
  -e LOAD_FORMAT=safetensors     # instanttensor cudaHostRegister still fails on this host (driver 610.43.02)
  -e KV_OFFLOADING_SIZE=0        # native CPU KV offload disabled (set to e.g. 48.5 to enable)
  -e DSPARK_MODEL="$MODEL_DIR"    # DSpark target model (used for spec config when MODEL_PATH is set)
)

docker run "${DOCKER_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc '
    # The image may ship PCIe / fused-all-reduce tunables that hurt this 2-GPU box.
    # Clear them so we fall back to the plain NCCL path, then launch the serve script.
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE \
          VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND \
          VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS \
          VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER \
          VLLM_CACHE_DIR
    exec /usr/local/bin/serve-ds4-flash.sh
  '

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
echo "First launch after an image change warms the cache (~5 min); reuses $CACHE_DIR otherwise."
