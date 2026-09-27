#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-Vision-Exp (multimodal) on 2x RTX PRO 6000 Blackwell (TP2)
# via the Jovian Judgement r3 image, using fixed probabilistic DSpark K3 (the deepest
# DSpark mode this checkpoint supports — it has exactly 3 next-token draft layers).
# Serving contract per the r3 spec: TP2/DCP1, b12x-a8-dglin backend, FP8 compressed
# MLA KV, GPU KV storage by default (LMCache RAM is an opt-in profile, see below).
# Runs detached in the background; you start it manually (no auto-restart).
# Re-run this script to (re)start.
set -euo pipefail

IMAGE="voipmonitor/vllm:jovian-judgement-vllmd6f9e77-b12x283a63e-fi803c466-cu133-torch213-20260904-r3"
NAME="ds4-vision-jovian-r3"
MODEL_DIR="/models/original/DeepSeek-V4-Flash-Vision-Exp"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/ds4-vision-jovian-r3"  # warm JIT/autotune cache; keep this
PORT="8000"

# ---- Download guard ---------------------------------------------------------
# The checkpoint is ~48 safetensors shards; refuse to start on a partial download.
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || \
   find "$MODEL_DIR/.cache" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: '$MODEL_DIR' looks like an incomplete download" >&2
    echo "       (missing model.safetensors.index.json or .incomplete shards present)." >&2
    echo "       Wait for the download to finish, then re-run this script." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR" "$CACHE_DIR/tmp"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"                 # container name (used to stop / inspect / log)
  --init                         # real init as PID 1 for clean signal handling
  --restart no                   # never auto-restart; you launch this manually
  --gpus all                     # expose GPUs to the container
  --runtime nvidia               # via the NVIDIA container runtime
  --ipc host                     # share host IPC namespace (large shared-mem tensors)
  --shm-size 32g                 # /dev/shm size for NCCL and worker comms
  --network host                 # host networking; PORT is a host port
  --ulimit memlock=-1            # unlimited locked memory (pinned / RDMA buffers)
  --ulimit nofile=1048576:1048576  # high fd limit for model files
  --ulimit stack=67108864        # 64 MiB thread stack
  -v /models:/models:ro          # mount the model tree (read-only)
  -v "$CACHE_DIR":/cache         # persist JIT / autotune cache across runs (reuse this release-scoped mount)
  -v "$CACHE_DIR/tmp":/container-tmp  # container scratch space (matches the r3 compose profile)

  # ---- GPU & NCCL ----------------------------------------------------------
  -e CUDA_VISIBLE_DEVICES=0,1    # use the first two GPUs (TP2)
  -e CUTE_DSL_ARCH=sm_120a       # target Blackwell sm_120a kernels
  -e NCCL_P2P_LEVEL=SYS          # permit GPU-to-GPU P2P across the PCIe system
  -e NCCL_PROTO=LL,LL128,Simple  # NCCL protocols allowed
  -e NCCL_IB_DISABLE=1           # no InfiniBand on this host
  -e NCCL_SOCKET_IFNAME=lo       # r3 compose default for single-node TP2
  -e GLOO_SOCKET_IFNAME=lo       # r3 compose default
  -e OMP_NUM_THREADS=2           # r3 compose default

  # ---- HuggingFace ---------------------------------------------------------
  -e HF_HUB_OFFLINE=1            # never contact Hugging Face; local weights only
  -e HF_HUB_ENABLE_FILE_HASHING=0 # skip hash checks on local files

  # ---- Model & serving config (Jovian Judgement r3 contract) ----------------
  -e MODEL="$MODEL_DIR"          # local checkpoint path (bypasses HF Hub)
  -e SERVED_MODEL_NAME=DeepSeek-V4-Flash-Vision-Exp  # name clients pass in the OpenAI "model" field
  -e DS4_MODEL_VARIANT=vision    # selects the Vision launcher path in the image
  -e PORT="$PORT"                # server port
  -e MODE=dspark                 # fixed DSpark serving (use dspark-mtp0 for target-only override)
  -e BACKEND=b12x-a8-dglin       # B12X W8A8 target path (the r3-qualified backend)
  -e TP_SIZE=2                   # tensor-parallel across 2 GPUs
  -e DCP_SIZE=1                  # decode context parallel = 1 (only qualified config)
  -e DSPARK_DEPTH_MODE=fixed     # fixed draft depth
  -e DSPARK_TOKENS=3             # K3: deepest DSpark mode this checkpoint supports (3 draft layers)
  -e DRAFT_SAMPLE_METHOD=probabilistic  # r3-qualified draft sampling
  -e MAX_NUM_SEQS=4              # scheduler concurrency (r3 contract value)
  -e MAX_NUM_BATCHED_TOKENS=4096 # prefill scheduler budget (r3 contract value)
  -e MAX_MODEL_LEN=900000        # was 1048576 (full 1M); capped at 900k so the KV pool fits
                                 # at GPU_MEMORY_UTILIZATION=0.95 (1M needs 6.95GiB KV, only 6.54GiB
                                 # available at 0.95 -> startup refused; see 2026-09-06 crash notes)
  -e GPU_MEMORY_UTILIZATION=0.95   # 0.975 (r3-qualified) runtime-OOMed: ~7-20MB free on both GPUs when
                                 # a 92k-token prefill landed next to 2 DSpark decodes; the fused
                                 # qnorm/rope/kv op couldn't map a 20MB scratch buffer (2026-09-06).
                                 # 0.95 + MAX_MODEL_LEN=900000 mirrors the r3 LMCache-RAM pairing
                                 # (900000/0.951) and leaves ~4.8GB/GPU of workspace headroom.
  -e GRAPH=auto                  # CUDA graphs FULL_AND_PIECEWISE, cap 16 (launcher default)
  -e ALLREDUCE_MODE=auto         # let the launcher pick the allreduce path
  -e LOAD_FORMAT=safetensors     # instanttensor cudaHostRegister failed on this host (driver 610.43.02,
                                 # see start_ds4_flash_0731.sh); retest with "instanttensor" if the
                                 # driver changed — r3 qualifies INSTANTTENSOR_BACKEND=BUFFERED
  -e LMCACHE_MODE=off            # GPU KV storage (default). Set "ram" for LMCache RAM replay,
                                 # which requires MAX_MODEL_LEN=900000 + GPU_MEMORY_UTILIZATION=0.951
  -e LMCACHE_TRANSFER_MODE=auto  # direct transfer when LMCache is enabled
  -e LMCACHE_L1_GB=8             # LMCache L1 size (only used when LMCACHE_MODE=ram)
  -e PYTHONHASHSEED=0            # r3 compose default (reproducible hashing)
)

docker run "${DOCKER_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc '
    # The image may ship PCIe / fused-all-reduce tunables that hurt this 2-GPU box.
    # Clear them so we fall back to the plain NCCL path, then launch the serve script
    # through the LMCache wrapper (a passthrough when LMCACHE_MODE=off).
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE \
          VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND \
          VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS \
          VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER \
          VLLM_CACHE_DIR
    exec /usr/local/bin/lmcache-mp-wrapper.sh /usr/local/bin/serve-ds4-flash.sh
  '

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
echo "First launch after an image change warms the cache (~5 min); reuses $CACHE_DIR otherwise."
echo "NOTE: port $PORT — do not run alongside the other DS4 scripts (they also use 8000)."
