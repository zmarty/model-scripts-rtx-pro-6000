#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next (NVFP4, RadixArk flavor) on 2x RTX PRO 6000 Blackwell
# (TP2) via the dedicated vLLM Qwen3.8 Flash-Next nightly image.
#
# Checkpoint: /models/nvfp4/RadixArk-Qwen3.8-Flash-Next-NVFP4   (126 GB)
#   arch:  Qwen4ExpForConditionalGeneration / model_type=qwen4_exp (same as the
#          Inferact checkpoint, so the same dedicated image + flags apply)
#   quant: modelopt NVFP4 W4A4 on the routed experts only (group_size 16);
#          attention / GDN / shared experts / MTP stay BF16.
#
# Why this flavor is ~45 GB smaller than Inferact (171 GB -> 126 GB):
#   the 51B-param PLE n-gram embedding table ships FP8 E4M3 (~48 GB) instead of
#   BF16 (~91 GB). The FP4 experts (64 GB) and BF16 remainder (~15 GB) are the
#   same in both.
#
# Consequence: FULL GPU RESIDENCY is possible AND fast (see MEASURED):
#   GPU-resident weights ~= 64 + 15 + 48 + 1.5 (MTP) ~= 128.5 GB; 2x 97.9 GB at
#   0.95 util ~= 186 GB, leaving ~57 GB for KV cache + CUDA graphs.
#
# REQUIRED PATCH: stock vLLM cannot load this checkpoint's FP8 PLE table.
#   It only enables the FP8 PLE path when the outer quant config is Fp8Config
#   (here: modelopt), so it dies with
#     ValueError: no module or parameter named 'ngram_embedding.weight_scale'
#   This script bind-mounts a patched ple_layer.py over the image file; see
#   /models/patches/qwen38flashnext-sm120/PROVENANCE.md.
#   PLE_PATCH=minimal (default): stock file + the 11-line ple_embedding_dtype
#     gate patch from rtx6kpro PR #87. Once the FP8 method is selected, the
#     STOCK loader registers/loads weight_scale and the stock
#     _dequantize_embeddings applies it - no other changes needed, in both
#     full-residency and PLE_OFFLOAD=1 modes.
#   PLE_PATCH=x00byte (fallback): the earlier full-file replacement from the
#     dual-Spark recipe (gate + weight_scale parameter->buffer conversion that
#     only their own offload-worker changes required).
#   The patch keeps the table FP8 in VRAM AND applies the exported scale at
#   lookup (unpatched builds that don't crash serve wrong embeddings).
#   NOTE: VLLM_PLE_FP8_CHECKPOINT (seen in some online examples) is NOT a real
#   upstream vLLM env var - it exists in one private fork only; this image
#   (v0.1.dev20073+g8e685d198) ignores it. The patch is the actual fix.
#
# MEASURED on this box (1000-token decode, single request, temp 0, 2026-08-28):
#   config                                                        tok/s   host RAM  KV pool
#   EAGER=1 PLE_OFFLOAD=0 (fully eager, full VRAM)                  ~18     0         1.94M tok
#   EAGER=0 PLE_OFFLOAD=1 (inductor compile, table offloaded)       ~94     ~48 GB    3.86M tok
#   EAGER=0 PLE_OFFLOAD=0 + graphs-only cc + marlin                ~114     0         1.94M tok
#   EAGER=0 PLE_OFFLOAD=0 + graphs-only cc + MTP + cutlass         ~138     0         1.48M tok  [DEFAULT]
# Key insights:
#   - The eager PLE path on this build is pathologically slow (5x), way beyond
#     the normal CUDA-graph tax. Never run fully eager except to debug.
#   - The startup OOM that blocked "compile + full residency" comes from torch
#     INDUCTOR autotune cloning the full 47.69 GiB PLE table during profiling
#     (64 GiB resident + 47.69 GiB > 95 GiB/card). --enforce-eager is just
#     shorthand for -cc.mode=none -cc.cudagraph_mode=none, and the two are
#     decoupled: mode=none (no inductor -> no clone) + cudagraph_mode=
#     FULL_DECODE_ONLY (graphs around the eager model) starts fine and is the
#     FASTEST config - on-GPU PLE rows beat the offload path's PCIe prefetch.
#   - SPEC=1 (MTP) requires MOE_BACKEND=flashinfer_cutlass: the MTP draft
#     head's MoE is UNQUANTIZED BF16 and marlin is NVFP4-only.
#   - KV pool is smaller with MTP (draft head weights/activations) and largest
#     with offload; 1.48M tokens is still 5.65x concurrency at 262K.
#
# DEFAULT (fastest, zero host RAM): EAGER=0 PLE_OFFLOAD=0 SPEC=1
#   COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
#   MOE_BACKEND=flashinfer_cutlass
# Fallbacks: PLE_OFFLOAD=1 COMPILATION_CONFIG=off SPEC=0 (inductor + offload,
#   ~94 tok/s, biggest KV pool) or EAGER=1 PLE_OFFLOAD=0 (debug, ~18 tok/s).
#
# Machine notes (same box profile as start_qwen38_flash_next_inferact_vllm.sh):
#   - 2x RTX PRO 6000 Blackwell -> TP2.
#   - Blackwell sm_120 cannot use custom all-reduce.
#   - Dedicated image is required (PyPI install unsupported); pulled on first run.
#   - Only one model served at a time on this box: stop the Inferact container
#     before starting this one.
#
# Optional features (opt-in) are listed as comments at the bottom:
#   MTP speculative decoding, text-only, YaRN 1M.
#
# Runs detached; re-run this script to (re)start. Stop with stop_qwen38_flash_next_radixark_vllm.sh
set -euo pipefail

IMAGE="vllm/vllm-openai:qwen38-flash-next"   # dedicated recipe image (nightly)
NAME="qwen38-flash-next-radixark-vllm"
MODEL="/models/nvfp4/RadixArk-Qwen3.8-Flash-Next-NVFP4"
PATCH_DIR="/models/patches/qwen38flashnext-sm120"
PLE_PATCH_TARGET="/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"

PLE_PATCH="${PLE_PATCH:-minimal}"  # minimal (PR #87 gate patch, default) | x00byte (full-file fallback)
case "$PLE_PATCH" in
  minimal) PLE_PATCH_FILE="$PATCH_DIR/ple_layer_minimal.py" ;;
  x00byte) PLE_PATCH_FILE="$PATCH_DIR/ple_layer.py" ;;
  *) echo "ERROR: PLE_PATCH must be 'minimal' or 'x00byte'" >&2; exit 1 ;;
esac
[ -f "$PLE_PATCH_FILE" ] || { echo "ERROR: missing $PLE_PATCH_FILE (see PROVENANCE.md there)" >&2; exit 1; }
PORT="8000"                                   # only one model served at a time on this box
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/qwen38-flash-next-radixark-vllm"  # warm JIT/autotune cache
EAGER="${EAGER:-0}"              # 0 = use COMPILATION_CONFIG (fast); 1 = --enforce-eager (debug)
PLE_OFFLOAD="${PLE_OFFLOAD:-0}"  # 1 = VLLM_PLE_CPU_OFFLOAD=1 (FP8 table -> host RAM, ~48 GB)
SPEC="${SPEC:-1}"                # 1 = MTP speculative decoding (needs MOE_BACKEND=flashinfer_cutlass)
SPEC_TOKENS="${SPEC_TOKENS:-3}"  # draft tokens per step (mean acceptance length measured: 2.54)
# graphs-without-inductor by default (see MEASURED in header); "off" = vLLM default compile.
# NB: do NOT write this as ${COMPILATION_CONFIG:-{...}} - bash brace matching
# inside that expansion mis-parses and appends a stray '}' when the var IS set.
if [ -z "${COMPILATION_CONFIG:-}" ]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi
# marlin is NVFP4-only; the MTP draft head's MoE is unquantized BF16 -> cutlass when SPEC=1.
# MOE_BACKEND=auto leaves --moe-backend unpinned (vLLM picks per layer; proven
# to work with MTP by rtx6kpro PR #87 - candidate to benchmark vs cutlass).
if [ -z "${MOE_BACKEND:-}" ]; then
  if [ "$SPEC" = "1" ]; then MOE_BACKEND="flashinfer_cutlass"; else MOE_BACKEND="marlin"; fi
fi

if [ "$EAGER" = "0" ] && [ "$PLE_OFFLOAD" = "0" ] && { [ -z "$COMPILATION_CONFIG" ] || [ "$COMPILATION_CONFIG" = "off" ]; }; then
  echo "ERROR: EAGER=0 + PLE_OFFLOAD=0 with the default inductor compile cannot start on this box:" >&2
  echo "       inductor autotune clones the full 47.69 GiB PLE table during profiling" >&2
  echo "       on top of ~64 GiB/GPU of resident weights -> CUDA OOM." >&2
  echo "       Keep the default graphs-only COMPILATION_CONFIG, or use PLE_OFFLOAD=1, or EAGER=1." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
# Make sure we have the dedicate image (no-op if already present).
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
  # patched ple_layer.py ($PLE_PATCH): select the FP8 PLE path for this
  # hybrid checkpoint (stock build rejects it / silently upcasts it to BF16).
  -v "$PLE_PATCH_FILE":"$PLE_PATCH_TARGET":ro
)

if [ "$PLE_OFFLOAD" = "1" ]; then
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if [ "$avail_gb" -lt 50 ]; then
    echo "ERROR: PLE_OFFLOAD=1 wants >= ~50 GB free host RAM (FP8 table), only ${avail_gb} GB available." >&2
    exit 1
  fi
  DOCKER_ARGS+=(
    --cap-add=SYS_PTRACE           # pidfd_getfd for CUDA IPC (PLE offload worker)
    -e VLLM_PLE_CPU_OFFLOAD=1      # FP8 PLE table in pinned host RAM
    # rtx6kpro PR #87: pin container RAM so the ~48 GB offloaded table can
    # never be swapped out (memswap == mem => no swap for the container).
    --memory 62g --memory-swap 62g
  )
fi

# ---- vLLM server arguments ------------------------------------------------
# ENTRYPOINT is already ["vllm", "serve"], so we pass args directly.
VLLM_ARGS=(
  "$MODEL"

  --served-model-name qwen3.8-flash-next
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"

  --quantization modelopt_fp4   # also auto-detected; pinned (NVFP4 experts, PLE stays FP8)
  --tensor-parallel-size 2      # 2x RTX PRO 6000 Blackwell
)

[ "$EAGER" = "1" ] && VLLM_ARGS+=(--enforce-eager)   # debug only, ~5x slower (see MEASURED)
# compilation config (default: graphs without inductor; skipped in eager / when "off"):
if [ "$EAGER" != "1" ] && [ -n "$COMPILATION_CONFIG" ] && [ "$COMPILATION_CONFIG" != "off" ]; then
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi
# MTP speculative decoding (built-in MTP head):
[ "$SPEC" = "1" ] && VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
# MoE backend (default: flashinfer_cutlass when SPEC=1 since the MTP head MoE is
# unquantized BF16 and marlin is NVFP4-only, else marlin; "auto" = leave unpinned):
[ "$MOE_BACKEND" != "auto" ] && VLLM_ARGS+=(--moe-backend "$MOE_BACKEND")

VLLM_ARGS+=(
  --gpu-memory-utilization 0.95 # rtx_pro_6000 profile
  --max-num-seqs 16             # rtx_pro_6000 profile (recipe base default is 256)
  --max-num-batched-tokens 8192 # rtx_pro_6000 profile
  --enable-prefix-caching
  --no-enable-flashinfer-autotune

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
  # -- TORCH_CUDA_ARCH_LIST=12.0f : some RadixArk examples set this for sm_120
  #   JIT builds; the dedicated image already targets Blackwell, so only add it
  #   if you see arch-related JIT errors.
  # -- --enforce-eager : the dual-DGX-Spark recipe needs this (torch.compile
  #   deadlocks GB10 unified memory). x86 + 2x RTX PRO 6000 should be fine with
  #   graphs; add it back if startup hangs in cudagraph capture.
)

# ---- Launch ---------------------------------------------------------------
docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"

echo "Started '$NAME' on port $PORT. Follow startup with:  docker logs -f $NAME"
echo "Verify with: curl -s http://localhost:$PORT/v1/models"
