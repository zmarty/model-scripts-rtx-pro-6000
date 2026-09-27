#!/usr/bin/env bash
# Start DeepSeek-V4-Flash on 2x RTX PRO 6000 (TP2) via the Lucifer vLLM image.
# Runs detached in the background; you start it manually (no auto-restart).
# Low-concurrency / interactive-coding tuned. Re-run this script to (re)start.
set -euo pipefail

IMAGE="voipmonitor/vllm:lucifer"
NAME="ds4-lucifer-tp2"
MODEL="/models/original/DeepSeek-V4-Flash"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/lucifer-vllm7c6bbf4"   # warm JIT/autotune cache; keep this
PORT="8000"

mkdir -p "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"                 # container name (used to stop / inspect / log)
  --init                         # real init as PID 1 for clean signal handling
  --restart no                   # never auto-restart; you launch this manually
  --gpus all                     # expose GPUs to the container...
  --runtime nvidia               # ...via the NVIDIA container runtime
  --ipc host                     # share host IPC namespace (large shared-mem tensors)
  --shm-size 32g                 # /dev/shm size for NCCL and worker comms
  --network host                 # host networking; PORT is a host port
  --ulimit memlock=-1            # unlimited locked memory (pinned / RDMA buffers)
  --ulimit stack=67108864       # 64 MiB thread stack
  -v /models:/models             # mount the model tree into the container
  -v "$CACHE_DIR":/cache         # persist JIT / autotune cache across runs
  -e CUDA_VISIBLE_DEVICES=0,1    # use the first two GPUs (TP2)
  -e CUTE_DSL_ARCH=sm_120a       # target Blackwell sm_120a kernels
  -e HF_HUB_OFFLINE=1            # never contact Hugging Face; local weights only
  -e NCCL_P2P_LEVEL=SYS          # permit GPU-to-GPU P2P across the PCIe system
  -e NCCL_PROTO=LL,LL128,Simple  # NCCL protocols allowed
  -e NCCL_IB_DISABLE=1           # no InfiniBand on this host
)

# ---- vLLM server arguments ------------------------------------------------
VLLM_ARGS=(
  "$MODEL"                                    # positional: path to the model to serve

  --served-model-name DeepSeek-V4-Flash       # name clients pass in the OpenAI "model" field
  --trust-remote-code                         # load the model's bundled custom code
  --host 0.0.0.0                              # listen on all interfaces
  --port "$PORT"                              # server port
  --load-format auto                          # auto-detect the weight format

  --tensor-parallel-size 2                    # shard the model across 2 GPUs (TP2)
  --kv-cache-dtype fp8                         # fp8 KV cache -> fits much more context
  --block-size 256                            # tokens per KV cache page / block
  --gpu-memory-utilization 0.90               # fraction of VRAM vLLM may claim

  --max-model-len 262144                      # max context length per request (256K)
  --max-num-seqs 8                            # max concurrent sequences (low-concurrency use)
  --max-num-batched-tokens 8192               # max tokens per scheduler step (caps prefill chunk)
  --max-cudagraph-capture-size 192            # largest batch captured as a CUDA graph

  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'  # torch.compile / cudagraph modes
  --async-scheduling                          # overlap scheduling with GPU execution
  --no-scheduler-reserve-full-isl             # don't pre-reserve full input length (packs KV tighter)
  --enable-chunked-prefill                    # split long prefills into token chunks
  --enable-prefix-caching                     # reuse KV for shared prompt prefixes
  --enable-flashinfer-autotune                # autotune FlashInfer kernels at startup

  --attention-backend SPARSE_MLA_SM120        # sparse MLA attention kernel for sm_120
  --kernel-config.moe_backend flashinfer_cutlass  # MoE expert GEMM backend

  --tokenizer-mode deepseek_v4                # DeepSeek-V4 tokenizer handling
  --reasoning-parser deepseek_v4              # parse <think> reasoning output
  --tool-call-parser deepseek_v4              # parse tool / function calls
  --enable-auto-tool-choice                   # let the model decide when to call tools

  --default-chat-template-kwargs.thinking=true          # enable thinking / reasoning by default
  --default-chat-template-kwargs.reasoning_effort=high  # default reasoning effort

  --speculative-config.method mtp                         # speculative decoding: MTP heads
  --speculative-config.num_speculative_tokens 2           # draft 2 tokens per step
  --speculative-config.draft_sample_method probabilistic  # sample (not greedy) the draft tokens
)

docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  /bin/bash -lc '
    # The image ships PCIe / fused-all-reduce tunables that hurt this 2-GPU box.
    # Clear them so we fall back to the plain NCCL path, then launch the server.
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE \
          VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND \
          VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS \
          VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER \
          VLLM_CACHE_DIR
    exec vllm serve "$@"
  ' bash "${VLLM_ARGS[@]}"

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
echo "First launch after an image change warms the cache (~5 min); reuses $CACHE_DIR otherwise."
