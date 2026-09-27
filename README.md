# model-scripts-rtx-pro-6000

Ops repository for a headless LLM inference workstation: **2× NVIDIA RTX PRO
6000 Blackwell (96 GB each, 192 GB total VRAM)**, running models with vLLM in
Docker. All scripts live at the root of `/models` on the box, next to the
weights they serve.

> ⚠️ **This repo is public.** See [AGENTS.md](AGENTS.md) for the policy:
> never commit secrets, model weights, logs, or venvs.

## Hardware

| | |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell Workstation, 96 GB, 600 W each |
| CPU / RAM | AMD Ryzen 9 7950X3D / 192 GB DDR5 |
| Board / OS | ASUS ROG Crosshair X670E Hero, Ubuntu 24.04 (open NVIDIA driver 615.71.09, CUDA 13.1) |
| Storage | 8 TB NVMe (`/models`) + 4 TB NVMe |

Full details — driver stack, the IOMMU kernel-args fix for TP=2, and the fan
control work that stopped hard power-offs under load — are in
[README-hardware.md](README-hardware.md) and
[README-case-fan-gpu-cooling.md](README-case-fan-gpu-cooling.md).

## Repo layout

Tracked:

```
start_*.sh / stop_*.sh   serving launch/teardown scripts (one pair per model)
README-*.md              serving guides and hardware notes
eval/                    benchmark + thermal tooling (lm-eval-harness wrappers, NVML monitors)
patches/                 vLLM backend patches (e.g. qwen38flashnext-sm120 sparse attention)
```

Ignored (huge, machine-local — re-downloadable or regenerable):

```
original/                unquantized checkpoints (~1.1 TB)
fp8/ int4/ nvfp4/        quantized builds made on this box
DeepSeek-V4-Flash-0731/  a single large checkpoint
embeddings/ hf-cache/    auxiliary models + Hugging Face cache
.venv/ eval/venv/        Python environments
*.log eval/results/      run logs and raw results
```

## Serving scripts

Every model runs as a Docker container (vLLM image) serving an
OpenAI-compatible API on **port 8000**, tensor-parallel across both GPUs
(TP=2). Pattern: `sudo ./start_<model>.sh` then `sudo ./stop_<model>.sh`.

| Script pair | Model | Format / notes |
|---|---|---|
| `ds4_flash` | DeepSeek-V4-Flash | BF16 original, vLLM fork image `lucifer` |
| `ds4_flash_0731` | DeepSeek-V4-Flash 0731 snapshot | DSpark speculative decoding variant |
| `ds4_flash_v9_mtp` | DeepSeek-V4-Flash | MTP (multi-token prediction) spec decoding |
| `ds4_flash_vision` | DeepSeek-V4-Flash-Vision-Exp | multimodal checkpoint |
| `glm53_flash_spark` | GLM-5.3-Flash-NVFP4-Spark | NVFP4, pulled from HF cache |
| `gemma4_31b` | google-gemma-4-31B-it | + EAGLE-style draft model for spec decoding |
| `qwen38_flash_next_fp8_vllm` | Qwen3.8-Flash-Next | FP8 official build |
| `qwen38_flash_next_radixark_vllm` | Qwen3.8-Flash-Next | NVFP4 (RadixArk) build |

Check what's serving:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
```

Deeper docs: [README-DS4-Flash-serving.md](README-DS4-Flash-serving.md)
(DeepSeek V4 Flash specifics, token limits, DSpark/MTP quirks) and
[README-Qwen3.8-Flash-Next-vLLM.md](README-Qwen3.8-Flash-Next-vLLM.md)
(Qwen3.8 benchmark results and tuning).

## Eval tooling (`eval/`)

- `run_suite.sh` / `run_gen_only.sh` / `run_thermal_suite.sh` —
  lm-evaluation-harness against the local vLLM server, with optional
  thermal monitoring.
- `thermal_monitor.py` / `monitor_temps.sh` / `gemm_burn.py` — NVML-based
  temperature/power logging and GPU burn tests (built while fixing the
  shutdown-under-load problem).
- `model-inventory-2026-09-26.md`, `SUMMARY.md` — what's on disk and
  quantization-variant quality results (FP8 vs NVFP4).

## Patches (`patches/`)

`qwen38flashnext-sm120/` — vLLM backend patches for Qwen3.8 Flash-Next on
Blackwell (sparse attention backend + PLE layer), with `PROVENANCE.md`
documenting origin.

## Related blog posts

- [Dual RTX PRO 6000 LLM guide](https://www.ovidiudan.com/2025/12/25/dual-rtx-pro-6000-llm-guide.html) — TP=2 IOMMU fix, model recipes
- [Fixing RTX Pro 6000 Blackwell shutdowns with custom fan control](https://www.ovidiudan.com/2026/01/17/nvidia-rtx-pro-6000-blackwell-fan-control.html)
- [DeepSeek V4 Flash on RTX PRO 6000](https://www.ovidiudan.com/2026/06/20/deepseek-v4-flash-rtx-pro-6000-vllm.html)
- [Qwen3.8 Flash-Next NVFP4 recipe](https://www.ovidiudan.com/2026/08/27/recipe-qwen38-flash-next-nvfp4-rtx-pro-6000.html)
