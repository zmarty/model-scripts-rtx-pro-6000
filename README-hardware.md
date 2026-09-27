# Hardware

These scripts serve LLMs on a headless workstation (`zmarty-aorus`, Ubuntu 24.04 LTS).
Everything is sized around **192 GB of combined GPU VRAM** — hence the mix of
FP8 / INT4 / NVFP4 checkpoints under `/models`.

## Summary

| Component | Detail |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each (97887 MiB) |
| GPU power limit | 600 W each |
| CPU | AMD Ryzen 9 7950X3D (16c/32t) |
| RAM | 192 GB DDR5 |
| Motherboard | ASUS ROG Crosshair X670E Hero |
| OS | Ubuntu 24.04 LTS, kernel 6.11.0-29-generic |
| NVIDIA driver | 615.71.09 **open kernel module**, CUDA 13.1, PCIe 5.0 x16 |
| Storage | Samsung 9100 PRO 8 TB NVMe (`/models`, weights) + WD_BLACK SN850X 4 TB |

## NVIDIA driver & software stack

- **Open kernel module, not proprietary.** The driver is the NVIDIA UNIX
  **Open Kernel Module** 615.71.09 (`Dual MIT/GPL`, per
  `/proc/driver/nvidia/version`). Blackwell (GB202) GPUs are only supported
  by the open kernel modules, so this isn't optional on this box.
- Installed from the NVIDIA CUDA repo (`cuda-drivers 615.71.09-2ubuntu1`,
  `cuda-keyring`), together with the **CUDA 13.1** toolkit
  (`/usr/local/cuda-13.1`).
- Most serving runs use **Docker** with vLLM images (the DeepSeek V4 Flash
  and Qwen3.8 Flash-Next recipes pull dedicated vLLM/nightly images), so the
  host toolkit mostly matters for quantization builds (`fp8/`, `int4/`,
  `nvfp4/`) and eval tooling.

## Firmware / kernel notes (learned the hard way)

Two incidents shaped the current configuration, both documented on the blog:

1. **TP=2 hangs** — two months of vLLM tensor-parallel-2 runs hanging with
   `No available shared memory broadcast block found in 60 seconds`. The fix
   was kernel args in `/etc/default/grub`:

   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash md_iommu=on iommu=pt amdgpu.dcdebugmask=0x10"
   ```

   (live in `/proc/cmdline` today). Note `md_iommu=on` is carried over from
   the blog post and is likely a typo for `amd_iommu=on`; the effective part
   is `iommu=pt` (IOMMU passthrough mode), which restored stable GPU-to-GPU
   P2P communication on the ROG Crosshair X670E Hero.

2. **Hard power-offs under sustained load** — the default Blackwell fan
   curve ramps too slowly at 600 W sustained draw. Fixed with an NVML-based
   fan-control daemon + systemd service, plus custom case-fan wiring/pwm
   control. See `README-case-fan-gpu-cooling.md` and the blog posts below.

## Related blog posts (zmarty.github.io)

- 2025-12-25 — *Dual RTX PRO 6000 LLM guide* (TP=2 IOMMU fix, model recipes)
- 2026-01-17 — *Fixing RTX Pro 6000 Blackwell shutdowns with custom fan control*
- 2026-06-20 — *DeepSeek V4 Flash on RTX PRO 6000 (vLLM fork)*
- 2026-08-27 — *Qwen3.8 Flash-Next NVFP4 recipe (n-gram lookup offload)*

## Notes

- **Both GPUs are identical 96 GB cards.** Tensor-parallel (TP=2) serving
  gives ~192 GB VRAM; single-GPU runs are also possible for models ≤ 96 GB.
- The RTX PRO 6000 Blackwell supports **NVFP4** natively, which is why
  `/models/nvfp4/` checkpoints (GLM 5.3 Flash Spark, RadixArk Qwen3.8) run
  at high throughput on this box.
- The 7950X3D has a single NUMA node; host offload (e.g. large KV pools
  needing ~48 GB host RAM) is straightforward.
- `/models` lives on the 8 TB Samsung drive and also holds the quantized
  checkpoint builds (`fp8/`, `int4/`, `nvfp4/`) converted from `original/`.
- Cooling is custom-managed, not stock: see
  [README-case-fan-gpu-cooling.md](README-case-fan-gpu-cooling.md) for the
  case-fan rewiring and the `nvidia-fan-control` setup for the GPUs.
  Thermal/benchmark tooling used to validate it lives in `eval/`.
