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
| OS | Ubuntu 24.04 LTS |
| NVIDIA driver | 615.71.09, PCIe 5.0 x16 |
| Storage | Samsung 9100 PRO 8 TB NVMe (`/models`, weights) + WD_BLACK SN850X 4 TB |

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
