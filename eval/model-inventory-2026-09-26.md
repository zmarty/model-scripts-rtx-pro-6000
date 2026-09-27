# /models inventory vs. Artificial Analysis — 2026-09-26

Volume: `/dev/nvme0n1p1` 7.3 TB, **5.68 TB used, 1.4 TB free (81%)**.
Scores = **Artificial Analysis Intelligence Index v4.3.2** (live EN model pages, fetched 2026-09-26).
Image models = AA Image Arena Quality **Elo** (AA does not put image models on the Intelligence Index).

> ⚠️ Index-version warning: AA rebased its index (v4.0 → v4.3.2, ~7 Sep 2026). Launch-week press numbers
> (DSv4-Flash 47, 0731 50, MiniMax-M2.7 50, GLM-5 50, gpt-oss-120b 58, Flash-Next 55.8) are **not** current
> scores. All numbers below are the live page values.

## 1. Full inventory

| Path | GB | Base model | AA index | Released (AA) | AA status |
|---|---|---|---|---|---|
| `fp8/Qwen-Qwen3.8-Flash-Next-FP8` | 173 | Qwen3.8-Flash-Next | **40** #6/115 OW | 2026-08-26 | ✅ current (serving now) |
| `nvfp4/RadixArk-Qwen3.8-Flash-Next-NVFP4` | 126 | " | 40 | 2026-08-26 | ✅ current, eval-verified ≈ FP8 |
| `nvfp4/Inferact-Qwen3.8-Flash-Next-NVFP4` | 171 | " | 40 | 2026-08-26 | ⚠️ current model, unusable build here (BF16 PLE ⇒ 93 GB host RAM, eval BLOCKED) |
| `DeepSeek-V4-Flash-0731/` | 156 | DeepSeek V4 Flash 0731 | **34** #10/115 | 2026-07-31 | ❌ deprecated → V4.1 Flash (39) |
| `original/DeepSeek-V4-Flash` (FP8) | 149 | DeepSeek V4 Flash 0420 | **24** #32/115 | 2026-04-24 | ❌ deprecated → 0731 |
| `original/DeepSeek-V4-Flash-DSpark` | 156 | DSpark drafter (wraps V4 Flash) | not on AA | ~2026-06-26 | ➖ tooling, no AA score |
| `original/DeepSeek-V4-Flash-Vision-Exp` | 157 | DeepSeek V4 Flash Vision | **35** #7/174 | 2026-08-21 | ✅ current, only vision DS |
| `int4/LordNeel-…-W4A16-FP8` | 146 | DSv4-Flash W4A16+MTP | not on AA | 2026-06 | 3rd quant of same base |
| `original/deepseek-ai-DeepSeek-OCR` | 7 | DeepSeek-OCR | not on AA | 2025-10-20 | ➖ superseded by OCR2 (2026-01-27) |
| `nvfp4/lukealonso-MiniMax-M2-NVFP4` | 122 | MiniMax-M2 | **19** #55/115 | 2025-10 | ❌ deprecated |
| `awq/cyankiwi-MiniMax-M2-AWQ-4bit` | 122 | " | 19 | 2025-10 | ❌ deprecated |
| `awq/QuantTrio-MiniMax-M2-AWQ` | 113 | " | 19 | 2025-10 | ❌ deprecated |
| `gguf/unsloth/MiniMax-M2-GGUF` (Q5_K_XL) | 151 | " | 19 | 2025-10 | ❌ deprecated |
| `exl3/turboderp-MiniMax-M2-exl3-3.04bpw` | 83 | " | 19 | 2025-10 | ❌ deprecated |
| `awq/cyankiwi-MiniMax-M2.1-AWQ-4bit` | 122 | MiniMax-M2.1 | **21** #47/115 | 2025-12 | ❌ deprecated |
| `awq/QuantTrio-MiniMax-M2.1-AWQ` | 117 | " | 21 | 2025-12 | ❌ deprecated |
| `awq/mratsim-MiniMax-M2.1-FP8-INT4-AWQ` | 123 | " | 21 | 2025-12 | ❌ deprecated |
| `nvfp4/lukealonso-MiniMax-M2.1-NVFP4` | 122 | " | 21 | 2025-12 | ❌ deprecated |
| `awq/cyankiwi-MiniMax-M2.7-AWQ-4bit` | 122 | MiniMax-M2.7 | **23** #37/115 | 2026-03 | ❌ deprecated → M3 (29) |
| `awq/QuantTrio-MiniMax-M2.7-AWQ` | 122 | " | 23 | 2026-03 | ❌ deprecated |
| `nvfp4/lukealonso-MiniMax-M2.7-NVFP4` | 136 | " | 23 | 2026-03 | ❌ deprecated |
| `nvfp4/NinjaBoffin-MiniMax-M2.7-NVFP4` | 131 | " | 23 | 2026-03 | ❌ deprecated |
| `gguf/ubergarm/Kimi-K2-Instruct-GGUF` | 22 | Kimi K2 | **13** (est.) | 2025-07-11 | ❌ deprecated (128k, slow) |
| `awq/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix` | 69 | GLM-4.5-Air | **11** #15/65 | 2025-07-28 | ❌ deprecated → GLM-4.7-Flash |
| `gguf/Unsloth/GLM-4.5-Air-GGUF/UD-Q8_K_XL` | 119 | " | 11 | 2025-07-28 | ❌ deprecated |
| `original/GLM-4.5-Air-FP8` | 105 | " | 11 | 2025-07-28 | ❌ deprecated |
| `gguf/unsloth/GLM-4.6-GGUF` (Q3_K_XL) | 148 | GLM-4.6 | **19** R / 15 NR | 2025-09-30 | ❌ deprecated → GLM-4.7 |
| `awq/cyankiwi-GLM-4.6V-AWQ-8bit` | 110 | GLM-4.6V | **11** R / 8 NR | 2025-12-08 | ❌ deprecated → GLM-5V-Turbo |
| `gguf/unsloth/GLM-4.6V-GGUF` (Q8_K_XL) | 119 | " | 11 / 8 | 2025-12-08 | ❌ deprecated |
| `original/GLM-4.6V-FP8` | 103 | " | 11 / 8 | 2025-12-08 | ❌ deprecated |
| `gguf/unsloth/GLM-4.7-GGUF` (Q3_K_S) | 145 | GLM-4.7 | **22** R / 17 NR | 2025-12-22 | ❌ deprecated → GLM-5 → 5.3 (45) |
| `original/GLM-4.7-Flash` | 59 | GLM-4.7-Flash | **15** R / 11 NR | 2026-01-19 | ❌ deprecated (but 3B-active, $0.06, 118 t/s NR) |
| `awq/cyankiwi-INTELLECT-3-AWQ-8bit` | 88 | INTELLECT-3 (GLM-4.5-Air post-train) | **11** | 2025-11-27 | ✅ no banner, but no AA speed/cost data |
| `awq/QuantTrio-Qwen3-235B-A22B-Thinking-2507-AWQ` | 116 | Qwen3-235B-A22B-2507 | **13** #81/115 | 2025-07-25 | ❌ deprecated → Qwen3.5 397B |
| `nvfp4/NVFP4-…-Thinking-2507-FP4` | 125 | " | 13 | 2025-07-25 | ❌ deprecated |
| `nvfp4/nvidia/Qwen3-235B-A22B-NVFP4` | 125 | " | 13 | 2025-07-25 | ❌ deprecated |
| `nvfp4/RedHatAI-Qwen3-235B-A22B-NVFP4` | 134 | " | 13 | 2025-07-25 | ❌ deprecated |
| `gptq/Qwen-Qwen3-235B-A22B-GPTQ-Int4` | 117 | Qwen3-235B-A22B (Apr-2025) | ≤13 | 2025-04 | ❌ deprecated |
| `awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ` | 117 | Qwen3-VL-235B (R) | **13** #76/115 | 2025-09-23 | ❌ deprecated |
| `gguf/unsloth/Qwen3-VL-235B-A22B-Thinking-GGUF` | 158 | " | 13 | 2025-09-23 | ❌ deprecated |
| `original/Qwen-Qwen3-32B` | 62 | Qwen3-32B | **9** R / 7 NR, 33k ctx | 2025-04-28 | ❌ deprecated |
| `original/RedHatAI-Qwen3-32B-speculator.eagle3` | 3 | drafter for Qwen3-32B | not on AA | 2025-09-17 | ❌ target deprecated |
| `original/Qwen3.5-122B-A10B-FP8` | 119 | Qwen3.5-122B-A10B | 16 R / **18** NR | 2026-02-24 | ✅ no banner, dominated locally |
| `original/Qwen-Qwen3.6-27B` | 52 | Qwen3.6-27B | **21** R / 20 NR | 2026-04-22 | ❌ deprecated → Qwen3.8 27B |
| `original/Qwen-Qwen-Image` | 54 | Qwen-Image | Elo **887** #102 T2I | 2025-08-04 | ⚠️ beaten by Qwen-Image-2.1 (Elo 1035) |
| `original/Qwen-Qwen-Image-Edit-2509` | 54 | Qwen Image Edit Plus 2509 | Elo **980** #51 edit | 2025-09-25 | ⚠️ beaten by 2511 (1023) / 2.1 (1071) |
| `awq/cyankiwi-Devstral-2-123B-…-AWQ-4bit` | 140 | Devstral 2 123B | **9** #7/39 | 2025-12-09 | ❌ deprecated → Mistral Medium 3.5 |
| `original/Devstral-Small-2-24B-Instruct-2512` | 49 | Devstral Small 2 | **8** #18/75 | 2025-12-09 | ❌ deprecated → Mistral Medium 3.5 |
| `original/openai-gpt-oss-120b` | 183 | gpt-oss-120b (high) | **12** #9/65, 192 t/s #3 | 2025-08-05 | ✅ not deprecated (fast tier) |
| `original/nvidia-gpt-oss-120b-Eagle3` + `-v2` | 3 | Eagle3 drafters | not on AA | 2025-08-20 / 2025-10-06 | ➖ v3 (2026-05-08) is current |
| `original/Snowflake-Arctic-LSTM-Speculator-gpt-oss-120b` | 7 | LSTM drafter | not on AA | 2025-08-21 | ➖ needs Arctic-Inference plugin |
| `original/nvidia-Llama-3.3-70B-Instruct-FP4` | 40 | Llama 3.3 70B Instruct | **8** #12/39 | 2024-12 | ➖ no banner, oldest+lowest score here |
| `original/google-gemma-4-31B-it` (+`-assistant`) | 60 | Gemma 4 31B | **19** R / 14 NR, 35 t/s | 2026-04 | ✅ current, text+image+video |
| `original/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 59 | Nemotron 3 Nano 30B A3B | **9** R, 197 t/s | 2025-12-15 | ✅ current, $0.02/index-task |
| `embeddings/google-siglip2-base-patch16-224` | 2 | SigLIP2 vision encoder | not on AA | 2025-02 | ➖ component, do not delete blindly |
| `embeddings/jinaai-jina-reranker-m0` | 5 | jina-reranker-m0 | not on AA | 2025-04-08 | ➖ CC-BY-NC-4.0 (non-commercial) |
| empty dirs: `gguf/lmstudio-community/MiniMax-M2-GGUF`, `gguf/noctrex/MiniMax-M2-MXFP4_MOE-GGUF`, `gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF` | 0 | — | — | — | 🧹 leftovers |

## 2. Current open-weights leaders on AA (for the "what to pull next" decision)

| Model | Index | Notes |
|---|---|---|
| Xiaomi MiMo-V2.6-Pro | **46** | AA: top open-weights model |
| GLM-5.3 (max) | **45** | best agentic-coding (Terminal-Bench 4.0 42%), 2026-08-18 |
| Kimi K3 (max) | **44** | 2.8T/104B, expensive + slow per AA |
| GLM-5.3-Flash | 42 | small-tier leader |
| **Qwen3.8-Flash-Next (yours)** | **40** | you already hold the #4 open-weights model |
| DeepSeek V4.1 Flash | 39 | you hold 0731 (34) and 0420 (24) |
| MiniMax-M3 | 29 | you hold only deprecated M2.x |

## 3. Cleanup plan (see `cleanup_models.sh`)

- **Tier 1 — delete now: ~4.0 TB.** AA-deprecated bases where a newer/better local copy exists, plus
  4–7 duplicate quantizations of the same deprecated base, plus the Inferact build that cannot run here.
- **Tier 2 — decide: ~0.9 TB.** April DSv4-Flash + DSpark + int4 MTP, gpt-oss-120b + drafters,
  Qwen3.5-122B, Qwen-Image pair, DeepSeek-OCR, jina-reranker.
- **Never touch:** `fp8/Qwen-Qwen3.8-Flash-Next-FP8` (live container), `nvfp4/RadixArk-…`,
  `DeepSeek-V4-Flash-0731`, `original/DeepSeek-V4-Flash-Vision-Exp`, `google-gemma-4-31B-it`,
  `embeddings/google-siglip2-*`, `patches/`, `eval/`.
