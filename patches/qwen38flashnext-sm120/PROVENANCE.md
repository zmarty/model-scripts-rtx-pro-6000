# Patches for Qwen3.8-Flash-Next on SM120 (RTX PRO 6000 Blackwell)

Bind-mounted into the `vllm/vllm-openai:qwen38-flash-next` container at runtime
(read-only) by the `start_qwen38_flash_next_*.sh` scripts.

## ple_layer_minimal.py (DEFAULT, mounted by `PLE_PATCH=minimal`)

- **Source:** the image's **stock** `ple_layer.py` (extracted from
  `vllm/vllm-openai:qwen38-flash-next`, build v0.1.dev20073+g8e685d198)
  plus the **11-line gate patch** from
  [local-inference-lab/rtx6kpro PR #87](https://github.com/local-inference-lab/rtx6kpro/pull/87)
  (`patches/qwen38-flash-next-nvfp4-ple-fp8.patch`, Apache-2.0),
  validated by that PR on the same image build and hardware class.
- **Container path:** same as below.
- **The patch (1 logical change, ~11 lines):**
  `_get_ple_embedding_quant_method()` takes the text config and returns
  `Qwen3_8FlashNextPLEFp8EmbeddingMethod()` when
  `config.ple_embedding_dtype == "float8_e4m3fn"`, regardless of the outer
  quant config (`modelopt` here). The call site passes `config` through.
- **Why this is sufficient:** once the FP8 method is selected, the **stock**
  code does everything else — `create_weights` registers `weight_scale` as a
  `PerTensorScaleParameter`, `AutoWeightsLoader` loads it (non-offload) or the
  stock offload path registers the `_offload_weight_scale` buffer
  (`VLLM_PLE_CPU_OFFLOAD=1`), and stock `_dequantize_embeddings` /
  `_get_embedding_weight_scale` apply the scale at lookup in both modes.
  Verified against the extracted stock file 2026-08-28.
- **Advantage over the x00byte file:** a minimal diff against the image's own
  stock file, so it is trivial to rebase/review when the image updates.

## ple_layer.py (x00byte full-file replacement, fallback via `PLE_PATCH=x00byte`)

- **Source:** https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe (Apache-2.0),
  itself a patch on top of vLLM `vllm/vllm-openai:qwen38-flash-next`
  (build v0.1.dev20073+g8e685d198, Apache-2.0).
- **Container path:**
  `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`
- **Why:** the RadixArk NVFP4 checkpoint is hybrid — ModelOpt NVFP4 routed
  experts + an FP8 E4M3 PLE n-gram table with a single global
  `ngram_embedding.weight_scale`. Stock vLLM only enables its FP8 PLE path when
  the outer quant config is `Fp8Config`; here it is `modelopt`, so loading the
  RadixArk checkpoint dies with
  `ValueError: no module or parameter named 'ngram_embedding.weight_scale'`
  (and on builds where it doesn't crash, the table is silently upcast to BF16
  with no scale applied = wrong embeddings, and double the VRAM).
- **The patch (3 changes):**
  1. `_get_ple_embedding_quant_method()` also selects the FP8 path when the
     text config declares `ple_embedding_dtype == "float8_e4m3fn"`
     (the RadixArk config does).
  2. `weight_scale` is no longer registered as a parameter in
     `create_weights` (it collided with the loader's buffer:
     "attribute 'weight_scale' already exists").
  3. The loader registers the global `ngram_embedding.weight_scale` as a
     buffer; `forward()` dequantizes lookups with it.
- **Note (2026-08-28):** changes 2–3 were only necessitated by x00byte's own
  offload-worker changes; on this image build the stock parameter-based path
  works once the gate fires. Kept as a known-good fallback (it served our
  ~138 tok/s default config).

## qwen_sparse_attn_backend.py

(pre-existing patch, not related to the PLE FP8 fix)
