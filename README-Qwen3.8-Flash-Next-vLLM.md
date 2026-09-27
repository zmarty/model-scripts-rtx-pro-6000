# Serving Qwen3.8-Flash-Next (NVFP4 + FP8) with vLLM on 2× RTX PRO 6000

(Formerly `README-Qwen3.8-Flash-Next-RadixArk-vLLM.md` — renamed when the FP8 flavor was added; §8.)

**Date:** 2026-08-28 (updated same day: minimal patch + new configs + FP8 flavor, see §5/§8)
**Box:** 2× RTX PRO 6000 Blackwell (SM120, 95.01 GiB usable / 97,887 MiB total each), x86
**Result:** ✅ Serving on port 8000 at **~157 tok/s decode** (single request) with the default config:
**entire model in VRAM (zero host RAM), CUDA graphs without inductor, MTP speculative decoding**.
KV cache 1,480,759 tokens (5.65× concurrency at 262,144 ctx). See §5 for the full benchmark table.

**2026-08-28 evening update:** quality benchmarks (lm-eval) of the FP8 vs RadixArk NVFP4
flavors show **no measurable quality difference** (§10) and both sit in the expected range of
the official tech-report numbers. **We settled on the FP8 flavor as the daily driver**
(fastest, no patch needed); RadixArk is the zero-host-RAM fallback. Inferact row pending a
reboot (orphaned host RAM blocks its CPU-offload table — see §10).

**2026-08-28 PM update:** the bind-mounted patch is now the **minimal 11-line gate patch**
(from [rtx6kpro PR #87](https://github.com/local-inference-lab/rtx6kpro/pull/87)) applied to
the image's own stock `ple_layer.py` (`PLE_PATCH=minimal`, default) instead of the x00byte
full-file replacement (`PLE_PATCH=x00byte`, fallback). Same KV pool, same correctness, and
**~14% faster** (138 → 157 tok/s): stock's parameter-based FP8 embedding path beats x00byte's
custom buffer/dequant path. Also new: `PLE_OFFLOAD=1` + full inductor + MTP + unpinned MoE
backend measures **~163 tok/s with a 3.07M-token KV pool**, and a sibling FP8 flavor
(official `Qwen/Qwen3.8-Flash-Next-FP8`) serves at **~168 tok/s** (§8).

---

## 1. The two NVFP4 flavors on disk, and why RadixArk is ~45 GB smaller

| Component | Inferact (`/models/nvfp4/Inferact-Qwen3.8-Flash-Next-NVFP4`) | RadixArk (`/models/nvfp4/RadixArk-Qwen3.8-Flash-Next-NVFP4`) |
|---|---|---|
| Total on disk | **171 GB** | **126 GB** |
| NVFP4 routed experts | 64 GB (`nvfp4_experts-*.safetensors`) | 64 GB (`layer-*-experts-*.safetensors`) — identical scope |
| BF16 attention / GDN / shared experts / etc. | ~15 GB (inside `model-*.safetensors`) | 15 GB (`model-bf16-*.safetensors`) — identical |
| **PLE n-gram table (51B params)** | **~91 GB, BF16** (rest of `model-*.safetensors`, 106 GB total) | **~48 GB, FP8 E4M3** (`model-plefp8-*.safetensors`) + one global `weight_scale` |
| MTP head | 1.5 GB (`nvfp4_experts_mtp.safetensors`) | included |

The **entire** size difference is the PLE (n-gram) embedding table dtype: BF16 vs FP8 E4M3,
verified directly in the safetensors headers:

```
Inferact:  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight  BF16
RadixArk:  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight  F8_E4M3
RadixArk:  ...ngram_embedding.weight_scale  (single global scale, in model-plefp8-00009)
```

Both checkpoints are otherwise the same architecture (`Qwen4ExpForConditionalGeneration`,
`model_type=qwen4_exp`) and same quant scheme (ModelOpt NVFP4 W4A4, group_size 16, routed
experts only; attention/linear_attn/shared experts/MTP/PLE/vision/embeddings excluded).

The RadixArk text config declares the table dtype explicitly — this becomes important later:

```json
"text_config": { "ple_embedding_dtype": "float8_e4m3fn", "ple_layer_ids": [2], ... }
```

**Implication:** RadixArk's GPU-resident weights ≈ 64 + 15 + 48 + 1.5 ≈ **128.5 GB**, which
fits in 2× 97.9 GB at 0.95 util (~186 GB) with ~57 GB to spare → full GPU residency is
plausible. The Inferact flavor can never do this on this box (its BF16 table alone is ~91 GB),
which is why its script requires `VLLM_PLE_CPU_OFFLOAD=1`.

---

## 2. Script layout / naming convention

Renamed everything to `qwen38_flash_next_<flavor>_<engine>` ("flash next", quant flavor included):

| File | Serves | Engine |
|---|---|---|
| `start_qwen38_flash_next_inferact_vllm.sh` / `stop_...` | Inferact NVFP4 (BF16 PLE, offload required) | vLLM |
| `start_qwen38_flash_next_radixark_vllm.sh` / `stop_...` | RadixArk NVFP4 (FP8 PLE, full GPU residency) | vLLM |
| `start_qwen38_flash_next_fp8_vllm.sh` / `stop_...` (added PM, §8) | Official FP8 (no patch, offload mandatory) | vLLM |

(The SGLang pair for the RadixArk checkpoint was deleted from `/models` during this work and
no longer exists.)

Container names and cache dirs match the script names (`qwen38-flash-next-<flavor>-vllm`,
`~/.cache/qwen38-flash-next-<flavor>-vllm`). Both serve on port 8000 — one model at a time.

---

## 3. `VLLM_PLE_FP8_CHECKPOINT` is NOT a real vLLM env var

An online example for serving this checkpoint used:

```bash
VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_FP8_CHECKPOINT=1 TORCH_CUDA_ARCH_LIST=12.0f ... vllm serve ...
```

Investigation:

- At container startup vLLM logged: `Unknown vLLM environment variable detected: VLLM_PLE_FP8_CHECKPOINT`.
- `envs.py` in the image (`vllm/vllm-openai:qwen38-flash-next`, build `v0.1.dev20073+g8e685d198`)
  knows **only** `VLLM_PLE_CPU_OFFLOAD` (from upstream PR
  [vllm-project/vllm#53899](https://github.com/vllm-project/vllm/pull/53899)).
- GitHub-wide search: **0 repos, 0 issues, 0 PRs, 0 discussions** mention `VLLM_PLE_FP8_CHECKPOINT`;
  exactly **1 commit** (someone's fork). It is not upstream and not in this image.

**Conclusion:** only certain patched vLLM builds recognize it. The widely-available fix for this
checkpoint is not an env var at all — it's a source patch (below).

---

## 4. Failure #1: stock vLLM cannot load the RadixArk FP8 PLE table

First boot (no patch) died during weight load:

```
ValueError: There is no module or parameter named 'ngram_embedding.weight_scale' in
Qwen3_8FlashNextNGramEmbedding. The available parameters belonging to ngram_embedding
(VocabParallelEmbedding) are: {'ngram_embedding.weight'}
```

Root cause (confirmed by the checkpoint layout and by the dual-Spark recipe README):

- The checkpoint is **hybrid**: ModelOpt NVFP4 routed experts, but the PLE table ships as
  FP8 shards with a single global `ngram_embedding.weight_scale`.
- Stock vLLM only enables its FP8 PLE path when the outer quant config is an `Fp8Config`.
  Here the outer config is `modelopt`, so the gate never fires.
- On builds where this doesn't crash outright, the failure mode is worse: the FP8 bytes are
  **silently upcast to BF16 with no scale applied** → wrong embeddings (garbage-ish output)
  and double the VRAM. (Consistent with what we saw pre-crash: 90.4 GB/GPU already used
  during the load, i.e. a ~180 GB footprint ≈ 128.5 + 48 GB of upcast table.)

### The fix: patched `ple_layer.py` — two variants, minimal is default

**Current default (`PLE_PATCH=minimal`):** the image's own stock `ple_layer.py` + the
**11-line gate patch** from [rtx6kpro PR #87](https://github.com/local-inference-lab/rtx6kpro/pull/87)
(Apache-2.0), installed at `/models/patches/qwen38flashnext-sm120/ple_layer_minimal.py`.
It only makes `_get_ple_embedding_quant_method()` select the FP8 PLE method when the text
config declares `ple_embedding_dtype == "float8_e4m3fn"` (the RadixArk config does),
regardless of the outer quant config being ModelOpt rather than `Fp8Config`.

That one gate is sufficient — verified against the stock file extracted from this exact
image: once the FP8 method is selected, **stock** `create_weights` registers `weight_scale`
as a `PerTensorScaleParameter`, the stock `AutoWeightsLoader` loads it (full residency) or
the stock offload path registers the `_offload_weight_scale` buffer (`VLLM_PLE_CPU_OFFLOAD=1`),
and stock `_dequantize_embeddings` applies it at lookup, in both modes. Re-validated live
2026-08-28: identical KV pool (1,480,759 tokens), identical correct output, and **~14%
faster decode than the x00byte file** (157 vs 138 tok/s, 1000-token runs).

**Fallback (`PLE_PATCH=x00byte`):** the original full-file replacement from
[`x00byte/Qwen3.8-Flash-Dual-Spark-Recipe`](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe)
at `/models/patches/qwen38flashnext-sm120/ple_layer.py`. Its two extra changes
(`weight_scale` parameter→buffer conversion) were only necessitated by x00byte's own
offload-worker changes; unnecessary on this image build. Kept as a known-good fallback.
(See `PROVENANCE.md` next to both files. Also discussed in the NVIDIA DGX Spark forum thread
["Qwen3.8-Flash-Next"](https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228), posts ~#66, #152.)

With either patch, the table loads as FP8 in VRAM: post-load usage dropped from ~90.4 GB/GPU
(upcasting attempt) to ~64 GiB/GPU (~128 GB total, matching the expected 128.5 GB).

---

## 5. Failure #2: profiling OOM with torch.compile — fixed by decoupling graphs from inductor

With the patch in place and compilation enabled, startup died in vLLM's memory-profiling
autotune block:

```
RuntimeError: Failed to run autotuning code block: CUDA out of memory.
Tried to allocate 47.69 GiB. GPU 0 has a total capacity of 95.01 GiB of which 30.73 GiB is free.
Including non-PyTorch memory, this process has 64.27 GiB memory in use.
```

- 47.69 GiB is **exactly** the FP8 PLE table size — inductor's autotune transiently clones the
  full table on top of the ~64 GiB already resident per GPU → 112 GiB needed > 95 GiB available.
- Lowering `--gpu-memory-utilization` does **not** help: the OOM happens during profiling,
  before KV sizing; the weights + fixed transient exceed the physical card.
- The DGX Spark recipe survives this because GB10 has ~120 GiB/rank (and they additionally run
  eager because torch.compile deadlocks GB10 unified memory — their "Problem #1").

**Benchmarked configs** (1000-token single-request decode, temp 0, 2026-08-28;
PM runs use the minimal patch, AM runs used the x00byte patch):

| Config | Decode | Host RAM | KV pool |
|---|---|---|---|
| `EAGER=1 PLE_OFFLOAD=0` (fully eager, full VRAM) | ~18 tok/s | 0 | 1.94M tok (7.4× @ 262K) |
| `PLE_OFFLOAD=1` inductor compile, SPEC=0, marlin | ~94 tok/s | ~48 GB | 3.86M tok (14.7×) |
| `PLE_OFFLOAD=0` + graphs-only cc, SPEC=0, marlin | ~114 tok/s | 0 | 1.94M tok |
| `PLE_OFFLOAD=0 SPEC=1` + graphs-only cc + cutlass, **x00byte patch** | ~138 tok/s | 0 | 1.48M tok (5.65×) |
| **`PLE_OFFLOAD=0 SPEC=1` + graphs-only cc + cutlass, minimal patch (DEFAULT)** | **~157 tok/s** (151–162) | **0** | 1.48M tok (5.65×) |
| `PLE_OFFLOAD=1 SPEC=1 COMPILATION_CONFIG=off MOE_BACKEND=auto` (inductor + offload + MTP) | **~163 tok/s** (154–171) | ~48 GB | **3.07M tok (11.72×)** |
| FP8 flavor (`start_qwen38_flash_next_fp8_vllm.sh`, §8) | **~168 tok/s** (160–174) | ~48 GB | 1.53M tok (5.85×) |

PM-update insights:

- **The minimal patch is ~14% faster than x00byte's** at the same config — stock's
  parameter-based FP8 embedding path beats the custom buffer/dequant path.
- **Inductor compile is worth ~4% over graphs-only** (163 vs 157) when the PLE table is
  offloaded (offload frees the profiling headroom that made inductor OOM at full residency).
  The offload+inductor+MTP combo also **doubles the KV pool** vs the default (3.07M vs 1.48M)
  at the cost of ~48 GB host RAM — the best config for many long-context sessions.
- **`MOE_BACKEND=auto` (unpinned) works fine with MTP** (validated by PR #87 and re-validated
  here): vLLM's oracle picks per-layer backends, so the unquantized BF16 MTP draft-head MoE
  no longer forces a global `flashinfer_cutlass`. Our marlin error was self-inflicted pinning.

Key insights:

- **The eager PLE path on this build is pathologically slow (5×)**, far beyond the normal
  CUDA-graph tax. Never run fully eager except to debug.
- The profiling clone comes from **torch inductor autotune** ("Failed to run autotuning code
  block" = `torch/_inductor/codegen/wrapper.py`). `--enforce-eager` is just shorthand for
  `-cc.mode=none -cc.cudagraph_mode=none` — and the two are **decoupled**:
  `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` runs eager PyTorch
  (no inductor → no clone) while still capturing decode CUDA graphs. It starts fine with the
  table resident (0.21 GiB of graph memory) and is **faster than the offload config** —
  on-GPU PLE rows beat the offload path's per-step PCIe prefetch.
- **MTP speculative decoding** (`SPEC=1`, default) adds ~1.2×: mean acceptance length 2.54,
  per-position acceptance 66%/50%/37%, avg draft acceptance 51.2% (counting task).
  **Needs a non-marlin MoE backend**: the MTP draft head's MoE is *unquantized* BF16
  (both checkpoints exclude `mtp.*` from quantization) and marlin is NVFP4-only →
  `ValueError: moe_backend='marlin' is not supported for unquantized MoE`.
  Default is `flashinfer_cutlass`; `MOE_BACKEND=auto` (unpinned, per-layer oracle) also
  works — validated PM (see PM-update insights above).
- KV pool trade-offs: offload = largest pool (3.86M); MTP = smallest (1.48M, draft head costs
  VRAM) but still 5.65× concurrency at 262K.

`PLE_OFFLOAD=1` adds `--cap-add=SYS_PTRACE` + `-e VLLM_PLE_CPU_OFFLOAD=1` + `--memory 62g
--memory-swap 62g` and preflight-checks for ≥50 GB free host RAM. The offload path registers
the scale as `_offload_weight_scale` (stock code, used by both patch variants), so the scale
is applied correctly in both modes. The script refuses the one non-starter combination
(`EAGER=0 PLE_OFFLOAD=0` with inductor compile) with an explanatory error.

---

## 6. Final working configuration (default: full VRAM residency + decode graphs + MTP)

Image: `vllm/vllm-openai:qwen38-flash-next` (vLLM `v0.1.dev20073+g8e685d198`)

Key server args (see the script for the full list):

```
--quantization modelopt_fp4        # pinned; auto-detect also worked
--tensor-parallel-size 2
--gpu-memory-utilization 0.95
--max-num-seqs 16
--max-num-batched-tokens 8192
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'  # graphs, no inductor
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'     # SPEC=1
--moe-backend flashinfer_cutlass  # with MTP (unquantized draft-head MoE); MOE_BACKEND=auto also works
--disable-custom-all-reduce        # sm_120: custom all-reduce unsupported
--enable-prefix-caching --no-enable-flashinfer-autotune
--tool-call-parser qwen3_xml --reasoning-parser qwen3 --enable-auto-tool-choice
```

Plus: bind-mounted patched `ple_layer.py`, `HF_HUB_OFFLINE=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
With `PLE_OFFLOAD=1`: `--cap-add=SYS_PTRACE` + `VLLM_PLE_CPU_OFFLOAD=1`.

Startup timeline on this box: ~2.5 min weight load (206 shards, page-cache warm) +
~75 s engine init (profile + KV alloc + cudagraph capture + MTP warmup).

### Verification

- `GET /v1/models` → `qwen3.8-flash-next`, `max_model_len: 262144` ✅
- `GPU KV cache size: 1,480,759 tokens, Maximum concurrency for 262,144 tokens per request: 5.65x` ✅
  (offload mode: 3,859,817 / 14.72×; eager full-residency: 1,936,131 / 7.39×)
- Chat test ("Give three prime numbers above 90, then say hello in French"):
  returned `97, 101, 103` + `Bonjour`, with working reasoning tokens (77) — confirms the
  FP8 PLE table is dequantized correctly (a bad PLE path would degrade output quality) ✅
  (re-verified with MTP enabled: same correct output)
- Decode, default mode: 1000 tokens in 7.25 s ≈ **138 tok/s** (AM, x00byte patch);
  **re-measured PM with the minimal patch: 151–162 tok/s (~157 avg)**, same config ✅;
  SpecDecoding metrics: mean acceptance length 2.54, avg draft acceptance 51.2% ✅
- Decode, other modes: graphs-only no-MTP 8.75 s (~114 tok/s); inductor+offload 10.71 s
  (~94 tok/s); fully eager 56.85 s (~18 tok/s) ✅
- Steady-state VRAM, default mode: ~91.5 / 97.9 GB per GPU, **zero host RAM used** ✅
- One transient, self-recovered warning during KV init in full-residency mode:
  `expandable_segments: memory mapping failed with OOM ... trying to map 20 MiB` — benign,
  startup completed normally right after.

---

## 7. Operational notes / gotchas

- **One model at a time**: all three vLLM script pairs (inferact / radixark / fp8) use port
  8000 and need both GPUs. Stop the running container before starting another flavor.
- **Cache dirs are per-flavor** (`~/.cache/qwen38-flash-next-<flavor>-vllm`) so JIT/torch.compile
  caches don't collide.
- **Patch is mandatory** for the RadixArk checkpoint on this image: the script hard-fails at
  preflight if the selected variant (`PLE_PATCH=minimal` → `ple_layer_minimal.py`,
  `PLE_PATCH=x00byte` → `ple_layer.py`) is missing from
  `/models/patches/qwen38flashnext-sm120/`.
- If a future vLLM image merges upstream FP8-PLE support for hybrid ModelOpt checkpoints,
  re-test **without** the bind mount (check `envs.py` / the model's `ple_layer.py` in the new
  image first) and delete the patch mount if it's no longer needed — a stale patch file can
  silently override newer upstream fixes.
- `TORCH_CUDA_ARCH_LIST=12.0f` (seen in online examples) is not needed with the dedicated image;
  it's only a fallback if arch-related JIT errors appear.
- Known upstream caveat for the offload path: `VLLM_PLE_CPU_OFFLOAD=1` at **TP=1** deadlocks
  during warmup ([vllm#53960](https://github.com/vllm-project/vllm/issues/53960)) — irrelevant
  here (we're TP2), but don't copy these flags to a single-GPU box.
- MTP speculative decoding is ON by default here (`SPEC=1`); the default backend with it is
  `MOE_BACKEND=flashinfer_cutlass` (the MTP draft head's MoE is unquantized BF16, marlin is
  NVFP4-only). `SPEC=0` flips the default backend back to marlin automatically.
  `MOE_BACKEND=auto` leaves `--moe-backend` unpinned (per-layer oracle; validated with MTP).
- `PLE_OFFLOAD=1` now also pins container RAM (`--memory 62g --memory-swap 62g`, from PR #87)
  so the ~48 GB offloaded table can never be swapped out.
- Script bug fixed 2026-08-28 PM: the compilation-config default must NOT be written as
  `${COMPILATION_CONFIG:-{...}}` — bash mis-parses the braces inside that expansion and
  appends a stray `}` when the variable IS set (e.g. `COMPILATION_CONFIG=off` became `off}`
  → `--compilation-config 'off}'` → vLLM JSON validation error). It now uses an explicit
  `if [ -z ... ]` default.
- If you want the largest possible KV pool (many long-context sessions): run the offload
  fallback `PLE_OFFLOAD=1 SPEC=0 COMPILATION_CONFIG=off ./start_qwen38_flash_next_radixark_vllm.sh`
  (3.86M tokens, ~94 tok/s, needs ~48 GB host RAM). The better speed/KV trade is the same
  with MTP: `PLE_OFFLOAD=1 COMPILATION_CONFIG=off SPEC=1 MOE_BACKEND=auto` (3.07M tokens,
  ~163 tok/s).

## 8. Sibling flavor: official FP8 checkpoint (`start_qwen38_flash_next_fp8_vllm.sh`)

Added 2026-08-28 PM. Checkpoint: `/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8` (185.6 GB,
`hf download Qwen/Qwen3.8-Flash-Next-FP8`), whole-model fine-grained FP8 W8A8 (dynamic
activations, 128×128 weight blocks). Recipe follows the validated FP8 compose from
[rtx6kpro PR #87](https://github.com/local-inference-lab/rtx6kpro/pull/87) adapted to our
script conventions. Differences vs the RadixArk NVFP4 scripts:

- **No patch at all**: the outer quant config IS `Fp8Config`, so stock vLLM's FP8 PLE gate
  fires natively (this is why the RadixArk checkpoint needs the gate patch and this one
  doesn't).
- **PLE CPU offload is mandatory** (185.6 GB cannot be GPU-resident on 2× 96 GB with any KV
  left): `VLLM_PLE_CPU_OFFLOAD=1` + `SYS_PTRACE` + `--memory 62g --memory-swap 62g`,
  preflight-checks ≥50 GB free host RAM.
- **Full inductor compile + FlashInfer autotune ON** (PR-validated for this path; the
  inductor profiling clone of the PLE table is a non-issue because the table is offloaded).
- **MoE backend unpinned** (`MOE_BACKEND=auto`), MTP on (3 draft tokens), no
  `--quantization` pin (auto-detected fp8).

Measured 2026-08-28 (1000-token decode, temp 0): **160/169/174 tok/s (~168 avg)**,
MTP mean acceptance length 2.60, KV pool **1,534,605 tokens (5.85× @ 262K)**, correct
output (`97, 101, 103` + `Bonjour`). ~7% faster than our NVFP4 default, but needs ~48 GB
host RAM and ~60 GB more disk. Start/stop: `start_qwen38_flash_next_fp8_vllm.sh` /
`stop_qwen38_flash_next_fp8_vllm.sh` (port 8000 — one model at a time, as always).

## 9. Files touched

- `start_qwen38_flash_next_radixark_vllm.sh` / `stop_qwen38_flash_next_radixark_vllm.sh` (new;
  PM update: `PLE_PATCH` selector [minimal default / x00byte fallback], `MOE_BACKEND=auto`
  support, `--memory/--memory-swap 62g` in offload mode, `${VAR:-{...}}` bash bug fix)
- `start_qwen38_flash_next_fp8_vllm.sh` / `stop_qwen38_flash_next_fp8_vllm.sh` (new, PM; §8)
- `start_qwen38_flash_next_inferact_vllm.sh` / `stop_qwen38_flash_next_inferact_vllm.sh`
  (renamed from `*_qwen38_flash_nvfp4.sh`; container name, cache dir, header docs updated)
- `/models/patches/qwen38flashnext-sm120/ple_layer_minimal.py` (new, PM: stock image file +
  PR #87's 11-line gate patch; DEFAULT)
- `/models/patches/qwen38flashnext-sm120/ple_layer.py` (new, from x00byte recipe, Apache-2.0;
  now the `PLE_PATCH=x00byte` fallback)
- `/models/patches/qwen38flashnext-sm120/PROVENANCE.md` (new; PM: documents both variants)
- `/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8/` (new, PM: 185.6 GB official FP8 checkpoint)
- `/models/eval/` (new, evening: lm-eval harness venv, `run_suite.sh` / `run_gen_only.sh`,
  `SUMMARY.md`, per-flavor `results/` with raw samples; §10)
- This document.

## 10. Quality benchmark comparison (lm-evaluation-harness, 2026-08-28 evening)

Suite: lm-eval 0.4.12 against the OpenAI-compatible server on :8000, one flavor at a time.
Scripts/results/samples in `/models/eval/` (`run_suite.sh`, `SUMMARY.md`, `results/<flavor>/`).

- **MC tasks (MMLU, ARC-C, HellaSwag, Winogrande): raw-completion loglikelihood, temp 0**
  (server-side prefix caching makes these fast). MMLU = 15% per-subject subsample 0-shot
  (~2100 docs); HellaSwag = first 2000.
- **GSM8K + IFEval: chat mode** (`local-chat-completions` + `--apply_chat_template`) with the
  model card's recommended **thinking sampling params: temp=1.0, top_p=0.95, top_k=20**
  (max_gen_toks 2048 / 4096).
- **HumanEval: raw completion, greedy** (code prompts don't trigger the EOS problem below).

| Task (metric) | FP8 (official) | RadixArk NVFP4 | Inferact NVFP4 |
|---|---|---|---|
| MMLU acc | **0.8580** | 0.8491 | ⛔ not yet measured |
| ARC-C acc_norm | 0.6365 | **0.6408** | ⛔ |
| HellaSwag acc_norm | **0.7855** | 0.7830 | ⛔ |
| Winogrande acc | 0.7190 | **0.7206** | ⛔ |
| HumanEval pass@1 | 0.7927 | **0.8110** | ⛔ |
| GSM8K exact_match (flex) | **0.8810** | 0.8779 | ⛔ |
| IFEval prompt strict | **0.8244** | 0.8096 | ⛔ |
| IFEval inst strict | **0.8369** | 0.8118 | ⛔ |

**Verdict: RadixArk NVFP4 is statistically indistinguishable from the official FP8 checkpoint**
— every delta is within ~1 stderr (MMLU ±0.0075, IFEval ±0.016, HumanEval ±0.031). The NVFP4
expert quantization + FP8 PLE table costs no measurable quality on these benchmarks while
being ~60 GB smaller on disk, needing zero host RAM (vs ~48 GB), and allowing full VRAM
residency.

**Decision (2026-08-28 evening): we settled on the FP8 flavor as the daily driver.** It is
the fastest measured config (~168 tok/s, §8), needs no `ple_layer.py` patch, and is quality-equivalent
to RadixArk on this suite; its costs (~48 GB host RAM for the PLE offload, ~60 GB more disk)
are acceptable on this box. RadixArk remains the zero-host-RAM fallback.

### Comparison with official published numbers

The official model card publishes no classic academic suite (its language table is agentic:
DeepSWE 58.7, SWE-bench Pro 62.5, Toolathlon 73.5, IFBench 81.3, GPQA Diamond 91.7, HLE 35.9,
LiveCodeBench v6 91.9). Classic-suite numbers exist only for the **base model** in the tech
report (Table 11, Qwen's internal few-shot pipeline):

| Benchmark | Official (base, tech report) | Ours: FP8 | Ours: RadixArk NVFP4 | Notes |
|---|---|---|---|---|
| MMLU | 90.36 | 85.80 | 84.91 | official = base model, few-shot; ours = post-trained, 0-shot loglikelihood |
| GSM8K | 93.29 | 88.10 | 87.79 | official = base, few-shot CoT; ours = post-trained chat @ temp 1.0, single sample |
| EvalPlus (code) | 78.76 | 79.27 (HumanEval) | 81.10 (HumanEval) | EvalPlus = stricter HumanEval+/MBPP+ avg; our plain HumanEval lands right on it |
| IFBench (card) | 81.3 | 82.44 (IFEval strict) | 80.96 (IFEval strict) | different benchmarks — not comparable, coincidentally close |
| ARC-C / HellaSwag / Winogrande | not published | 63.7 / 78.6 / 71.9 | 64.1 / 78.3 / 72.1 | no official reference exists |

Read: our numbers sit where expected once methodology differences are accounted for (base vs
post-trained, few-shot vs 0-shot, temp 1.0 sampling) — the ~4–5 pt MMLU/GSM8K gaps are
explained by those, not by quantization. Strongest evidence: the FP8 and RadixArk columns
track each other within noise on every task, and HumanEval matches the official EvalPlus
ballpark.

**Inferact not yet measured:** its BF16 PLE table needs ~93 GB host RAM for the CPU offload,
and the box currently has **~92–97 GB of orphaned anonymous memory** (`Active(anon)` ≈ 97 GB
vs 0.5 GB attributable to any live process — looks like pinned pages leaked when a PLE
offload worker was OOM-killed; predates this eval). Both startup attempts were OOM-killed at
the end of the table load, and each failed attempt leaks a few GB more — **do not retry the
Inferact script until the box is rebooted** (or the leaked pages are otherwise reclaimed).
After a reboot: `start_qwen38_flash_next_inferact_vllm.sh` + `/models/eval/run_suite.sh inferact`.

Eval gotchas discovered (worth knowing before re-running):

- **Raw-completion few-shot generation is broken for this model**: after a 5-shot
  `...Answer:` prompt it greedily emits `<|endoftext|>` as the FIRST token (p≈0.70) →
  empty responses (GSM8K scored 0.017 this way — an artifact, not model quality). MC
  loglikelihood tasks are unaffected (no generation involved). Use chat mode for any
  generation benchmark.
- **vLLM enforces OpenAI's max-4-stop-sequences limit**; lm-eval sends 5+EOS on HumanEval
  → HTTP 400. Fixed by patching `handle_stop_sequences` in `/models/eval/venv` to cap at 4
  (EOS string dropped first — vLLM stops on the model's EOS token natively, so nothing is lost).
- lm-eval chat models need `--apply_chat_template`; IFEval needs `pip install langdetect
  immutabledict`; HumanEval needs `HF_ALLOW_CODE_EVAL=1` + `--confirm_run_unsafe_code`.
- Tokenizer for the harness points at `/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8` (same
  tokenizer for all flavors).
