# DeepSeek‑V4‑Flash on 2× RTX PRO 6000 (TP2) — Serving Guide & Debug Saga

This document captures everything we set up, debugged, and learned while getting
**DeepSeek‑V4‑Flash** running under vLLM on a 2× RTX PRO 6000 (Blackwell, `sm_120a`)
box using the community "Eldritch Enlightenment" vLLM images.

It doubles as (a) an operations guide for the working setup and (b) a post‑mortem of
the DSpark failure so we don't repeat the investigation.

---

## 1. Hardware & environment

| Item | Value |
|---|---|
| GPUs | 2× RTX PRO 6000 (Blackwell), arch `sm_120a` |
| Parallelism | Tensor parallel size 2 (TP2) |
| Interconnect | PCIe (no NVLink, no InfiniBand) |
| Model root | `/models` (bind‑mounted into the container) |
| Container runtime | Docker, NVIDIA runtime |
| Workload profile | **Low concurrency**, interactive coding / chat |

Model checkpoints on disk:

```
/models/original/DeepSeek-V4-Flash          # standard checkpoint (known-good)
/models/original/DeepSeek-V4-Flash-DSpark   # DSpark checkpoint (speculative variant)
```

---

## 2. TL;DR — what to run

**Use the stable v9 + MTP:2 script.** It is the recommended, working setup.

```bash
cd /models
./start_ds4_flash_v9_mtp.sh
docker logs -f ds4-v9-mtp-tp2      # follow startup; first launch warms cache (~5 min)
```

It serves an OpenAI‑compatible API on `http://localhost:8000/v1` as model
`DeepSeek-V4-Flash`.

### Scripts in this folder

| Script | Image | Checkpoint | Spec decode | Status |
|---|---|---|---|---|
| `start_ds4_flash_v9_mtp.sh` | **v9** ds4dspark | standard | **MTP:2** | ✅ **Recommended, working** |
| `start_ds4_flash.sh` | `lucifer` (Jun 9) | standard | MTP:2 | ✅ Works (older image, fallback) |
| `start_ds4_flash_dspark.sh` | v9 ds4dspark | DSpark | dspark:5 | ❌ Broken on v9 (see §7) |

---

## 3. Design decisions baked into the scripts

All three scripts share the same structure and these deliberate choices:

- **Background service, manual start.** `docker run -d --restart no`, so the server
  runs detached (survives closing the terminal) but does **not** auto‑restart on
  reboot — you start it yourself when you need it.
- **Cache dir in the user's home, not `/root`.**
  `CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/…"`. This is just the host directory
  that holds the warm JIT/autotune cache; it is bind‑mounted to `/cache` in the
  container. First launch on a fresh cache re‑warms (~5 min); subsequent launches reuse it.
- **Low concurrency.** `--max-num-seqs 8` (not 64). This matches actual usage.
  Note: `--max-num-seqs` is only an **upper cap**, not a memory reservation. The KV
  cache is paged and bounded by `--gpu-memory-utilization`, so a high value never
  OOMs at steady state — vLLM just queues/preempts. `--max-model-len` is the per‑request
  ceiling (input+output), **not** a per‑sequence reservation.
- **Readable argument arrays.** Each flag lives on its own line in a commented bash
  array (`DOCKER_ARGS`, `VLLM_ARGS`) that is expanded into the real command. This is
  the only clean way to get a comment per switch — you can't comment inside a
  backslash‑continued `docker run` line or inside the single‑quoted `vllm serve` string.
- **Env unset wrapper.** The image bakes in PCIe/fused‑all‑reduce tunables that are
  problematic on this 2‑GPU PCIe box; the inner `bash -lc` clears them so we fall back
  to the plain NCCL path before `exec vllm serve`.

---

## 4. Context window & token limits

- Model architectural ceiling: `max_position_embeddings = 1,048,576` (1M tokens, input+output combined).
- Our server ceiling: `--max-model-len 262144` (256K). Raise up to 1M at the cost of KV‑cache VRAM.
- `generation_config.json` sets **no** `max_new_tokens`, so there is no model‑imposed output cap.
- Effective max output for a request = `max-model-len − prompt_tokens`, further limited by the client's `max_tokens`.

### VS Code custom model config

File: `~/.config/Code - Insiders/User/chatLanguageModels.json`, vendor `customoai`.

The rule is `maxInputTokens + maxOutputTokens ≤ max-model-len (262144)`. We set:

```jsonc
"maxInputTokens": 229376,
"maxOutputTokens": 32768
```

The original `262144 + 8192` overflowed the window, and `8192` output was too small
for a high‑effort reasoning model (the `<think>` chain counts as output).

---

## 5. Is DSpark worth it? (the strategic question)

From the v9 benchmark doc, mapping our config (TP2, lucifer‑cutlass):

| Metric | standard MTP:2 | DSpark | Verdict |
|---|---:|---:|---|
| coding peak median | 228 | **308** (+35%) | DSpark better for single‑stream coding |
| cc1 (single) | 215 | **228** | DSpark slightly better |
| cc32 / cc64 (concurrency) | **1844 / 2790** | 1740 / 2498 | MTP better at concurrency |
| prefill 8k/64k/128k | ~equal | ~equal | wash |

**Conclusion:** DSpark only wins for **low‑concurrency / interactive** use (which is
our profile), by ~35% on coding throughput. For high concurrency, MTP:2 wins. This
made DSpark worth *attempting* — but see §7.

### Community reality check (from Discord)

Even where DSpark boots, users report it is **unstable in practice**:
- Random garbage output / sudden switches to **Chinese / Russian / Czech**, broken tool
  calls (the "Czech issue") — many independent reports.
- **Low speculative acceptance (~34–60%)**, which wastes compute; MTP:2 often accepts more.
- DSpark uses **more** KV cache than MTP:2 (larger checkpoint).
- Model card recommends `temperature = 1.0, top_p = 1.0`; temperature is a suspected factor in the garble.

---

## 6. The setup saga (chronological)

1. **"Why does the script need `/root`?"** — It didn't; `/root/.cache/...` was just the
   host cache path chosen because it ran as root. Switched to `$HOME/.cache`.
2. **Restart behavior** — Confirmed `docker run -d` = background, survives terminal close.
   Changed `--restart unless-stopped` → `--restart no` so it doesn't come back after reboot.
   Cleared the old exited container.
3. **Newer image?** — Found the `lucifer` tag was from Jun 9; the newest DS4‑specific
   build was `…ds4dspark-v9-…-20260703` (same day). Decided to try v9 + DSpark.
4. **Readability pass** — Rewrote both scripts to commented `DOCKER_ARGS` / `VLLM_ARGS`
   arrays; reduced concurrency to `--max-num-seqs 8`.
5. **DSpark bring‑up** — Created `start_ds4_flash_dspark.sh`, then hit a cascade of
   failures (§7).
6. **Fallback to stable** — Built `start_ds4_flash_v9_mtp.sh` (v9 image + standard
   checkpoint + MTP:2). **This one works** and is now running.

---

## 7. The DSpark failure — full post‑mortem

Getting DSpark to run on the **v9** image failed through three distinct layers.

### 7.1 Wrong attention backend name (fixed)

The v9 image renamed the sparse‑MLA backend.

- Old (`lucifer` image): `--attention-backend SPARSE_MLA_SM120`
- **v9 image: `--attention-backend FLASHINFER_MLA_SPARSE_DSV4`**

The "lucifer‑cutlass" path = `FLASHINFER_MLA_SPARSE_DSV4` + `--kernel-config.moe_backend flashinfer_cutlass`.

### 7.2 Parallel‑drafting token crash (dead‑end workaround)

```
ValueError: For parallel drafting, the draft model config must have
`pard_token`, `ptd_token_id`, or `dflash_config.mask_token_id` in its config.json.
```

Why: `method=dspark` force‑sets `parallel_drafting=True` (speculative.py:1004), which
makes the generic `EagleProposer` call `_init_parallel_drafting_params()`. That function
only knows `pard_token` / `ptd_token_id` / `dflash_config.mask_token_id` — it was never
taught DSpark's `dspark_noise_token_id`.

We tried adding `"pard_token": 128799` (the noise token) to the checkpoint's
`config.json`. It got past this crash but exposed a deeper one, so **the edit was
reverted** (`config.json.orig` is the backup).

### 7.3 The real incompatibility (root cause)

```
KeyError: 'model.layers.43.mtp_block.main_norm.weight'   (deepseek_v4/nvidia/mtp.py load_weights)
```

Investigation of the checkpoint index vs. the image loader:

| | DSpark checkpoint weights | v9 image `mtp.py` expects |
|---|---|---|
| Draft layers | **3** (`mtp.0`, `mtp.1`, `mtp.2`) | **1** MTP layer (`layers.43`) |
| DSpark parts | `mtp.0.main_norm`, `mtp.2.markov_head`, `mtp.2.confidence_head` | plain `mtp_block`, **no** `main_norm` |
| `config.json` | `num_nextn_predict_layers: 1`, **no** `n_mtp_layers` | builds 1 plain MTP layer from that |

So the **weights are full DSpark (3 layers + Markov/confidence heads), but `config.json`
describes a single plain MTP layer.** The image builds the wrong draft model and dies
loading `main_norm`.

Verified this is **not** a bad/wrong‑revision download: the pinned HF revision
`913f0657a874f76844e2e91cbe706dbcaceeb6d7` has a byte‑identical `config.json`
(`num_nextn_predict_layers: 1`, no `n_mtp_layers`). The v9 doc's claim of
`n_mtp_layers=3` simply does not match the actual published config.

**Conclusion:** v9's upstream‑merged DSpark loader is incompatible with the currently
published DSpark checkpoint. This matches the maintainer's own note that the DSpark
implementation is unreviewed vs. upstream. The community's working DSpark setups were
on the **v8** image.

### 7.4 Resolution

Abandoned DSpark on v9. Built `start_ds4_flash_v9_mtp.sh` = **v9 image + standard
`DeepSeek-V4-Flash` checkpoint + MTP:2 + `FLASHINFER_MLA_SPARSE_DSV4`**. It:

- resolves `DeepSeekV4MTPModel` cleanly (where DSpark crashed),
- is the strongest **stable** row in the v9 doc for this box,
- avoids the DSpark garble/Czech bug entirely,
- is a drop‑in on port 8000 (`--served-model-name DeepSeek-V4-Flash`), so VS Code needs no change.

Confirmed live: `Application startup complete`, `SERVING: ['DeepSeek-V4-Flash']`,
KV cache ≈ 8.46 GiB (≈1.97× concurrency at 256K).

---

## 8. Operations cheat sheet

```bash
# Start (recommended, stable)
./start_ds4_flash_v9_mtp.sh

# Follow startup / logs
docker logs -f ds4-v9-mtp-tp2

# Check it is serving + report served max_model_len
curl -s localhost:8000/v1/models | python3 -m json.tool

# Smoke-test generation
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":50}' \
  | python3 -m json.tool

# Stop / remove
docker stop ds4-v9-mtp-tp2
docker rm -f ds4-v9-mtp-tp2

# Container names by script:
#   start_ds4_flash_v9_mtp.sh -> ds4-v9-mtp-tp2   (v9, MTP2, stable)
#   start_ds4_flash.sh        -> ds4-lucifer-tp2  (old lucifer image)
#   start_ds4_flash_dspark.sh -> ds4-dspark-tp2   (v9 dspark, broken)
```

Do not run two of these on port 8000 at once.

---

## 9. Key gotchas / learnings

- **Backend name changed between images:** `SPARSE_MLA_SM120` (lucifer) →
  `FLASHINFER_MLA_SPARSE_DSV4` (v9). A wrong name aborts startup with a list of valid options.
- **v9 memory profiler** (`VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1`) needs
  `--gpu-memory-utilization 0.93` at `max-model-len 262144` (0.90 is fine on the old lucifer image).
- **`--max-num-seqs` is a cap, not a reservation.** High values don't OOM; KV is paged.
- **DSpark on v9 is broken** against the current checkpoint (§7.3). If you must run
  DSpark, replicate a **v8** recipe (v8 image, JSON `--speculative-config` with explicit `"model"`).
- **DSpark is unstable in practice** even when it boots (garble/Czech issue, low acceptance,
  more KV). MTP:2 is the safer default.
- **DS4/GLM are designed for `fp8` KV cache** — keep `--kv-cache-dtype fp8`.
- If you re‑download the DSpark checkpoint, restore `config.json` from
  `config.json.orig` (or just don't hand‑edit it).

---

*Environment: `vllm 0.11.2.dev279+eldritch.enlightenment.ds4dspark.v9…cu132`,
CUDA 13.2, PyTorch 2.12, NCCL 2.30.4.*
