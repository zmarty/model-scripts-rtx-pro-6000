#!/usr/bin/env bash
# /models cleanup — driven by eval/model-inventory-2026-09-26.md (Artificial Analysis Index v4.3.2, 2026-09-26)
# DRY RUN BY DEFAULT.  Use:  cleanup_models.sh --tier1 [--apply]   |   --tier2 [--apply]   |   --all [--apply]
set -uo pipefail
ROOT=/models
APPLY=0; DO_T1=0; DO_T2=0; JUNK=1

for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --tier1) DO_T1=1; JUNK=0 ;;
    --tier2) DO_T2=1; JUNK=0 ;;
    --all)   DO_T1=1; DO_T2=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

# ---- never delete (live serving + validated fallback + current-gen + components) ----
PROTECT=(
  "$ROOT/fp8/Qwen-Qwen3.8-Flash-Next-FP8"
  "$ROOT/nvfp4/RadixArk-Qwen3.8-Flash-Next-NVFP4"
  "$ROOT/DeepSeek-V4-Flash-0731"
  "$ROOT/original/DeepSeek-V4-Flash-Vision-Exp"
  "$ROOT/original/google-gemma-4-31B-it"
  "$ROOT/original/google-gemma-4-31B-it-assistant"
  "$ROOT/embeddings/google-siglip2-base-patch16-224"
  "$ROOT/patches" "$ROOT/eval" "$ROOT/.venv"
)

TIER1=(
  # MiniMax M2 / M2.1 / M2.7 — all AA-deprecated (19/21/23); AA pushes MiniMax-M3 (29). 7 formats of M2 alone.
  "$ROOT/awq/cyankiwi-MiniMax-M2-AWQ-4bit" "$ROOT/awq/QuantTrio-MiniMax-M2-AWQ"
  "$ROOT/gguf/unsloth/MiniMax-M2-GGUF" "$ROOT/exl3/turboderp-MiniMax-M2-exl3-3.04bpw"
  "$ROOT/nvfp4/lukealonso-MiniMax-M2-NVFP4"
  "$ROOT/awq/cyankiwi-MiniMax-M2.1-AWQ-4bit" "$ROOT/awq/QuantTrio-MiniMax-M2.1-AWQ"
  "$ROOT/awq/mratsim-MiniMax-M2.1-FP8-INT4-AWQ" "$ROOT/nvfp4/lukealonso-MiniMax-M2.1-NVFP4"
  "$ROOT/awq/cyankiwi-MiniMax-M2.7-AWQ-4bit" "$ROOT/awq/QuantTrio-MiniMax-M2.7-AWQ"
  "$ROOT/nvfp4/lukealonso-MiniMax-M2.7-NVFP4" "$ROOT/nvfp4/NinjaBoffin-MiniMax-M2.7-NVFP4"
  # Qwen3-235B-A22B / -2507 (index 13, deprecated) — 5 redundant quants
  "$ROOT/awq/QuantTrio-Qwen3-235B-A22B-Thinking-2507-AWQ"
  "$ROOT/nvfp4/NVFP4-Qwen3-235B-A22B-Thinking-2507-FP4"
  "$ROOT/nvfp4/nvidia/Qwen3-235B-A22B-NVFP4"
  "$ROOT/nvfp4/RedHatAI-Qwen3-235B-A22B-NVFP4"
  "$ROOT/gptq/Qwen-Qwen3-235B-A22B-GPTQ-Int4"
  # Qwen3-VL-235B (13/10, deprecated) — vision covered locally by DSv4-Flash-Vision (35) + Gemma 4 (19)
  "$ROOT/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ"
  "$ROOT/gguf/unsloth/Qwen3-VL-235B-A22B-Thinking-GGUF"
  # GLM <=4.7 — every one AA-deprecated (chain: 4.5-Air -> 4.7-Flash -> 5 -> 5.2 -> 5.3 (45))
  "$ROOT/awq/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix" "$ROOT/gguf/Unsloth/GLM-4.5-Air-GGUF"
  "$ROOT/original/GLM-4.5-Air-FP8" "$ROOT/gguf/unsloth/GLM-4.6-GGUF"
  "$ROOT/awq/cyankiwi-GLM-4.6V-AWQ-8bit" "$ROOT/gguf/unsloth/GLM-4.6V-GGUF" "$ROOT/original/GLM-4.6V-FP8"
  "$ROOT/gguf/unsloth/GLM-4.7-GGUF"
  # Qwen text, deprecated & dominated by Qwen3.8-Flash-Next (40)
  "$ROOT/original/Qwen-Qwen3-32B" "$ROOT/original/RedHatAI-Qwen3-32B-speculator.eagle3"
  "$ROOT/original/Qwen-Qwen3.6-27B"
  # Mistral Devstral — explicit AA deprecation banner on both
  "$ROOT/awq/cyankiwi-Devstral-2-123B-Instruct-2512-AWQ-4bit"
  "$ROOT/original/Devstral-Small-2-24B-Instruct-2512"
  # INTELLECT-3 = GLM-4.5-Air post-train, same AA score (11), no AA speed/cost coverage
  "$ROOT/awq/cyankiwi-INTELLECT-3-AWQ-8bit"
  # Kimi K2 (13, deprecated, 128k, "particularly expensive"/"notably slow")
  "$ROOT/gguf/ubergarm/Kimi-K2-Instruct-GGUF"
  # Llama 3.3 70B (8, Dec-2024, AA: "particularly expensive")
  "$ROOT/original/nvidia-Llama-3.3-70B-Instruct-FP4"
  # Inferact Qwen3.8 NVFP4: BF16 PLE table needs ~93 GB host RAM, eval BLOCKED, RadixArk is the validated equivalent
  "$ROOT/nvfp4/Inferact-Qwen3.8-Flash-Next-NVFP4"
)

TIER2=(
  "$ROOT/original/DeepSeek-V4-Flash"                 # April FP8, AA 24, deprecated (keep 0731 = 34)
  "$ROOT/original/DeepSeek-V4-Flash-DSpark"          # drafter, no AA score; only if you never use DSpark spec-decode
  "$ROOT/int4/LordNeel-DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8"   # 3rd quant of the same base
  "$ROOT/original/openai-gpt-oss-120b"               # 12 but fast (192 t/s), NOT deprecated — drop with its drafters
  "$ROOT/original/nvidia-gpt-oss-120b-Eagle3" "$ROOT/original/nvidia-gpt-oss-120b-Eagle3-v2"
  "$ROOT/original/Snowflake-Arctic-LSTM-Speculator-gpt-oss-120b"
  "$ROOT/original/Qwen3.5-122B-A10B-FP8"             # 18 NR, not deprecated, dominated by Flash-Next (40)
  "$ROOT/original/Qwen-Qwen-Image"                   # Elo 887 -> replace with Qwen-Image-2.1 (1035)
  "$ROOT/original/Qwen-Qwen-Image-Edit-2509"         # Elo 980 -> 2511 (1023) / 2.1 (1071)
  "$ROOT/original/deepseek-ai-DeepSeek-OCR"          # superseded by DeepSeek-OCR2
  "$ROOT/embeddings/jinaai-jina-reranker-m0"         # CC-BY-NC-4.0, useless without a rerank stage
)

JUNKDIRS=(
  "$ROOT/gguf/lmstudio-community/MiniMax-M2-GGUF"
  "$ROOT/gguf/noctrex/MiniMax-M2-MXFP4_MOE-GGUF"
  "$ROOT/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF"
  # cleared 2026-09-26: gguf stubs, .Trash-1000 (sudo), stale fp8 .bak, inferact start/stop scripts
)

is_protected() { local p="$1"; for x in "${PROTECT[@]}"; do [[ "$p" == "$x" || "$p" == "$x"/* ]] && return 0; done; return 1; }

total=0; n=0
rm_one() {
  local p="$1"
  [[ -e "$p" ]] || { printf '  (absent)  %s\n' "$p"; return; }
  if is_protected "$p"; then printf '  PROTECTED %s\n' "$p"; return; fi
  local g; g=$(du -xs --block-size=1G "$p" 2>/dev/null | cut -f1); g=${g:-0}
  if pgrep -f "$p" >/dev/null 2>&1; then
    printf '  IN-USE    %s (a running process references this path)\n' "$p"; return
  fi
  if [[ ! -w "$p" ]]; then local s; s=$(du -xs --block-size=1G "$p" 2>/dev/null | cut -f1); s=${s:-0}
    printf '  NEEDS-SUDO %6s GB  %s\n' "$s" "$p"; total=$((total+s)); n=$((n+1)); return; fi
  total=$((total+g)); n=$((n+1))
  printf '  %6s GB  %s\n' "$g" "$p"
  if [[ $APPLY -eq 1 ]]; then rm -rf -- "$p" && printf '            -> deleted\n'; fi
}

[[ $JUNK -eq 1 ]] && { echo "# junk (empty dirs / trash / stale .bak)"; for p in "${JUNKDIRS[@]}"; do rm_one "$p"; done; }
[[ $DO_T1 -eq 1 ]] && { echo; echo "# TIER 1 — AA-deprecated, superseded locally, duplicate quants"; for p in "${TIER1[@]}"; do rm_one "$p"; done; }
[[ $DO_T2 -eq 1 ]] && { echo; echo "# TIER 2 — judgement calls"; for p in "${TIER2[@]}"; do rm_one "$p"; done; }

echo
echo "== $n path(s), $((total/1024)).$(( (total%1024)*10/1024 )) TB $( [[ $APPLY -eq 1 ]] && echo DELETED || echo reclaimable )"
[[ $APPLY -eq 1 ]] || echo "(dry run — add --apply to delete)"
