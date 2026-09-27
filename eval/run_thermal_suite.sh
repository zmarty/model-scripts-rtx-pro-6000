#!/usr/bin/env bash
set -euo pipefail
VENV=/models/eval/venv
OUT=/models/eval/results/thermal-0829
mkdir -p "$OUT"
export HF_ALLOW_CODE_EVAL=1
TOK="tokenizer=/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8"
COMP="--model local-completions --model_args model=qwen3.8-flash-next,base_url=http://localhost:8000/v1/completions,num_concurrent=8,tokenized_requests=False,$TOK"
COMMON="--log_samples --output_path $OUT"
echo "=== loglikelihood tasks (num_concurrent=8) ==="
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks mmlu --limit 0.15 2>&1 | tail -8
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks arc_challenge 2>&1 | tail -6
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks hellaswag --limit 2000 2>&1 | tail -6
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks winogrande 2>&1 | tail -5
echo "=== THERMAL_LL_DONE ==="
