#!/usr/bin/env bash
# Usage: ./run_suite.sh <variant-tag>   (server must already be running on :8000)
set -euo pipefail
TAG="$1"
VENV=/models/eval/venv
OUT=/models/eval/results/$TAG
mkdir -p "$OUT"
export HF_ALLOW_CODE_EVAL=1
TOK=tokenizer=/models/fp8/Qwen-Qwen3.8-Flash-Next-FP8
COMP="--model local-completions --model_args model=qwen3.8-flash-next,base_url=http://localhost:8000/v1/completions,num_concurrent=32,tokenized_requests=False,$TOK"
CHAT="--model local-chat-completions --model_args model=qwen3.8-flash-next,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,tokenized_requests=False,$TOK"
COMMON="--log_samples --output_path $OUT"

echo "=== [$TAG] loglikelihood tasks, completion mode ==="
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks mmlu --limit 0.15 2>&1 | tail -8
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks arc_challenge 2>&1 | tail -6
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks hellaswag --limit 2000 2>&1 | tail -6
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks winogrande 2>&1 | tail -5

echo "=== [$TAG] humaneval, completion mode, greedy ==="
$VENV/bin/lm_eval $COMP $COMMON --gen_kwargs temperature=0.0 --tasks humaneval --confirm_run_unsafe_code 2>&1 | tail -6

echo "=== [$TAG] gsm8k + ifeval, chat mode, thinking sampling params ==="
$VENV/bin/lm_eval $CHAT --apply_chat_template $COMMON --gen_kwargs temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=2048 --tasks gsm8k 2>&1 | tail -7
$VENV/bin/lm_eval $CHAT --apply_chat_template $COMMON --gen_kwargs temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=4096 --tasks ifeval 2>&1 | tail -8

echo "=== [$TAG] DONE -> $OUT ==="
