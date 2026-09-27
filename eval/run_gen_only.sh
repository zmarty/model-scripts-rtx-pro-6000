#!/usr/bin/env bash
# Usage: ./run_gen_only.sh <variant-tag>  -- only humaneval (completion) + gsm8k/ifeval (chat)
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

# humaneval already done for fp8
echo "=== [$TAG] gsm8k + ifeval, chat mode, thinking sampling params ==="
$VENV/bin/lm_eval $CHAT --apply_chat_template $COMMON --gen_kwargs temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=2048 --tasks gsm8k 2>&1 | tail -7
$VENV/bin/lm_eval $CHAT --apply_chat_template $COMMON --gen_kwargs temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=4096 --tasks ifeval 2>&1 | tail -8

echo "=== [$TAG] GEN DONE -> $OUT ==="
