# Qwen3.8-Flash-Next quantization-variant quality eval (lm-evaluation-harness 0.4.12)
Server: vLLM :8000, one variant at a time. MC tasks: raw-completion loglikelihood, temp 0.
GSM8K/IFEval: chat mode (--apply_chat_template), thinking params temp=1.0 top_p=0.95 top_k=20, max_gen_toks 2048/4096.
HumanEval: raw completion, greedy. MMLU: 15% per-subject subsample, 0-shot. HellaSwag: first 2000.

| Task (metric)            | FP8 (official) | RadixArk NVFP4 | Inferact NVFP4 |
|---|---|---|---|
| MMLU acc (~2100 docs)    | 0.8580 | 0.8491 | BLOCKED |
| ARC-C acc_norm | 0.6365 | 0.6408 | BLOCKED |
| HellaSwag acc_norm | 0.7855 | 0.7830 | BLOCKED |
| Winogrande acc | 0.7190 | 0.7206 | BLOCKED |
| HumanEval pass@1 | 0.7927 | 0.8110 | BLOCKED |
| GSM8K exact (flex) | 0.8810 | 0.8779 | BLOCKED |
| IFEval prompt strict | 0.8244 | 0.8096 | BLOCKED |
| IFEval inst strict | 0.8369 | 0.8118 | BLOCKED |
