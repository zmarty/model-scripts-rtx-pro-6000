# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## ⚠️ This repo is PUBLIC

**Never commit secrets, credentials, or private information.** This includes:

- API keys / tokens of any kind (`HF_TOKEN`, `OPENAI_API_KEY`, GitHub PATs, bearer tokens, …)
- Passwords, SSH keys, certificates, `.env` files
- Internal hostnames, IPs (other than `localhost`), or infrastructure details
- Personal information, chat logs, or model outputs you wouldn't publish

Before committing, scan your changes for anything credential-shaped
(e.g. `grep -rEi 'token=|key|secret|password|bearer'` on new files).
If a secret is ever committed, treat it as compromised: rotate it
immediately and rewrite history (`git filter-repo`) — removing it in a
later commit is not enough.

## Repo layout

The git **worktree is `/models` itself** (the `.git` dir lives at
`/models/.git` with `core.worktree = ..`). Always run git commands from
`/models`. Do **not** move or delete `.git`, and do not change
`core.worktree`.

Tracked (commit these):

- `start_*.sh` / `stop_*.sh` — model serving launch/teardown scripts
- `README-*.md` — serving guides and hardware notes
- `eval/` — benchmark and thermal-monitoring tooling
- `patches/` — vLLM backend patches (source only)

Ignored (NEVER force-add or override the ignore rules):

- Model weights and checkpoints: `original/`, `fp8/`, `int4/`, `nvfp4/`,
  `DeepSeek-V4-Flash-0731/`, `embeddings/`, `hf-cache/` — hundreds of GB
  each; also any `*.safetensors`, `*.bin`, `*.gguf`, etc.
- Environments: `.venv/`, `eval/venv/`
- Logs and run artifacts: `*.log`, `eval/results/`, `__pycache__/`
- `lost+found/`

Never run `git add -f`, `git add .` with ignore rules modified, or
anything else that could stage weights/logs into the repo. A single
committed checkpoint file will exceed GitHub's limits and can't be
removed without history rewrite.

## Conventions

- Scripts are self-contained; document any new env-var knobs in the
  relevant README.
- Keep `AGENTS.md` and `.gitignore` up to date when adding new
  directories with large or machine-local artifacts.
