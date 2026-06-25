# VLM Annotation Pipeline

Runs candidate contact sheets through the local llama.cpp vision model and writes structured pickup/putdown annotations.

## Prerequisites

- llama.cpp is running at `http://localhost:8000`
- `Qwen3.6-27B-UD-Q4_K_XL` is loaded with `mmproj-BF16.gguf`
- The project environment is activated

Verify vision support:

```bash
MODEL_ID="Qwen3.6-27B-UD-Q4_K_XL"

curl -sG   --data-urlencode "model=${MODEL_ID}"   http://localhost:8000/props |
jq '.modalities'
```

Expected:

```json
{
  "vision": true,
  "audio": false
}
```

## Quick test

Run a small batch before the full job:

```bash
pickup-putdown annotate-vlm   .local/candidate_staging/candidates   --output-dir .local/vlm_annotations   --force   --vlm-base-url http://localhost:8000   --vlm-model Qwen3.6-27B-UD-Q4_K_XL   --vlm-timeout 180   --limit 2   -v
```

## Unattended tmux run

Create the output directory and start a detached session:

```bash
mkdir -p .local/vlm_annotations/logs

tmux new-session -d -s vlm-annotate   "set -o pipefail; pickup-putdown annotate-vlm   .local/candidate_staging/candidates   --output-dir .local/vlm_annotations   --force   --vlm-base-url http://localhost:8000   --vlm-model Qwen3.6-27B-UD-Q4_K_XL   --vlm-timeout 180   -v 2>&1 | tee .local/vlm_annotations/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log"
```

Watch the run:

```bash
tmux attach -t vlm-annotate
```

Detach with `Ctrl+B`, then `D`.

List sessions:

```bash
tmux ls
```

## Resume after interruption

Restart **without** `--force` so completed candidates are skipped:

```bash
pickup-putdown annotate-vlm   .local/candidate_staging/candidates   --output-dir .local/vlm_annotations   --vlm-base-url http://localhost:8000   --vlm-model Qwen3.6-27B-UD-Q4_K_XL   --vlm-timeout 180   -v
```

## Outputs

Written under `.local/vlm_annotations/`:

- `raw/` — VLM response and candidate metadata
- `normalized/` — validated candidate annotations
- `review_frames/` — extracted frames and contact sheets
- `events.csv` — canonical event rows
- `processing.csv` — per-candidate status ledger
- `summary.json` — aggregate run summary
- `logs/` — tmux run logs

## Check results

```bash
jq . .local/vlm_annotations/summary.json
```

```bash
awk -F, 'NR > 1 {count[$3]++} END {for (s in count) print s, count[s]}'   .local/vlm_annotations/processing.csv
```

## Common failures

| Error | Action |
|---|---|
| Connection refused | Start llama.cpp and verify port `8000` |
| Vision not supported | Confirm `mmproj-BF16.gguf` is loaded |
| HTTP 500 | Check `docker compose logs llm` |
| Interrupted run | Resume without `--force` |
