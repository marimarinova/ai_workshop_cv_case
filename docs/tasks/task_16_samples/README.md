# task_16 — Machine-readable sample output

Handoff artifacts for the batch CLI (`pickup-putdown infer`), per the task
handoff contract ("one machine-readable sample output or fixture" and "the
resolved configuration used for the acceptance run").

## Provenance — no-models path

These files were produced by the real orchestrator (`run_pipeline`, default
registry) on a tiny synthetic clip **without model checkpoints on disk**. This
is the path exercised by `tests/test_cli.py`:

```
PYTHONPATH=src pickup-putdown infer --input clip_demo.mp4 --output-dir out
```

With no `.pt`/`.joblib` weights present, every model-backed stage reports
`unavailable` and is skipped, so the run completes with `status: "ok"` and a
schema-valid but **empty** `predictions.csv` (header only). This demonstrates
the canonical output contract and graceful degradation; it does **not** contain
real detector events. The full acceptance run with real pickup/putdown events
is blocked on trained checkpoints (Task 7 / model weights) — see
[`../task_16_handoff_notes.md`](../task_16_handoff_notes.md).

## Files

- `predictions.csv` — canonical per-clip **model-output** file, the 7-column
  Prediction schema (`clip_id,pred_id,type,t_start,t_end,score,model`), header
  only. This is distinct from the ground-truth `events.csv` (annotation schema)
  that the evaluator consumes as its GT input.
- `summary.json` — per-clip run summary (`clip_dir/summary.json`). The
  `predictions_csv` path is relativised for the sample; `git_commit`/`run_id`
  are from the generating run.
- `resolved_config.yaml` — the resolved default `AppConfig` materialised to
  YAML (`clip_dir/resolved_config.yaml`), i.e. the configuration a stage
  subprocess would run under.
