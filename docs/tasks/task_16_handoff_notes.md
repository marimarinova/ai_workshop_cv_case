# task_16 — Handoff Notes (batch CLI / end-to-end inference)

Short note accompanying the `feature/task-16-batch-cli` work, per the task
handoff contract.

## Interface changes

- `pipeline.StageContext` gained two fields: `config_path` (resolved config
  materialised to YAML for CLI-based stages) and `cache_dir` (cross-clip cache
  root, derived from `AppConfig.cache_dir`).
- `pipeline.StageStatus` gained `"blocked"`. Top-level run status can now be
  `ok | no_person | blocked | failed`.
- `pipeline._atomic_write_json` was promoted to public `pipeline.atomic_write_json`.
- `infer` accepts a file **or** a directory and, in directory mode, writes
  `batch_summary.json`. Exit codes: `0` ok, `1` directory had ≥1 failure,
  `2` bad input, `4` single-file blocked, `5` single-file failed.

## Assumptions

- Stage subprocesses (`triage`, `propose`) load the same `AppConfig` schema, so
  the orchestrator forwards the resolved config via `--config`; this keeps the
  resume input-hash consistent with the behaviour it hashes.
- A stage is gated on its declared `inputs`: a *failed* upstream blocks it;
  a merely *unavailable* upstream cascades unavailability through the subtree.

## Known limitations

- **The executed triage/propose subprocess path (`_CliStage._invoke`) is not
  covered by tests without real model checkpoints.** Integration tests
  (`tests/test_cli.py`) drive the real registry, but with no `.pt` files on
  disk the model-backed stages report *unavailable*, so the actual subprocess
  invocation is exercised only when checkpoints are present (out of scope for
  CI smoke testing).
- Secondary configs not part of `AppConfig` — `tracker_config`,
  `shelves_config`, `camera_id` — are left at their CLI defaults and do not
  feed the resume input-hash.
- Cross-process orphan `*.partial-<pid>` stage directories from a crash in a
  previous process are not garbage-collected (no correctness impact: a missing
  marker forces reprocessing).
- No source videos, credentials, or model weights are committed.
