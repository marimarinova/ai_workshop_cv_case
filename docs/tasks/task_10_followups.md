# task_10 — Follow-ups

Deferred items from the Track A state-machine inference work, recorded so the
config and code do not mislead.

## Reserved config: `track_a.boundary_fallback_to_wrist_region`

Currently a **no-op** — toggling it has no effect. The state machine derives
event boundaries from the transition window (transfer onset -> stabilised
state). The wrist-region entry/exit fallback (task_10 step 6: "Wrist region
entry/exit is only a documented fallback") needs the wrist trajectory and shelf
geometry, which live in the inference/feature layer, plus a definition of when a
transfer is "ambiguous". That calibration is tied to task_7, so the field is
reserved until then. Enforced by a test asserting the flag does not change
`decode_events` output.

## Other known follow-ups (tracked elsewhere)

- Task 9 should expose a label-free feature core; `features.py` is a TEMPORARY
  shim until then.
- `TrackAStage` (`stage.py`) is a deferred Task 16 adapter, wired nowhere until
  `pipeline.py` is in master.
- Pre-existing, out of scope: the `fcntl` Windows incompatibility in
  `ingestion/cache.py` and the Task 9 strict-mypy debt — separate cleanup PRs.
