# Agent Instructions

This repository is a Python-first computer-vision project for detecting pickup and putdown events in video. Prefer small, safe, targeted changes that preserve existing behavior and keep experiments reproducible.

## Working rules

* Inspect relevant code and configuration before editing.
* Follow existing structure, naming, and conventions.
* Make the smallest change that fully solves the task.
* Modify existing modules before adding new abstractions.
* Do not add speculative features or unrelated refactors.
* Keep data loading, preprocessing, inference, post-processing, evaluation, and export as separate concerns.
* Explain assumptions when they affect implementation.
* Prefer clear errors and diagnostics over silent fallbacks.
* Do not run destructive commands without explicit approval.

Destructive commands include:

```bash
git reset
git clean
rm -rf
docker compose down -v
```

## Python

* Follow PEP 8 and repository conventions.
* Use existing dependencies and the Python standard library first.
* Add dependencies only when they materially improve correctness or maintainability.
* Update the correct `pyproject.toml` or requirements file when adding one.
* Use precise type hints.
* Keep functions small and testable.
* Isolate side effects.
* Use explicit exceptions and useful error messages.
* Use project logging instead of ad hoc `print` calls where practical.
* Never log secrets, tokens, credentials, or private keys.

## Computer vision and ML

Treat reproducibility as a requirement.

* Keep raw input data immutable where practical.
* Write derived outputs to clearly named directories.
* Do not overwrite experiment artifacts unless explicitly intended.
* Record model name, checkpoint, configuration, thresholds, dataset version, evaluation split, and metrics.
* Preserve source clip and frame identifiers in predictions and evaluation outputs.
* Split train, validation, and test data by source clip or recording group when frame-level splitting would cause leakage.
* Keep baseline, trained-model, heuristic, and VLM-verifier outputs separate.
* Save qualitative examples alongside quantitative metrics when useful.
* Do not run expensive training or full-dataset processing without explicit approval.

## Event-detection rules

* Follow repository definitions of `pickup`, `putdown`, item count, visibility, and event timestamp.
* Do not silently change event semantics or label interpretation.
* Keep frame extraction, tracking, pose estimation, temporal detection, and VLM verification separable.
* Preserve timestamps and mappings between videos, frames, tracks, detections, and final events.
* Make temporal windows, thresholds, and confidence rules explicit.
* Prefer deterministic evaluation and output ordering where practical.

## Docker, configuration, and scripts

* Prefer existing Makefile and Docker Compose workflows.
* Keep ports, volumes, environment variables, and health checks explicit.
* Do not introduce host-specific paths unless documented.
* Update `.env.example` when adding required environment variables.
* Never commit real `.env` files or secrets.
* Use safe defaults for local development.

For Bash scripts, use strict mode where appropriate:

```bash
set -Eeuo pipefail
```

Quote variables, validate arguments, return non-zero on failure, and avoid printing secrets.

## Code changes

* Keep public interfaces stable unless the task requires changing them.
* Add or update tests for non-trivial behavior.
* Comments should explain why, not restate the code.
* Avoid global mutable state.
* Avoid compatibility layers unless required.
* Keep configuration explicit and reviewable.

## Verification

Run the smallest relevant set of checks. Prefer existing project commands.

```bash
ruff check .
ruff format --check .
python -m pytest
python -m compileall src
git diff --check
git diff
```

For shell scripts:

```bash
shellcheck path/to/script.sh
```

For Docker Compose changes:

```bash
docker compose config
```

Do not claim a check passed unless it was actually run and completed successfully.

## Optional Code Context Engine

The following section applies only when the named Code Context Engine tools are
available in the current agent environment.

If a required CCE tool is unavailable, use normal repository inspection tools
instead and continue without failing the task.

The version markers delimit the managed CCE instructions for automated updates.
They do not provide conditional execution by themselves.

<!-- cce-block-version: 4 -->

## Context Engine (CCE)

This project may use Code Context Engine for code retrieval and cross-session
memory.

When CCE tools are available, follow the rules in this section.

### Searching the codebase

Use `context_search` before broad file exploration when:

* Answering questions about repository code.
* Locating a feature, function, class, configuration, or workflow.
* Exploring architecture or dependencies.
* Finding existing patterns before implementing a change.
* Determining which files require direct inspection.

Use a direct file read when:

* The user names a specific file.
* You need exact and complete file contents.
* You are preparing to edit a known file.
* You need to verify a search result before changing code.

Use supporting tools where available:

* `expand_chunk` to inspect complete source around a search result.
* `related_context` to find callers, imports, dependencies, and related symbols.

Do not use `context_search` merely to avoid reading a file whose exact path and
purpose are already known.

### Cross-session memory

Use CCE memory for durable repository context, not as a general event log.

Before a non-trivial repository question or implementation task, call
`session_recall` when prior decisions may affect:

* Architecture.
* Naming or layout conventions.
* Model, library, or framework selection.
* Data and evaluation conventions.
* Previously accepted or rejected approaches.
* Continuation of earlier implementation work.

Use a descriptive topic phrase:

```text
session_recall("temporal event detection architecture")
session_recall("pickup putdown timestamp definition")
session_recall("VLM verifier output schema")
```

Avoid vague single-word queries.

Recalled memory is supporting context, not authoritative state. Verify relevant
claims against current code and configuration before implementing changes.
Current repository state and explicit user instructions take precedence over
stored memory.

After making a durable, non-obvious decision, call:

```text
record_decision(decision="...", reason="...")
```

Record decisions when:

* Selecting one approach over another.
* Resolving an ambiguous requirement.
* Establishing a lasting project convention.
* Rejecting an approach for a specific reason.

After adding or substantially changing an important code area, call:

```text
record_code_area(file_path="...", description="...")
```

Record code areas when:

* Adding or substantially modifying a function, class, service, script, or
  workflow.
* Tracing a non-obvious flow that future work should locate quickly.

Do not record trivial reads, formatting-only edits, temporary diagnostics, or
routine implementation details.

### Drilling deeper from recalled context

When `session_recall` returns a relevant source session, use available drill-down
tools rather than asking the user to repeat known context:

* `session_timeline` to inspect summarized steps from that session.
* `session_event` to inspect a referenced prior tool event.

Use these tools only when the recalled summary does not contain enough detail.

### Output style

For normal code discussion:

* Use direct, concise technical language.
* Avoid filler and repeated restatement.
* Show changed lines with limited surrounding context unless a full file is
  requested.
* Write file paths, commands, and error messages in full.
* Clearly state security warnings and destructive-action requirements.

The **Required final report** section below overrides these compression rules.
Implementation tasks must still end with the complete required report.

<!-- /cce-block -->

## Required final report

Finish every implementation task with a complete report containing:

1. **Summary** — what was implemented and why.
2. **Files changed** — every added, modified, renamed, or deleted file.
3. **Behavior changes** — user-visible or system-visible effects.
4. **Verification** — commands run and their results.
5. **Skipped checks** — checks not run and the reason.
6. **Assumptions and risks** — unresolved concerns, limitations, or follow-up work.

Keep the report concise, but do not omit failures, partial results, or unverified claims.

## Final instruction

Do the smallest safe thing that solves the explicit task, preserves the existing architecture, and keeps the video-event workflow reproducible.
