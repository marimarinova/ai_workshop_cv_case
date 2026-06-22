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

<!-- cce-block-version: 4 -->
## Context Engine (CCE)

This project uses Code Context Engine for intelligent code retrieval and
cross-session memory.

### Searching the codebase

**You MUST use `context_search` instead of reading files directly** when
exploring the codebase, answering questions about code, or understanding how
things work. This is a hard requirement, not a suggestion. `context_search`
returns the most relevant code chunks with confidence scores instead of whole
files, and tracks token savings automatically.

When to use `context_search`:
- Answering questions about the codebase ("how does X work?", "where is Y?")
- Exploring structure or architecture
- Finding related code, functions, or patterns
- Any time you would otherwise read a file just to understand it

When to use `Read` instead:
- You need to edit a specific file (read before editing)
- You need the exact, complete content of a known file path

Other search tools:
- `expand_chunk` — get full source for a compressed result
- `related_context` — find what calls/imports a function

### Cross-session memory — use it actively

This project has persistent memory across Claude Code sessions. **You must
use it both ways: recall before answering, record after deciding.** Memory
that is not recorded is lost; memory that is not recalled does nothing.

**Before answering a non-trivial question, call `session_recall`.**
Especially when:
- The question touches architecture, design, or naming choices
- The user asks "what / why / how did we ..."
- You are about to recommend an approach the team may have already chosen
  or already rejected

Pass a topic phrase, not a single word — e.g. `session_recall("auth flow")`,
not `session_recall("auth")`. Recall is vector-similarity-based, so paraphrases
match. If recall returns relevant entries, lead with them ("Per a prior
decision: ...") instead of re-deriving the answer.

**After making a non-obvious decision, call `record_decision`.** Especially:
- Choosing one library / pattern / approach over another
- Resolving an ambiguity in the spec or requirements
- Establishing a convention the project should follow going forward
- Anything you would not want to re-litigate next session

Format: `record_decision(decision="...", reason="...")`. Keep both fields
short and specific — they are surfaced verbatim at the start of future
sessions.

**After meaningful work in a file, call `record_code_area`.** Especially when:
- You added or substantially modified a function/class
- You traced through a non-obvious flow and want future-you to find it fast

Format: `record_code_area(file_path="...", description="...")`.

Skip recording for trivial reads, formatting changes, or one-off lookups —
the goal is durable signal, not an event log.

### Drilling deeper from a recall hit

`session_recall` results are tagged with the source session id, e.g.
`[turn sid:abc123|n:5]`. To drill in:

- `session_timeline(session_id="abc123")` — walk the per-turn summaries of
  that session in order. Use this when the user asks "what was the
  reasoning?" or "how did we get there?".
- `session_event(event_id=N)` — fetch a specific tool event's raw input
  and output (capped at 4 KB at read time). Use this when a turn summary
  references a tool result you actually need to inspect.

Both are read-only and cheap. Prefer them over re-running tool calls or
asking the user to re-paste context.

### Output style

Respond in compressed style. Drop articles (a, an, the) in prose. Use
sentence fragments over full sentences. Use short synonyms (fix not resolve,
check not investigate). Pattern: [thing] [action] [reason]. [next step].
No filler, hedging, pleasantries, trailing summaries, or restating what
the user said. One sentence if one sentence is enough.

When suggesting code changes, show only the changed lines with 3 lines of
context. Never rewrite entire files. Multiple changes in one file: show each
change separately. Never echo back unchanged code the user already has.

Code blocks, file paths, commands, error messages: always written in full.
Security warnings and destructive action confirmations: use full clarity.
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
