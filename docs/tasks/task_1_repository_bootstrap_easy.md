# task_1_easy: Repository Bootstrap, Configuration, and Core Schemas

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_1`  
**Difficulty:** `easy`  
**Dependencies:** None  
**Parallel work:** Tasks 2, 3, and 4 after the initial package skeleton is merged.

## Objective

Create the implementation repository foundation, dependency management, shared configuration conventions, and typed schemas that all later tasks consume.

## Inputs

- Concepts-aligned implementation plan
- Canonical schemas from the case: `clips.csv`, `events.csv`, and `predictions.csv`
- Python 3.12 (system or `pyenv`; no `pyenv` required if system Python is 3.12+)

## Deliverables

- Installable `pickup_putdown` Python package
- Pinned dependencies in `requirements.txt` (no `uv`, `poetry`, `pdm`, or any dependency-locking tool)
- Configuration loader and environment override support
- Pydantic models for clips, events, predictions, active spans, candidates, and ignore intervals
- Structured run metadata and logging helpers
- Initial unit-test configuration and CI smoke test

## Expected Files or Modules

- `pyproject.toml`, `requirements.txt`, `.gitignore`, `Makefile`
- `src/pickup_putdown/config.py`
- `src/pickup_putdown/common/schemas.py`
- `src/pickup_putdown/common/run_metadata.py`
- `tests/test_schemas.py`

## Implementation Steps

1. Create the package layout from the implementation plan and configure formatting, linting, typing, and tests. Install with `pip install -e .` (no `uv`, `poetry`, `pdm`, or dependency-locking tools).
2. Implement configuration loading from YAML with explicit validation and optional environment-variable overrides for secrets and storage endpoints.
3. Implement canonical field names exactly as required by the case. Internal schemas may contain extra fields, but canonical exporters must preserve the required columns.
4. Add schema validators for `t_start < t_end`, non-negative timestamps, allowed event types, allowed confidence values (`high`, `med`, `low`), and score range `[0,1]`.
5. Implement run metadata containing run ID, Git commit, dataset version, split version, resolved configuration, seed, model identifier, and checkpoint hash.
6. Add a common exception hierarchy and command exit-code conventions.
7. Add CI that imports the package and runs unit tests without requiring GPU models or source videos.

## Acceptance Criteria

- [ ] `python -m pytest` succeeds in a clean environment.
- [ ] Invalid event intervals and unsupported enum values fail validation.
- [ ] The same configuration resolves deterministically on two runs.
- [ ] The package can be installed in editable mode and imported.
- [ ] No raw-video, model-weight, credential, or cache path is tracked by Git.

## Out of Scope

- Cloud bucket indexing
- Video inference
- Model training
- Annotation-tool deployment

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
