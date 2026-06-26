# Layer 1 Track A Implementation Plan

## Current State

Tasks 1–9 are complete:

* Ingestion and triage
* Shelf regions and proposals
* Annotation workflow
* Dataset quality and splits
* Shared evaluator
* Track A feature extraction infrastructure

Available data:

* Canonical reviewed events: `task_7_vlm/events.csv`
* Reviewed task manifest: `task_7_review/review_manifest.csv`
* Source videos and candidate clips in S3
* Existing feature extraction and evaluation modules

Task 10 must add:

* Hand-state classifier
* Shelf-transition classifier
* Repeating state machine
* Inference pipeline
* CLI commands
* `configs/track_a.yaml`

---

## Phase 1: Build the Track A Dataset

Build a feature dataset from reviewed candidates only.

Steps:

1. Load the canonical reviewed events.
2. Load the reviewed task manifest.
3. Map reviewed candidates to their source videos and metadata.
4. Assign train, validation, and test splits at clip level, grouped by recording day.
5. Extract and cache MobileNetV3 embeddings in:

```text
.local/track_a_features/
```

6. Validate split isolation.

Only the following may be used as supervised training data:

* Verified pickup events
* Verified putdown events
* Reviewed tasks explicitly confirmed to contain zero events

Unreviewed candidates must remain unlabeled.

---

## Phase 2: Train State Classifiers

Add:

```text
src/pickup_putdown/layer1/track_a/hand_state.py
src/pickup_putdown/layer1/track_a/shelf_state.py
```

Use logistic regression on frozen MobileNetV3 embeddings.

### Hand-State Classes

* `empty`
* `carrying`
* `uncertain`

Derived labels:

* Pickup: pre = empty, post = carrying
* Putdown: pre = carrying, post = empty

### Shelf-State Classes

* `object_removed`
* `object_placed`
* `no_change`
* `uncertain`

Derived labels:

* Pickup post-sample = `object_removed`
* Putdown post-sample = `object_placed`
* Reviewed zero-event task = `no_change`

Do not treat unmatched or unreviewed candidates as negatives.

Add `scikit-learn` to the optional dependencies in `pyproject.toml`.

---

## Phase 3: Implement the State Machine

Add:

```text
src/pickup_putdown/layer1/track_a/state_machine.py
```

State cycle:

```text
OUTSIDE
→ APPROACHING
→ CONTACT
→ TRANSFER
→ WITHDRAWING
→ OUTSIDE
```

Process observations independently per:

* `actor_id`
* `hand_side`
* `region_id`

Event rules:

```text
Pickup:
hand empty → carrying
AND shelf object_removed
```

```text
Putdown:
hand carrying → empty
AND shelf object_placed
```

The state machine must:

* Preserve temporal order
* Emit zero, one, or multiple events
* Keep adjacent pickup and putdown events separate
* Treat interaction without persistent transfer as background

---

## Phase 4: Implement Inference

Add:

```text
src/pickup_putdown/layer1/track_a/inference.py
configs/track_a.yaml
```

Inference flow:

1. Load candidates and pose observations.
2. Extract or load cached features.
3. Run the hand and shelf classifiers.
4. Run the state machine.
5. Estimate event boundaries.
6. Compute confidence.
7. Export canonical `predictions.csv`.

---

## Phase 5: Add CLI Commands

Add to `src/pickup_putdown/cli.py`:

| Command                 | Purpose                                      |
| ----------------------- | -------------------------------------------- |
| `build-track-a-dataset` | Build the reviewed Track A feature dataset   |
| `train-track-a`         | Train and save both classifiers              |
| `infer-track-a`         | Run Track A inference and export predictions |

Add Makefile targets:

```text
track-a-dataset
train-track-a
infer-track-a
```

---

## Phase 6: Evaluate

Use the existing Task 8 evaluator.

Report:

* Precision
* Recall
* F1
* Temporal IoU thresholds 0.3 and 0.5

Generate previews for:

* False positives
* Missed events

---

## Main Files

```text
src/pickup_putdown/layer1/track_a/hand_state.py
src/pickup_putdown/layer1/track_a/shelf_state.py
src/pickup_putdown/layer1/track_a/state_machine.py
src/pickup_putdown/layer1/track_a/inference.py
configs/track_a.yaml
src/pickup_putdown/cli.py
pyproject.toml
Makefile
tests/test_hand_state.py
tests/test_shelf_state.py
tests/test_state_machine.py
tests/test_inference.py
```

---

## Data Rules

* Use `task_7_vlm/events.csv` as canonical ground truth.
* Use only reviewed tasks for supervised training.
* A candidate without a matching event is not automatically negative.
* Only reviewed tasks with zero confirmed events are verified negatives.
* Keep all other candidates unlabeled.

---

## S3 Locations

```text
s3://chillnbite-cameras/anon/candidates/
s3://chillnbite-cameras/anon/vlm/2026-06-26/task_7_vlm/
s3://chillnbite-cameras/anon/vlm/2026-06-26/task_7_review/
```

Example download:

```bash
aws s3 sync \
  s3://chillnbite-cameras/anon/vlm/2026-06-26/task_7_vlm/ \
  .local/task_7_vlm/

aws s3 sync \
  s3://chillnbite-cameras/anon/vlm/2026-06-26/task_7_review/ \
  .local/task_7_review/
```

---

## Validation

```bash
ruff check src/pickup_putdown/layer1/track_a/
ruff format --check .

python -m pytest \
  tests/test_hand_state.py \
  tests/test_shelf_state.py \
  tests/test_state_machine.py \
  tests/test_inference.py \
  -v

python -m compileall src
```

Smoke test:

```bash
pickup-putdown infer-track-a \
  .local/source_videos/D2_S20260520141725_E20260520142151_anon.mp4 \
  --config configs/track_a.yaml \
  -o .local/track_a_output/
```

---

## Recommended Order

1. Build the reviewed dataset.
2. Train the classifiers.
3. Implement the state machine.
4. Implement inference.
5. Add CLI and Makefile targets.
6. Evaluate the generated predictions.
