# Local Run Plan

Run Layer 1A, Layer 1B, and Layer 2 consecutively on laptop without remote server access.
Classifier weights and Qwen model are stuck on the remote server.

---

## Inventory

### What you have locally

- `candidates.parquet` (15 candidates, `clip_person_clear`)
- `tracks_pose.parquet` (543 pose observations, `clip_person_clear`)
- `person_clear.mp4` (140s, 20fps, 3840x2160) in `.local/triage_acceptance/videos/`
- `configs/track_a.yaml`, `configs/track_b1.yaml`, `configs/shelves.yaml`
- 1000 candidate JSONs in `.local/candidate_staging/metadata_flat/` (no actor_id/hand_side/region_id)
- 1000 candidate videos on S3, 924 source videos on S3
- Model `llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL` available locally (Q4, ~8GB RAM)

### What is missing locally

- `hand_state.joblib`, `shelf_state.joblib` (classifier weights on remote server)
- `track_a_features/`, `track_a_artifacts/` directories
- Source videos downloaded (none in `.local/source_videos/`)
- Trained VideoMAE checkpoint (`models/track_b1/`)
- Qwen server running (targets `localhost:8080`)

### What is on S3 (`chillnbite-cameras/anon/`)

- 924 source videos (smallest ~9MB, largest ~2.9GB)
- 1000 candidate videos
- 75 clips with candidate metadata JSONs
- 12 clips with VLM-annotated events (41 events: 25 pickups, 16 putdowns)
- 1000 normalized VLM annotation JSONs (~1KB each)
- **No** classifier artifacts, no feature datasets, no trained checkpoints

---

## Layer 1A (Track A) — Inference Pipeline

**Flow:** Candidates + pose tracks → MobileNetV3 feature extraction → hand-state/shelf-state classifiers → state machine → deduplicated canonical predictions.

**Blocker:** `TrackAInferencePipeline.run()` loads `hand_state.joblib` and `shelf_state.joblib` at inference.py:1395-1402. Missing files → crash.

### Option A: Train classifiers locally from scratch (recommended for quality)

1. Download the 12 clips with VLM events from S3 (~2-3GB)
2. Download candidate videos for those clips (~500 videos, ~5GB)
3. Run `dataset_builder.build_feature_dataset()` or `reviewed_dataset.build_reviewed_dataset()` to extract MobileNetV3 embeddings and build `feature_dataset.parquet` with labels from VLM events
4. Run `hand_state.py` and `shelf_state.py` to train logistic regression classifiers → produces `hand_state.joblib`, `shelf_state.joblib`
5. Run full inference pipeline on 1-2 clips as smoke test

**Pros:** Real predictions, end-to-end validation.
**Cons:** Requires GPU or slow CPU embedding extraction, ~5-10GB downloads, hours of work.

### Option B: Skip classifiers, use heuristic fallback (fastest)

1. Create stub `hand_state.joblib` / `shelf_state.joblib` that return uniform probabilities (0.33 each class)
2. The state machine will still run but produce mostly "uncertain" (no events emitted)
3. Validates the pipeline wiring: data flow, state machine, dedup, output format
4. Use existing `clip_person_clear` data (15 candidates, 543 pose obs) — no downloads

**Pros:** ~10 minutes, no downloads, validates all pipeline wiring.
**Cons:** Predictions will be empty or noise (classifiers can't distinguish classes).

### Option C: Use MobileNetV3 embeddings directly as evidence

1. Extract embeddings locally (no classifier weights needed)
2. Skip sklearn classifiers, pass embedding raw distances to state machine as evidence
3. Validates feature extraction + state machine wiring
4. Requires downloading source videos for pose extraction (~2-3GB)

**Pros:** Real feature extraction, no classifier training.
**Cons:** Medium effort, needs source video downloads.

---

## Layer 1B (Track B1) — VideoMAE Window Classifier

**Flow:** Candidates + pose tracks → sliding window frame extraction → VideoMAE encoder + classification head → temporal smoothing + peak detection → canonical predictions.

**Blocker:** `load_checkpoint()` in `videomae_classifier.py:483` requires a `.pth` file. The VideoMAE encoder (`MCG-NJU/videomae-small`) downloads from HuggingFace automatically, but the trained classification head weights are missing.

### Option A: Train from scratch locally (heavy)

1. Download 12 clips with VLM events from S3
2. Build training dataset using `dataset.py` logic (extract windows from candidate videos, label from VLM events)
3. Train VideoMAE model locally (requires GPU, ~50 epochs, likely hours)
4. Run inference on 1-2 clips

**Pros:** Real trained model.
**Cons:** Requires GPU, hours of training time. Not recommended without GPU on laptop.

### Option B: Frozen VideoMAE encoder with random classification head (lightest)

1. VideoMAE encoder downloads automatically from HuggingFace (no local weights needed)
2. Create a random classification head, run inference with raw encoder outputs
3. Apply a simple heuristic: if mean VideoMAE embedding magnitude exceeds threshold → event candidate
4. Validates full pipeline: window generation → frame extraction → model forward → smoothing → peak detection → output format
5. No training needed, but predictions will be random noise
6. Requires downloading source/candidate videos for frame extraction (~2-3GB)

**Pros:** Validates all pipeline wiring, no training.
**Cons:** Predictions are noise, needs video downloads.

### Option C: Pre-trained VideoMAE as feature extractor + simple heuristic

1. Load frozen VideoMAE encoder, extract embeddings for each window
2. Use cosine similarity to positive/negative example embeddings (compute from a few manually inspected candidates)
3. Simple threshold-based classification instead of learned head
4. Validates pipeline without training

**Pros:** Better than random, no training.
**Cons:** Medium effort, needs video downloads.

---

## Layer 2 (VLM Verification)

**Flow:** Layer 1 predictions OR active spans → window generation → frame extraction → Qwen VLM inference → validated predictions → merge + evaluation.

**Blocker:** `QwenClientConfig.base_url = "http://localhost:8080"` — needs a running Qwen server. No local model weights for Qwen3.6.

### Option A: Run Qwen locally with llama.cpp (complete test)

1. Start a llama.cpp server locally serving `llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL` on port 8080
2. Download 1-2 source clips from S3
3. Generate windows from active spans → render frames → call Qwen → get predictions
4. Full end-to-end validation of Layer 2

**Pros:** Most complete test, real VLM output.
**Cons:** Requires ~8GB RAM, a few minutes to start server, needs video downloads.

### Option B: Mock VLM responses for pipeline validation (fastest)

1. Create a `MockQwenClient` that returns deterministic fake `Layer2WindowResponse` objects
2. Run window generation + frame rendering (needs video file) + mock VLM call
3. Validates full Layer 2 pipeline: windows → frames → mock inference → schema validation → merge → canonical output
4. No model needed, no downloads needed (use local `person_clear.mp4`)
5. ~10 minutes, validates wiring only

**Pros:** Fastest, no downloads, no model.
**Cons:** No real VLM inference, only validates wiring.

### Option C: Use existing VLM annotations as ground truth input

1. Download the 1000 normalized JSONs from S3 (~1MB total)
2. Parse them as "predictions" (they already have events with labels, timestamps)
3. Run `merge_predictions.merge_predictions()` + `evaluate_layer2()` to validate evaluation pipeline
4. Tests Layer 2 evaluation without inference

**Pros:** Minimal downloads, validates evaluation pipeline.
**Cons:** Skips VLM inference entirely, only tests merge + evaluation.

---

## Recommended Execution Order

If you want to test each layer "as good as possible" with minimal data:

1. **Layer 1A: Option B** (stub classifiers) — validates full data flow with zero downloads. ~10 min.
2. **Layer 1B: Option B** (frozen encoder + random head) — validates window extraction + model forward + peak detection. ~30 min (includes downloading 1-2 source clips).
3. **Layer 2: Option A** (run Qwen locally) — if you have 8GB+ RAM, this gives the most complete test. Otherwise Option B (mock).
