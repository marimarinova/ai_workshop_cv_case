# GPU Pipeline Optimization Analysis

## Date

2026-06-23

## Current State

**Data:** 319 source videos, 208 GB total. 67 ledger entries. 21 videos in skip list (all largest: 1.1–3.6 GB, 952–1588 s).

**Latest run (`local_e3bd3aae7954`):** 34 selected → 8 completed, 52 failed (26 unique × 2 error slots). 85 total candidates produced.

**Run history today:** Consistent 20–35% completion rate; remainder timeout.

### Completed vs. Failed Video Profiles

| Metric | Completed (8) | Timeout (16) | WQ Deadlock (10) |
|--------|--------------|--------------|-----------------|
| Avg duration | 159 s | 476 s | 315 s |
| Avg size | 349 MB | 1 019 MB | 688 MB |
| Range duration | 66–268 s | 244–1 260 s | 140–588 s |
| Range size | 144–585 MB | 512–2 605 MB | 316–1 303 MB |

Completed videos are consistently under 300 s / 600 MB. Once videos exceed ~400 s, failure becomes the norm.

## Two Failure Modes

### 1. CLI Subprocess Timeout (1800 s)

Hardcoded in `worker.py:187` (`subprocess.run(timeout=1800)`) and `coordinator.py:201` (`future.result(timeout=1800)`).

Affects 16 videos (244–1 260 s). No clean duration boundary — even 244 s / 512 MB videos fail because of GPU contention with 8 concurrent workers.

### 2. Worker Queue Timeout (10 s per frame)

`RuntimeError: Worker N error: timeout was…` from `frame_pipeline.py:472` — the internal `frame_queue.get(timeout=10.0)` in `DecoderPool.get_frame()`.

Affects 10 videos (140–588 s). The 10 s per-frame timeout fires when the GPU consumer is starved by contention. The `FrameReorderer` waits for frame N, but decoder workers are blocked because the queue is full. This is a deadlock under GPU contention.

## Why 1800 s Is Not Enough

Processing requires two sequential GPU stages per video:

1. **Triage** at `target_fps: 1.0` → for a 600 s video: 600 YOLO frames
2. **Propose** at `target_fps: 8.0` on active spans → 200–800 more frames

With 8 GPU workers competing:

- Single YOLO inference at 640×640: ~80–150 ms solo
- With 8 workers (memory-bandwidth bound): ~600–1 200 ms per inference
- 600 triage frames × 1 s = **600 s wall time** (triage only)
- Propose adds **200–500 s**
- Total: **800–1 100 s** for a "moderate" 600 s video
- For 1 000 s+ videos: **1 800 s+** even without overhead

The 1 800 s timeout covers the entire triage subprocess. With GPU contention, effective processing rate drops below the timeout threshold for most videos over 300 s.

## Optimization Levers

### Config-Only Changes (Implemented)

| Setting | Before | After | Location | Rationale |
|---------|--------|-------|----------|-----------|
| `gpu_workers` | 8 | **5** | `process_all_local.sh` | 8 workers saturates GPU memory bandwidth. 5 gives each worker ~60% more bandwidth. Net throughput similar or better because inference is bandwidth-bound at 640 px. |
| `target_fps` (triage) | 1.0 | **0.5** | `configs/candidates.yaml` | Person tracking at 0.5 fps still adequate for 2 s+ events. Halves triage frame count, doubling throughput for long videos. |
| `pipeline_frame_timeout_s` | 10 s | **30 s** | `configs/candidates.yaml` | 10 s per-frame timeout too aggressive under contention. 30 s accommodates GPU starve without masking real deadlocks. |
| `pipeline_n_decoders` | 2 | **4** | `configs/candidates.yaml` | More decoder workers keeps the queue fed when GPU consumer slows. CPU-bound, so low cost. |
| `pipeline_queue_depth` | 16 | **32** | `configs/candidates.yaml` | Larger buffer absorbs decoder-consumer speed mismatch under contention. |

### Code Changes (Implemented)

| Setting | Before | After | Location | Rationale |
|---------|--------|-------|----------|-----------|
| Triage timeout | 1 800 s | **3 600 s** | `worker.py:187` | Covers 2× longer videos. Combined with fewer workers, enables processing of 1 000 s+ clips. |
| Propose timeout | 1 800 s | **3 600 s** | `worker.py:211` | Same rationale — propose stage also benefits from longer budget. |
| Coordinator timeout | 1 800 s | **3 600 s** | `coordinator.py:201` | Outer envelope for GPU worker futures. |

### Architecture Changes (Not Implemented — Future Work)

| Change | Description |
|--------|-------------|
| **Adaptive FPS** | Set `target_fps` based on video duration: 1 fps for < 300 s, 0.5 fps for > 600 s. Requires code in `PersonTracker.__post_init__`. |
| **Priority queue** | Process shorter videos first when GPU workers limited. Guarantees progress instead of all workers stuck on large videos. |
| **Per-stage timeout** | Separate timeout for triage vs propose, so a hung propose does not waste the full budget. |
| **GPU load monitoring** | Dynamic worker count based on actual GPU utilization rather than fixed count. |

## Expected Outcome

With `gpu_workers=5`, `target_fps=0.5`, timeout=3 600 s, `pipeline_frame_timeout_s=30 s`:

- 5 workers × 3 600 s budget = can process ~5 videos up to 1 500 s concurrently
- Reduced contention → each inference ~40–50% faster
- Worker-queue deadlocks eliminated by 30 s per-frame timeout
- Triage frame count halved by 0.5 fps
- Estimated throughput: **15–25 videos/hour** vs current ~8/hour, with higher success rate on large videos

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 0.5 fps misses short events (< 2 s) | Medium | Review candidate counts; if too few, increase back to 0.75 fps. |
| 5 workers underutilize GPU | Low | Monitor GPU utilization; adjust to 4 or 6 as needed. |
| 3 600 s timeout masks hangs | Low | Worker-queue timeout (30 s) still catches per-frame deadlocks. |
| Larger queue depth uses more RAM | Low | 32 frames at 640×640 = ~50 MB per worker, negligible. |
