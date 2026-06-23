# Multiprocessed Frame Decoding Pipeline

## Overview

This document describes the multiprocessed frame decoding optimization implemented for Task 3 (person triage/tracking). The optimization parallelizes CPU-bound frame decoding to improve GPU utilization during YOLO inference.

## The Problem: CPU-GPU Bottleneck

### Before Optimization

The original `PersonTracker` processed frames sequentially:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEQUENTIAL PROCESSING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Decode Frame 0   YOLO    Decode Frame 1   YOLO    Decode Frame 2   YOLO   │
│  ├────175ms────┤ ├─15ms─┤ ├────175ms────┤ ├─15ms─┤ ├────175ms────┤ ├─15ms─┤│
│                                                                             │
│  GPU: ░░░░░░░░░░░ ████░░░░░░░░░░░░░ ████░░░░░░░░░░░░░ ████                  │
│       (idle)     (busy)   (idle)   (busy)   (idle)   (busy)                 │
│                                                                             │
│  GPU Utilization: ~8%                                                       │
│  Time per frame: ~190ms                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Bottleneck Analysis:**
- 4K H.264 frame decoding: **~150-200ms per frame** (CPU-bound)
- YOLO inference: **~10-20ms per frame** (GPU-bound)
- GPU sits idle **~90% of the time** waiting for frame decoding

### Performance Impact

For a typical 152-second video at 1 FPS sampling (435 frames):
- Sequential decode time: 435 × 175ms = **76 seconds** just for decoding
- YOLO inference time: 435 × 15ms = **6.5 seconds**
- Total: **~83 seconds** minimum, often 15+ minutes with overhead

## The Solution: Multi-Producer Consumer Architecture

### After Optimization

The `PipelinedPersonTracker` decouples frame decoding from inference:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINED PROCESSING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  DECODER 0       │──┐                                                    │
│  │  (Worker Process)│  │     ┌──────────────────┐    ┌──────────────────┐  │
│  │  Frames 0,2,4,...│  ├────▶│  SHARED MEMORY   │───▶│  CONSUMER        │  │
│  └──────────────────┘  │     │  Ring Buffer     │    │  (Main Process)  │  │
│                        │     │  8 frame slots   │    │  YOLO + ByteTrack│  │
│  ┌──────────────────┐  │     │  ~200MB (4K)     │    │  GPU inference   │  │
│  │  DECODER 1       │──┘     └──────────────────┘    └──────────────────┘  │
│  │  (Worker Process)│                                                       │
│  │  Frames 1,3,5,...│                                                       │
│  └──────────────────┘                                                       │
│                                                                             │
│  GPU: ████████████████████████████████████████████████████████████████     │
│       (continuously busy - frames always ready in queue)                    │
│                                                                             │
│  GPU Utilization: ~17%+                                                     │
│  Effective decode time: ~87.5ms (175ms ÷ 2 workers)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Multiple Decoder Workers**: 2 workers decode frames in parallel, effectively halving decode time
2. **Interleaved Frame Assignment**: Worker 0 handles frames 0,2,4,... and Worker 1 handles frames 1,3,5,...
3. **Shared Memory**: Zero-copy frame passing between processes using `multiprocessing.shared_memory`
4. **Ring Buffer**: 8 frame slots provide backpressure - decoders block when buffer is full
5. **Frame Reordering**: Consumer reorders frames since they may arrive out-of-order from parallel workers

### Why ByteTrack Requires Sequential Processing

ByteTrack maintains internal state for tracking objects across frames. Processing frames out of order would cause:
- Inconsistent track IDs
- Lost tracks
- Incorrect temporal associations

**Solution**: Parallel decoding, sequential inference. The `FrameReorderer` buffers out-of-order frames and yields them in the correct sequence.

## Architecture Components

### File Structure

```
src/pickup_putdown/perception/
├── frame_pipeline.py      # Core pipeline infrastructure
├── pipelined_tracker.py   # PipelinedPersonTracker class
└── person_tracker.py      # Original PersonTracker (base class)
```

### Core Classes

#### `SharedFrameBuffer`
Manages shared memory ring buffer for zero-copy frame passing.

```python
buffer = SharedFrameBuffer(
    n_slots=8,           # Number of frame slots
    frame_height=2160,   # 4K height
    frame_width=3840,    # 4K width
)
buffer.write_frame(slot_index=0, frame=frame_data)
frame = buffer.read_frame(slot_index=0)
```

#### `DecoderPool`
Manages multiple decoder worker processes.

```python
with DecoderPool(n_workers=2, queue_depth=8) as pool:
    pool.start(video_path, sample_frames, source_fps)

    while True:
        result = pool.get_frame(timeout=30.0)
        if result is None:  # Worker finished
            break
        frame, metadata = result
        # Process frame...
        pool.return_slot(metadata.slot_index)
```

#### `FrameReorderer`
Buffers out-of-order frames and yields them sequentially.

```python
reorderer = FrameReorderer(total_frames=435)

# Frames may arrive as: 1, 0, 3, 2, 5, 4, ...
ready_frames = reorderer.add_frame(frame, metadata)
# Returns frames in order: 0, 1, 2, 3, ...
```

#### `PipelinedPersonTracker`
Drop-in replacement for `PersonTracker`.

```python
from pickup_putdown.perception.pipelined_tracker import PipelinedPersonTracker

tracker = PipelinedPersonTracker(
    video_path=video_path,
    triage_cfg=triage_cfg,
    use_pipeline=True,  # Enable pipelining (default)
)
observations, summaries = tracker.run()  # Same API
```

## Configuration

Pipeline settings in `TriageConfig` (`src/pickup_putdown/config.py`):

```python
class TriageConfig(BaseModel):
    # ... existing settings ...

    # Pipeline configuration
    pipeline_enabled: bool = True           # Enable/disable pipelining
    pipeline_queue_depth: int = 8           # Ring buffer slots
    pipeline_n_decoders: int = 2            # Number of decoder workers
    pipeline_resize_frames: bool = False    # Keep original resolution
    pipeline_frame_size: tuple = (640, 640) # Only if resize_frames=True
    pipeline_frame_timeout_s: float = 30.0  # Timeout for frame retrieval
```

### Environment Variable Override

Disable pipelining via environment variable:
```bash
PICKUP_PUTDOWN_TRIAGE_PIPELINE_ENABLED=false make task-3 VIDEO=...
```

## Performance Results

### Test Configuration
- **Video**: 4K H.264, 3840x2160, 20 FPS, 152 seconds
- **Sampling**: 1 FPS (435 frames)
- **Hardware**: NVIDIA GPU with CUDA

### Benchmark Results

| Metric | Sequential | Pipelined (2 workers) | Improvement |
|--------|------------|----------------------|-------------|
| Task 3 time | ~15 min | ~4 min | **~73% faster** |
| Tasks 3-5 time | ~19 min | ~7 min | **~63% faster** |
| GPU utilization | ~8% | ~17% | **2x better** |
| Results | baseline | ✅ identical | exact match |

### Output Verification

The pipelined version produces **identical results** to the sequential version:

```
=== Task 3: tracks_person.parquet comparison ==

Old run (sequential): 146 rows
New run (pipelined):  146 rows

Bounding boxes: ✅ Exact match (max_diff=0.000000)
Track IDs:      ✅ Exact match [1, 2, 3, 8, 13, 15]
Stable tracks:  ✅ Exact match [3, 8, 13]
Confidence:     ✅ Exact match
```

## Usage Examples

### Basic Usage (Default Settings)

```bash
# Pipeline is enabled by default
make task-3 VIDEO=/path/to/video.mp4
```

### Run Full Pipeline

```bash
make tasks-3-5 VIDEO=/path/to/video.mp4 RENDER_PREVIEWS=0
```

### Compare Pipelined vs Sequential

```bash
# With pipeline (default)
time make task-3 VIDEO=/path/to/video.mp4 RUN_ID=pipelined

# Without pipeline
PICKUP_PUTDOWN_TRIAGE_PIPELINE_ENABLED=false \
  time make task-3 VIDEO=/path/to/video.mp4 RUN_ID=sequential

# Compare outputs
python -c "
import pandas as pd
old = pd.read_parquet('.local/task_runs/sequential/task_3/tracks_person.parquet')
new = pd.read_parquet('.local/task_runs/pipelined/task_3/tracks_person.parquet')
print(f'Rows match: {len(old) == len(new)}')
print(f'Data match: {old.equals(new)}')
"
```

### Monitor GPU Utilization

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

## Data Flow

```
1. Main process probes video metadata (resolution, FPS, frame count)
2. Main process computes sample_frames list based on target_fps
3. Main process creates SharedFrameBuffer (8 slots × 3840×2160×3 bytes)
4. Main process starts 2 decoder workers:
   - Decoder 0: handles frames 0, 2, 4, 6, ...
   - Decoder 1: handles frames 1, 3, 5, 7, ...
5. Main process pre-fills slot_queue with [0, 1, 2, ..., 7]

Steady state loop:
┌─────────────────────────────────────────────────────────────────┐
│  Decoders (parallel):                                           │
│    1. Get available slot from slot_queue (blocks if empty)      │
│    2. Seek to frame position in video                           │
│    3. Decode frame from video                                   │
│    4. Write frame to shared memory slot                         │
│    5. Put FrameMetadata on frame_queue                          │
│                                                                 │
│  Consumer (sequential, with reordering):                        │
│    1. Get FrameMetadata from frame_queue                        │
│    2. Add to FrameReorderer (buffers out-of-order frames)       │
│    3. For each frame now in order:                              │
│       a. Read frame from shared memory                          │
│       b. Return slot to slot_queue (allows decoder to reuse)    │
│       c. Run YOLO inference                                     │
│       d. Update ByteTrack state                                 │
│       e. Record observations                                    │
└─────────────────────────────────────────────────────────────────┘

6. Each decoder sends SENTINEL when done
7. Consumer waits for all SENTINELs
8. Main process cleans up shared memory
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Decoder crash | SENTINEL sent, error propagated via error_queue |
| Consumer crash | shutdown_event set, decoders exit gracefully |
| Frame timeout | RuntimeError raised after 30s timeout |
| Memory cleanup | try/finally ensures shm.unlink() always called |

## Limitations and Trade-offs

### Memory Usage

| Configuration | Queue Memory |
|---------------|--------------|
| 4K (3840×2160), 8 slots | ~200 MB |
| 1080p (1920×1080), 8 slots | ~50 MB |
| 640×640 resized, 16 slots | ~20 MB |

### When Pipelining Helps Most

- ✅ High-resolution video (4K, 8K) where decode time >> inference time
- ✅ Long videos with many frames to process
- ✅ GPU with spare capacity

### When Pipelining Helps Less

- ⚠️ Low-resolution video where decode time is already fast
- ⚠️ Very short videos (startup overhead dominates)
- ⚠️ CPU-limited systems where decoders compete for resources

## Testing

Run the pipeline tests:

```bash
python -m pytest tests/test_pipelined_tracker.py -v
```

Test coverage includes:
- SharedFrameBuffer memory management
- FrameReorderer ordering logic
- DecoderPool worker lifecycle
- PipelinedPersonTracker integration
- Configuration handling

## Future Improvements

Potential optimizations not yet implemented:

1. **Task 5 pipelining**: Apply same optimization to `PoseTracker`
2. **Adaptive worker count**: Auto-tune based on CPU cores and video resolution
3. **GPU decode**: Use NVDEC for GPU-accelerated decoding (requires compatible hardware)
4. **Batch inference**: Process multiple frames per YOLO call (requires ByteTrack modifications)
