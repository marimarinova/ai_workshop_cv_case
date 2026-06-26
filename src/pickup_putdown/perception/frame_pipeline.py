"""Multiprocessed frame decoding pipeline for efficient video processing.

This module provides infrastructure for parallel frame decoding to improve
GPU utilization during YOLO inference. Multiple decoder workers decode frames
in parallel and write them to shared memory, while the main process consumes
frames in order for sequential inference (required by ByteTrack).

Architecture:
    +------------------+
    |  DECODER 0       |--+
    |  (Worker Process)|  |    +------------------+       +------------------+
    |  Frames 0,2,4,...|  +--> |  SHARED MEMORY   |       |  CONSUMER        |
    +------------------+  |    |  Ring Buffer     |  -->  |  (Main Process)  |
                          |    |  16 frame slots  |       |  YOLO + ByteTrack|
    +------------------+  |    |  ~20MB (resized) |       |  GPU inference   |
    |  DECODER 1       |--+    +------------------+       +------------------+
    |  (Worker Process)|
    |  Frames 1,3,5,...|
    +------------------+
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType
    from queue import Queue

logger = logging.getLogger(__name__)

# Sentinel value for signaling worker completion
SENTINEL = None


@dataclass(frozen=True)
class FrameMetadata:
    """Lightweight metadata for a decoded frame (picklable for queue transfer).

    Attributes
    ----------
    sample_index : int
        Index in the sample_frames list (0-based).
    source_frame_index : int
        Original frame number in the video.
    timestamp_s : float
        Timestamp in seconds from the start of the video.
    slot_index : int
        Index of the shared memory slot containing the frame data.
    worker_id : int
        ID of the decoder worker that produced this frame.
    """

    sample_index: int
    source_frame_index: int
    timestamp_s: float
    slot_index: int
    worker_id: int


class SharedFrameBuffer:
    """Shared memory ring buffer for zero-copy frame passing between processes.

    This class manages a shared memory region that holds multiple frame slots.
    Each slot can hold a single frame at the configured resolution.

    Parameters
    ----------
    n_slots : int
        Number of frame slots in the ring buffer.
    frame_height : int
        Height of frames in pixels.
    frame_width : int
        Width of frames in pixels.
    n_channels : int
        Number of color channels (default: 3 for BGR).
    create : bool
        If True, create new shared memory. If False, attach to existing.
    name : str | None
        Optional name for the shared memory segment. If None, auto-generated.
    """

    def __init__(
        self,
        n_slots: int,
        frame_height: int,
        frame_width: int,
        n_channels: int = 3,
        create: bool = True,
        name: str | None = None,
    ) -> None:
        self.n_slots = n_slots
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.n_channels = n_channels
        self.frame_shape = (frame_height, frame_width, n_channels)
        self.frame_size = frame_height * frame_width * n_channels
        self.total_size = self.frame_size * n_slots

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=self.total_size)
            logger.debug(
                "Created shared memory buffer: name=%s, size=%d bytes, slots=%d",
                self._shm.name,
                self.total_size,
                n_slots,
            )
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._shm = shared_memory.SharedMemory(name=name)

        self._buffer = np.ndarray(
            (n_slots, frame_height, frame_width, n_channels),
            dtype=np.uint8,
            buffer=self._shm.buf,
        )

    @property
    def name(self) -> str:
        """Return the shared memory segment name."""
        return self._shm.name

    def write_frame(self, slot_index: int, frame: np.ndarray) -> None:
        """Write a frame to the specified slot.

        Parameters
        ----------
        slot_index : int
            Index of the slot to write to.
        frame : np.ndarray
            Frame data (must match frame_shape).

        Raises
        ------
        ValueError
            If slot_index is out of range or frame shape doesn't match.
        """
        if slot_index < 0 or slot_index >= self.n_slots:
            raise ValueError(f"slot_index {slot_index} out of range [0, {self.n_slots})")
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match expected {self.frame_shape}"
            )

        np.copyto(self._buffer[slot_index], frame)

    def read_frame(self, slot_index: int) -> np.ndarray:
        """Read a frame from the specified slot.

        Parameters
        ----------
        slot_index : int
            Index of the slot to read from.

        Returns
        -------
        np.ndarray
            Copy of the frame data.
        """
        if slot_index < 0 or slot_index >= self.n_slots:
            raise ValueError(f"slot_index {slot_index} out of range [0, {self.n_slots})")

        return self._buffer[slot_index].copy()

    def close(self) -> None:
        """Close the shared memory handle (does not unlink)."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink (delete) the shared memory segment."""
        self._shm.unlink()


def _decoder_worker(
    worker_id: int,
    video_path: str,
    sample_frames: list[int],
    source_fps: float,
    frame_queue: "Queue[FrameMetadata | None]",
    slot_queue: "Queue[int]",
    error_queue: "Queue[tuple[int, Exception]]",
    shutdown_event: "EventType",
    shm_name: str,
    frame_height: int,
    frame_width: int,
    resize_frames: bool,
) -> None:
    """Worker process function for decoding frames.

    This function runs in a separate process and decodes frames from the video,
    optionally resizes them, and writes them to shared memory.

    Parameters
    ----------
    worker_id : int
        Unique identifier for this worker.
    video_path : str
        Path to the video file.
    sample_frames : list[int]
        List of source frame indices this worker should decode.
    source_fps : float
        Source video FPS for timestamp calculation.
    frame_queue : Queue[FrameMetadata | None]
        Queue for sending frame metadata to consumer.
    slot_queue : Queue[int]
        Queue of available slot indices (backpressure mechanism).
    error_queue : Queue[tuple[int, Exception]]
        Queue for reporting errors to the main process.
    shutdown_event : Event
        Event to signal graceful shutdown.
    shm_name : str
        Name of the shared memory segment.
    frame_height : int
        Target frame height (for resize or validation).
    frame_width : int
        Target frame width (for resize or validation).
    resize_frames : bool
        Whether to resize frames to target dimensions.
    """
    import cv2

    try:
        # Attach to shared memory
        buffer = SharedFrameBuffer(
            n_slots=0,  # Not used when attaching
            frame_height=frame_height,
            frame_width=frame_width,
            create=False,
            name=shm_name,
        )
        # We need to recreate the buffer with proper n_slots
        # Actually, we just need to attach to the shared memory directly
        buffer._shm.close()  # Close the temporary connection

        shm = shared_memory.SharedMemory(name=shm_name)
        # Determine n_slots from total size
        frame_size = frame_height * frame_width * 3
        n_slots = shm.size // frame_size

        buffer_array = np.ndarray(
            (n_slots, frame_height, frame_width, 3),
            dtype=np.uint8,
            buffer=shm.buf,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        logger.debug(
            "Worker %d started: %d frames to decode from %s",
            worker_id,
            len(sample_frames),
            video_path,
        )

        for idx, src_frame_idx in enumerate(sample_frames):
            if shutdown_event.is_set():
                logger.debug("Worker %d: shutdown requested, exiting", worker_id)
                break

            # Get an available slot (blocks if buffer is full)
            try:
                slot_index = slot_queue.get(timeout=10.0)
            except Exception:
                if shutdown_event.is_set():
                    break
                raise TimeoutError(f"Worker {worker_id}: timeout waiting for slot")

            # Seek and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Worker %d: failed to read frame %d",
                    worker_id,
                    src_frame_idx,
                )
                # Return slot and continue
                slot_queue.put(slot_index)
                continue

            # Resize if needed
            if resize_frames:
                frame = cv2.resize(frame, (frame_width, frame_height))

            # Write to shared memory
            np.copyto(buffer_array[slot_index], frame)

            # Calculate timestamp
            timestamp_s = src_frame_idx / source_fps

            # Send metadata to consumer
            # idx here is the local index within this worker's sample_frames
            # We need to compute the global sample_index
            # This is passed from DecoderPool which interleaves correctly
            metadata = FrameMetadata(
                sample_index=idx,  # Will be remapped by DecoderPool
                source_frame_index=src_frame_idx,
                timestamp_s=timestamp_s,
                slot_index=slot_index,
                worker_id=worker_id,
            )
            frame_queue.put(metadata)

        cap.release()
        shm.close()

        # Signal completion
        frame_queue.put(SENTINEL)
        logger.debug("Worker %d completed successfully", worker_id)

    except Exception as e:
        logger.exception("Worker %d error", worker_id)
        error_queue.put((worker_id, e))
        frame_queue.put(SENTINEL)


class DecoderPool:
    """Manages multiple decoder workers with interleaved frame assignment.

    This class handles starting and stopping decoder worker processes,
    distributing frames across workers, and coordinating shared memory.

    Parameters
    ----------
    n_workers : int
        Number of decoder worker processes.
    queue_depth : int
        Number of slots in the shared memory ring buffer.
    frame_height : int
        Target frame height.
    frame_width : int
        Target frame width.
    resize_frames : bool
        Whether workers should resize frames.
    """

    def __init__(
        self,
        n_workers: int = 2,
        queue_depth: int = 16,
        frame_height: int = 640,
        frame_width: int = 640,
        resize_frames: bool = True,
    ) -> None:
        self.n_workers = n_workers
        self.queue_depth = queue_depth
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.resize_frames = resize_frames

        self._workers: list[mp.Process] = []
        self._shared_buffer: SharedFrameBuffer | None = None
        self._frame_queue: mp.Queue | None = None
        self._slot_queue: mp.Queue | None = None
        self._error_queue: mp.Queue | None = None
        self._shutdown_event: mp.Event | None = None
        self._sample_index_maps: list[dict[int, int]] = []
        self._active = False

    def start(
        self,
        video_path: Path,
        sample_frames: list[int],
        source_fps: float,
    ) -> None:
        """Start decoder workers.

        Parameters
        ----------
        video_path : Path
            Path to the video file.
        sample_frames : list[int]
            List of source frame indices to decode.
        source_fps : float
            Source video FPS.
        """
        if self._active:
            raise RuntimeError("DecoderPool is already active")

        # Create shared resources
        self._shared_buffer = SharedFrameBuffer(
            n_slots=self.queue_depth,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
        )
        self._frame_queue = mp.Queue()
        self._slot_queue = mp.Queue()
        self._error_queue = mp.Queue()
        self._shutdown_event = mp.Event()

        # Pre-fill slot queue with available slots
        for i in range(self.queue_depth):
            self._slot_queue.put(i)

        # Split frames across workers (interleaved for better distribution)
        self._sample_index_maps = []
        for i in range(self.n_workers):
            worker_frames = sample_frames[i :: self.n_workers]
            # Map worker-local index to global sample index
            index_map = {
                local_idx: i + local_idx * self.n_workers
                for local_idx in range(len(worker_frames))
            }
            self._sample_index_maps.append(index_map)

            worker = mp.Process(
                target=_decoder_worker,
                args=(
                    i,  # worker_id
                    str(video_path),
                    worker_frames,
                    source_fps,
                    self._frame_queue,
                    self._slot_queue,
                    self._error_queue,
                    self._shutdown_event,
                    self._shared_buffer.name,
                    self.frame_height,
                    self.frame_width,
                    self.resize_frames,
                ),
            )
            worker.start()
            self._workers.append(worker)

        self._active = True
        logger.info(
            "Started %d decoder workers for %d frames",
            self.n_workers,
            len(sample_frames),
        )

    def get_frame(self, timeout: float = 10.0) -> tuple[np.ndarray, FrameMetadata] | None:
        """Get the next decoded frame.

        This method does NOT guarantee frame ordering. Use FrameReorderer
        for ordered consumption.

        Parameters
        ----------
        timeout : float
            Timeout in seconds for waiting on the queue.

        Returns
        -------
        tuple[np.ndarray, FrameMetadata] | None
            Frame data and metadata, or None if a worker has finished.

        Raises
        ------
        RuntimeError
            If a worker encountered an error.
        """
        if not self._active:
            raise RuntimeError("DecoderPool is not active")

        # Check for errors
        if not self._error_queue.empty():
            worker_id, error = self._error_queue.get_nowait()
            raise RuntimeError(f"Worker {worker_id} error: {error}") from error

        try:
            metadata = self._frame_queue.get(timeout=timeout)
        except Exception as e:
            raise TimeoutError(f"Timeout waiting for frame: {e}") from e

        if metadata is SENTINEL:
            return None

        # Remap sample_index from worker-local to global
        global_sample_idx = self._sample_index_maps[metadata.worker_id].get(
            metadata.sample_index, metadata.sample_index
        )
        metadata = FrameMetadata(
            sample_index=global_sample_idx,
            source_frame_index=metadata.source_frame_index,
            timestamp_s=metadata.timestamp_s,
            slot_index=metadata.slot_index,
            worker_id=metadata.worker_id,
        )

        # Read frame from shared memory
        frame = self._shared_buffer.read_frame(metadata.slot_index)

        return frame, metadata

    def return_slot(self, slot_index: int) -> None:
        """Return a slot to the pool for reuse.

        Parameters
        ----------
        slot_index : int
            Index of the slot to return.
        """
        if self._slot_queue is not None:
            self._slot_queue.put(slot_index)

    def stop(self) -> None:
        """Stop all decoder workers and clean up resources."""
        if not self._active:
            return

        logger.debug("Stopping decoder pool")

        # Signal workers to stop
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning("Worker did not stop gracefully, terminating")
                worker.terminate()
                worker.join(timeout=1.0)

        # Clean up shared memory
        if self._shared_buffer is not None:
            try:
                self._shared_buffer.close()
                self._shared_buffer.unlink()
            except Exception as e:
                logger.warning("Error cleaning up shared memory: %s", e)

        # Clean up queues
        for queue in [self._frame_queue, self._slot_queue, self._error_queue]:
            if queue is not None:
                try:
                    queue.close()
                    queue.join_thread()
                except Exception:
                    pass

        self._workers = []
        self._shared_buffer = None
        self._frame_queue = None
        self._slot_queue = None
        self._error_queue = None
        self._shutdown_event = None
        self._active = False

        logger.debug("Decoder pool stopped")

    def __enter__(self) -> "DecoderPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class FrameReorderer:
    """Reorders frames received out-of-order from multiple workers.

    Since multiple workers decode frames in parallel, frames may arrive
    out of order. This class buffers out-of-order frames and yields them
    in the correct sequential order.

    Parameters
    ----------
    total_frames : int
        Total number of frames expected.
    """

    def __init__(self, total_frames: int) -> None:
        self.total_frames = total_frames
        self.expected_sample_idx = 0
        self.pending: dict[int, tuple[np.ndarray, FrameMetadata]] = {}
        self.completed_workers: set[int] = set()

    def add_frame(
        self, frame: np.ndarray, metadata: FrameMetadata
    ) -> list[tuple[np.ndarray, FrameMetadata]]:
        """Add a frame and return any frames that are now in order.

        Parameters
        ----------
        frame : np.ndarray
            Frame data.
        metadata : FrameMetadata
            Frame metadata.

        Returns
        -------
        list[tuple[np.ndarray, FrameMetadata]]
            List of (frame, metadata) tuples that are now in order.
        """
        ready = []

        # If this is the expected frame, add it and drain pending
        if metadata.sample_index == self.expected_sample_idx:
            ready.append((frame, metadata))
            self.expected_sample_idx += 1

            # Drain any pending frames that are now in order
            while self.expected_sample_idx in self.pending:
                pending_frame, pending_meta = self.pending.pop(self.expected_sample_idx)
                ready.append((pending_frame, pending_meta))
                self.expected_sample_idx += 1
        else:
            # Buffer out-of-order frame
            self.pending[metadata.sample_index] = (frame, metadata)

        return ready

    def mark_worker_done(self, worker_id: int) -> None:
        """Mark a worker as completed.

        Parameters
        ----------
        worker_id : int
            ID of the completed worker.
        """
        self.completed_workers.add(worker_id)

    def is_complete(self, n_workers: int) -> bool:
        """Check if all frames have been received.

        Parameters
        ----------
        n_workers : int
            Total number of workers.

        Returns
        -------
        bool
            True if all frames have been received.
        """
        return self.expected_sample_idx >= self.total_frames and len(self.pending) == 0

    @property
    def n_pending(self) -> int:
        """Number of buffered out-of-order frames."""
        return len(self.pending)
