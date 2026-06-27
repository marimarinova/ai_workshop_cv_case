"""Track B1: VideoMAE fixed-window classifier for pickup/putdown detection."""

from pickup_putdown.layer1.track_b1.dataset import (
    LABEL_BACKGROUND,
    LABEL_NAMES,
    LABEL_PICKUP,
    LABEL_PUTDOWN,
    InferenceWindow,
    TrackB1Dataset,
    WindowConfig,
    WindowSample,
    build_window_manifest,
    create_dataloaders,
    generate_inference_windows,
    generate_inference_windows_for_candidate,
    generate_sliding_windows,
    get_label_weights,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (
    ClassificationHead,
    VideoMAEClassifier,
    create_model,
    load_checkpoint,
    predict_batch,
    save_checkpoint,
)
from pickup_putdown.layer1.track_b1.train import (
    EarlyStopping,
    EpochMetrics,
    TrainConfig,
    compute_metrics,
    run_tiny_overfit_test,
    train,
    train_one_epoch,
    validate,
)
from pickup_putdown.layer1.track_b1.inference import (
    EventPrediction,
    InferenceConfig,
    ScoreRegion,
    WindowPrediction,
    create_event_predictions,
    detect_score_peaks,
    infer_all_candidates,
    infer_candidate,
    merge_same_type_regions,
    predict_windows,
    save_predictions,
    smooth_predictions,
)

__all__ = [
    # Dataset
    "TrackB1Dataset",
    "WindowConfig",
    "WindowSample",
    "InferenceWindow",
    "build_window_manifest",
    "create_dataloaders",
    "get_label_weights",
    # Window generation (shared by training & inference)
    "generate_sliding_windows",
    "generate_inference_windows",
    "generate_inference_windows_for_candidate",
    # Labels
    "LABEL_BACKGROUND",
    "LABEL_PICKUP",
    "LABEL_PUTDOWN",
    "LABEL_NAMES",
    # Model
    "ClassificationHead",
    "VideoMAEClassifier",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
    "predict_batch",
    # Training
    "TrainConfig",
    "EpochMetrics",
    "EarlyStopping",
    "train",
    "train_one_epoch",
    "validate",
    "compute_metrics",
    "run_tiny_overfit_test",
    # Inference
    "InferenceConfig",
    "WindowPrediction",
    "ScoreRegion",
    "EventPrediction",
    "predict_windows",
    "smooth_predictions",
    "detect_score_peaks",
    "merge_same_type_regions",
    "create_event_predictions",
    "infer_candidate",
    "infer_all_candidates",
    "save_predictions",
]
