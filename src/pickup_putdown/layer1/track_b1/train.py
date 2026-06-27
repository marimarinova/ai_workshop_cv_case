"""Track B1 training loop for VideoMAE classifier.

This module provides:
- Training loop with validation
- Tiny overfit test (Gate B) for sanity checking
- Learning rate scheduling with warmup
- Early stopping
- Checkpoint management
- Metrics computation (accuracy, F1, precision, recall)

Training pipeline:
    1. Setup: Load data, create model, optimizer, scheduler
    2. Gate B: Tiny overfit test (verify model can memorize small batch)
    3. Training loop: Train epochs, validate, checkpoint best
    4. Output: Best checkpoint, training history, final metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from pickup_putdown.layer1.track_b1.dataset import (
    LABEL_NAMES,
    TrackB1Dataset,
    WindowConfig,
    build_window_manifest,
    create_dataloaders,
    get_label_weights,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (
    VideoMAEClassifier,
    create_model,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_epochs: int = 2

    # Batch size (should match dataloader)
    batch_size: int = 8

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path(".local/track_b1_checkpoints"))
    save_every_n_epochs: int = 5

    # Tiny overfit test (Gate B)
    tiny_overfit_samples: int = 16
    tiny_overfit_max_steps: int = 200
    tiny_overfit_target_loss: float = 0.1

    # Logging
    log_every_n_steps: int = 10

    # Model config
    model_name: str = "MCG-NJU/videomae-small"
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0
    dropout: float = 0.1

    # Device
    device: str = "auto"


# ============================================================
# METRICS TRACKING
# ============================================================


@dataclass
class EpochMetrics:
    """Metrics for one epoch."""

    loss: float
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_macro: float
    recall_macro: float
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/saving."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_per_class": self.f1_per_class,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
        }


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
    num_classes: int = 3,
) -> EpochMetrics:
    """Compute classification metrics.

    Args:
        predictions: Predicted class indices [N]
        labels: Ground truth labels [N]
        loss: Average loss value
        num_classes: Number of classes

    Returns:
        EpochMetrics with accuracy, F1, precision, recall
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # Accuracy
    accuracy = (predictions == labels).mean()

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(predictions, labels):
        confusion[true, pred] += 1

    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = {}

    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

        class_name = LABEL_NAMES.get(class_idx, f"class_{class_idx}")
        f1_per_class[class_name] = f1

    # Macro averages
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = np.mean(list(f1_per_class.values()))

    return EpochMetrics(
        loss=loss,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        confusion_matrix=confusion,
    )


# ============================================================
# TRAINING STEP
# ============================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict:
    """Train for one epoch.

    Args:
        model: VideoMAE classifier
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss)
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        scheduler: Optional learning rate scheduler (step per batch)

    Returns:
        Dict with training metrics (loss, accuracy, learning_rate)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 10)  # Log ~10 times per epoch

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Get sample weights if available
        sample_weights = batch.get("sample_weight")
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(pixel_values)

        # Compute loss
        if sample_weights is not None:
            # Per-sample weighted loss
            loss_unreduced = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (loss_unreduced * sample_weights).mean()
        else:
            loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Scheduler step (if per-batch scheduler)
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

        # Logging
        if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            batch_acc = (predictions == labels).float().mean().item()
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                f"loss={loss.item():.4f} acc={batch_acc:.4f} lr={current_lr:.2e}"
            )

    # Compute epoch metrics
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    elapsed = time.time() - start_time

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_predictions, all_labels, epoch_loss)

    logger.info(
        f"Epoch {epoch} training complete: "
        f"loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} "
        f"f1={metrics.f1_macro:.4f} time={elapsed:.1f}s"
    )

    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": elapsed,
    }


# ============================================================
# VALIDATION STEP
# ============================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Run validation.

    Args:
        model: VideoMAE classifier
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device

    Returns:
        EpochMetrics for validation set
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(pixel_values)
        loss = criterion(logits, labels)

        # Track metrics
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        predictions = logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    # Compute metrics
    epoch_loss = total_loss / total_samples
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_predictions, all_labels, epoch_loss)

    logger.info(
        f"Validation: loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} "
        f"f1={metrics.f1_macro:.4f} "
        f"f1_pickup={metrics.f1_per_class.get('pickup', 0):.4f} "
        f"f1_putdown={metrics.f1_per_class.get('putdown', 0):.4f}"
    )

    return metrics


# ============================================================
# TINY OVERFIT TEST (GATE B)
# ============================================================


def run_tiny_overfit_test(
    model: nn.Module,
    dataset: TrackB1Dataset,
    device: torch.device,
    config: TrainConfig,
) -> bool:
    """Gate B: Verify model can memorize small batch.

    Takes 8-16 samples and trains until loss approaches 0.
    If this fails, there's a bug in the data pipeline or model.

    Args:
        model: VideoMAE classifier (will modify weights!)
        dataset: Full training dataset
        device: Device
        config: Training configuration

    Returns:
        True if test passed, False if failed
    """
    logger.info("=" * 60)
    logger.info("GATE B: Running tiny overfit test")
    logger.info("=" * 60)

    # Create tiny subset
    num_samples = min(config.tiny_overfit_samples, len(dataset))
    indices = list(range(num_samples))
    tiny_dataset = Subset(dataset, indices)

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=num_samples,  # All samples in one batch
        shuffle=False,
        num_workers=0,
    )

    # Fresh optimizer for test
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.learning_rate * 10,  # Higher LR for faster convergence
        weight_decay=0.0,  # No regularization for overfit test
    )

    criterion = nn.CrossEntropyLoss()

    # Get the single batch
    batch = next(iter(tiny_loader))
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["label"].to(device)

    logger.info(f"Tiny overfit test: {num_samples} samples")
    logger.info(f"Label distribution: {torch.bincount(labels, minlength=3).tolist()}")

    # Training loop
    model.train()
    initial_loss = None

    for step in range(config.tiny_overfit_max_steps):
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        # Check progress
        if (step + 1) % 20 == 0:
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            logger.info(
                f"Tiny overfit step {step + 1}: loss={loss.item():.4f} acc={accuracy:.4f}"
            )

        # Success condition
        if loss.item() < config.tiny_overfit_target_loss:
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            logger.info(
                f"GATE B PASSED: loss={loss.item():.4f} < {config.tiny_overfit_target_loss} "
                f"at step {step + 1}, accuracy={accuracy:.4f}"
            )
            logger.info("=" * 60)
            return True

    # Failed
    final_loss = loss.item()
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()

    logger.error(
        f"GATE B FAILED: loss={final_loss:.4f} > {config.tiny_overfit_target_loss} "
        f"after {config.tiny_overfit_max_steps} steps"
    )
    logger.error(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    logger.error(f"Final accuracy: {accuracy:.4f}")
    logger.error("Possible issues: data pipeline bug, wrong labels, model architecture")
    logger.info("=" * 60)

    return False


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with warmup.

    Warmup: Linear increase for warmup_epochs
    Then: Cosine annealing to 0

    Args:
        optimizer: Optimizer to schedule
        num_epochs: Total epochs
        warmup_epochs: Warmup epochs
        steps_per_epoch: Batches per epoch

    Returns:
        LR scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if warmup_steps > 0:
        # Linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7,
        )

        # Combine: warmup then cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        # Just cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-7,
        )

    return scheduler


# ============================================================
# EARLY STOPPING
# ============================================================


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement threshold
            mode: "max" for metrics like F1 (higher is better),
                  "min" for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self._is_best = False

    def __call__(self, metric: float) -> bool:
        """Check if should stop.

        Args:
            metric: Current metric value

        Returns:
            True if should stop training
        """
        if self.best_value is None:
            self.best_value = metric
            self._is_best = True
            return False

        if self.mode == "max":
            improved = metric > self.best_value + self.min_delta
        else:  # mode == "min"
            improved = metric < self.best_value - self.min_delta

        if improved:
            self.best_value = metric
            self.counter = 0
            self._is_best = True
        else:
            self.counter += 1
            self._is_best = False

        return self.counter >= self.patience

    def is_best(self) -> bool:
        """Check if last metric was best so far."""
        return self._is_best


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    skip_tiny_overfit: bool = False,
) -> dict:
    """Main training loop.

    Args:
        model: VideoMAE classifier
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        device: Device to train on
        class_weights: Optional class weights for imbalanced data
        skip_tiny_overfit: Skip Gate B test (use if resuming training)

    Returns:
        Dict with:
            - best_metrics: Best validation metrics
            - best_checkpoint_path: Path to best model
            - training_history: List of per-epoch metrics
    """
    logger.info("=" * 60)
    logger.info("Starting Track B1 training")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Trainable params: {model.get_num_trainable_params():,}")
    logger.info(f"Total params: {model.get_num_total_params():,}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info("=" * 60)

    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Gate B: Tiny overfit test
    if not skip_tiny_overfit:
        # Note: This modifies model weights, so we'll reinitialize after
        test_passed = run_tiny_overfit_test(
            model=model,
            dataset=train_loader.dataset,
            device=device,
            config=config,
        )

        if not test_passed:
            raise RuntimeError(
                "Gate B (tiny overfit test) failed. "
                "Check data pipeline and model architecture before proceeding."
            )

        # Reinitialize model weights after tiny overfit test
        logger.info("Reinitializing model for actual training...")
        model = create_model(
            model_name=config.model_name,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
            dropout=config.dropout,
            device=str(device),
        )

    # Setup optimizer
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Setup loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Setup scheduler
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="max",  # Maximize F1
    )

    # Training history
    training_history = []
    best_metrics: Optional[EpochMetrics] = None
    best_checkpoint_path: Optional[Path] = None

    # Training loop
    for epoch in range(config.num_epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 60}")

        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            config=config,
            scheduler=scheduler,
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Record history
        training_history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics.to_dict(),
        })

        # Check if best
        should_stop = early_stopping(val_metrics.f1_macro)

        if early_stopping.is_best():
            best_metrics = val_metrics
            best_checkpoint_path = config.checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=val_metrics.to_dict(),
                checkpoint_path=best_checkpoint_path,
            )
            logger.info(f"New best model saved: F1={val_metrics.f1_macro:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            periodic_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=val_metrics.to_dict(),
                checkpoint_path=periodic_path,
            )

        # Early stopping
        if should_stop:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no improvement for {config.patience} epochs)"
            )
            break

    # Training complete
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if best_metrics is not None:
        logger.info(f"Best validation F1: {best_metrics.f1_macro:.4f}")
        logger.info(f"Best checkpoint: {best_checkpoint_path}")
        logger.info(f"F1 per class: {best_metrics.f1_per_class}")

    return {
        "best_metrics": best_metrics.to_dict() if best_metrics else None,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "training_history": training_history,
        "final_epoch": epoch + 1,
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================


def main(
    candidates_path: str,
    events_path: str,
    clips_path: str,
    video_dir: str,
    pose_tracks_dir: str,
    shelf_regions_path: str,
    output_dir: str,
    ignore_intervals_path: Optional[str] = None,
    config_path: Optional[str] = None,
    skip_tiny_overfit: bool = False,
) -> None:
    """CLI entry point for training.

    Args:
        candidates_path: Path to candidates.parquet
        events_path: Path to events.csv
        clips_path: Path to clips.csv with splits
        video_dir: Directory with video files
        pose_tracks_dir: Directory with pose tracks
        shelf_regions_path: Path to shelf regions YAML
        output_dir: Output directory for checkpoints/logs
        ignore_intervals_path: Optional path to ignore_intervals.parquet
        config_path: Optional path to config YAML
        skip_tiny_overfit: Skip Gate B test
    """
    import pandas as pd
    import yaml

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = TrainConfig()
    if config_path is not None:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    config.checkpoint_dir = Path(output_dir) / "checkpoints"

    # Load data
    logger.info("Loading data...")
    candidates_df = pd.read_parquet(candidates_path)
    events_df = pd.read_csv(events_path)
    clips_df = pd.read_csv(clips_path)

    if ignore_intervals_path is not None:
        ignore_intervals_df = pd.read_parquet(ignore_intervals_path)
    else:
        ignore_intervals_df = pd.DataFrame(
            columns=["clip_id", "t_start", "t_end", "reason"]
        )

    # Load shelf regions
    with open(shelf_regions_path) as f:
        shelf_regions_config = yaml.safe_load(f)
    shelf_regions = {
        r["region_id"]: r for r in shelf_regions_config.get("regions", [])
    }

    # Window config
    window_config = WindowConfig(
        window_duration_s=config_dict.get("window_duration_s", 2.5) if config_path else 2.5,
        window_stride_s=config_dict.get("window_stride_s", 0.5) if config_path else 0.5,
        num_frames=config_dict.get("num_frames", 16) if config_path else 16,
    )

    # Build manifests
    logger.info("Building window manifests...")
    train_manifest = build_window_manifest(
        candidates_df=candidates_df,
        events_df=events_df,
        ignore_intervals_df=ignore_intervals_df,
        clips_df=clips_df,
        config=window_config,
        split="train",
    )

    val_manifest = build_window_manifest(
        candidates_df=candidates_df,
        events_df=events_df,
        ignore_intervals_df=ignore_intervals_df,
        clips_df=clips_df,
        config=window_config,
        split="val",
    )

    if len(train_manifest) == 0:
        raise ValueError("No training samples found!")
    if len(val_manifest) == 0:
        raise ValueError("No validation samples found!")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        video_dir=Path(video_dir),
        pose_tracks_dir=Path(pose_tracks_dir),
        shelf_regions=shelf_regions,
        config=window_config,
        batch_size=config.batch_size,
        num_workers=4,
        clips_df=clips_df,
    )

    # Compute class weights
    class_weights = get_label_weights(train_manifest)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Create model
    logger.info("Creating model...")
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "mps" if config.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if config.device == "auto"
        else config.device
    )

    model = create_model(
        model_name=config.model_name,
        freeze_backbone=config.freeze_backbone,
        unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
        dropout=config.dropout,
        device=str(device),
    )

    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=class_weights,
        skip_tiny_overfit=skip_tiny_overfit,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training results saved to {output_path / 'training_results.json'}")


if __name__ == "__main__":
    import typer

    typer.run(main)
