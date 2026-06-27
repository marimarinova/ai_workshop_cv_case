"""Track B1 VideoMAE classifier model for pickup/putdown detection.

This module provides:
- VideoMAE encoder wrapper with HuggingFace pretrained weights
- Classification head for 3-class prediction (background/pickup/putdown)
- Configurable backbone freezing for transfer learning
- Checkpoint save/load utilities

Architecture:
    Input: [batch_size, num_frames, channels, height, width]
           [    8     ,    16     ,    3    ,  224  ,  224 ]
                        │
                        ▼
    VideoMAE Encoder (frozen or partially frozen)
    - Splits each frame into 14×14 patches (patch_size=16)
    - Processes patches through transformer blocks
    - Output: [batch_size, sequence_length, hidden_dim]
                        │
                        ▼
    Mean Pooling over sequence dimension
    - Output: [batch_size, hidden_dim]
                        │
                        ▼
    Classification Head (trainable)
    - LayerNorm → Dropout → Linear
    - Output: [batch_size, num_classes]
              [    8     ,     3     ]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEModel

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES: int = 3
LABEL_BACKGROUND: int = 0
LABEL_PICKUP: int = 1
LABEL_PUTDOWN: int = 2

# Default model configurations
DEFAULT_MODEL_NAME: str = "MCG-NJU/videomae-small"
VIDEOMAE_SMALL_HIDDEN_DIM: int = 384
VIDEOMAE_BASE_HIDDEN_DIM: int = 768


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

#TODO: If data set grows its better approach to have a classificationhead that is not just a simple matrix
class ClassificationHead(nn.Module):
    """Lightweight MLP head for 3-class video classification.

    Architecture:
        Input: [batch_size, hidden_dim]
               [    8     ,    384    ]
                    │
                    ▼
        LayerNorm(hidden_dim)
                    │
                    ▼
        Dropout(dropout_prob)
                    │
                    ▼
        Linear(hidden_dim → num_classes)
                    │
                    ▼
        Output: [batch_size, num_classes]
                [    8     ,     3     ]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        """Initialize classification head.

        Args:
            hidden_dim: Input feature dimension from encoder (384 for small, 768 for base).
            num_classes: Number of output classes (3: background/pickup/putdown).
            dropout: Dropout probability before final linear layer.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer with small weights for stable training."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            features: Pooled encoder features.
                      Shape: [batch_size, hidden_dim]

        Returns:
            logits: Class logits (not probabilities).
                    Shape: [batch_size, num_classes]
                    Index 0 = background, 1 = pickup, 2 = putdown
        """
        x = self.norm(features)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# ============================================================
# MAIN MODEL
# ============================================================


class VideoMAEClassifier(nn.Module):
    """VideoMAE encoder + classification head for event detection.

    This model wraps a pretrained VideoMAE encoder and adds a classification
    head for 3-class video classification (background/pickup/putdown).

    The encoder can be fully frozen (only head trains) or partially unfrozen
    (last N transformer blocks also train) for fine-tuning.

    Input tensor shape: [batch_size, num_frames, channels, height, width]
                        [    8     ,    16     ,    3    ,  224  ,  224 ]

    Output tensor shape: [batch_size, num_classes]
                         [    8     ,     3     ]
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ) -> None:
        """Initialize VideoMAE classifier.

        Args:
            model_name: HuggingFace model identifier (e.g., "MCG-NJU/videomae-small").
            num_classes: Number of output classes.
            dropout: Dropout probability in classification head.
            freeze_backbone: Whether to freeze encoder weights.
            unfreeze_last_n_blocks: Number of final transformer blocks to unfreeze
                                    (only used if freeze_backbone=True).
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # Load pretrained encoder
        #The VideoMAe obj 
        self.encoder = self._load_encoder(model_name)

        # Get hidden dimension from encoder config
        self.hidden_dim = self.encoder.config.hidden_size

        # Create classification head
        self.head = ClassificationHead(
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Configure freezing
        self._configure_freezing(freeze_backbone, unfreeze_last_n_blocks)

        logger.info(
            f"VideoMAEClassifier initialized: model={model_name}, "
            f"hidden_dim={self.hidden_dim}, num_classes={num_classes}, "
            f"freeze_backbone={freeze_backbone}, unfreeze_last_n_blocks={unfreeze_last_n_blocks}"
        )

    def _load_encoder(self, model_name: str) -> VideoMAEModel:
        """Load pretrained VideoMAE encoder from HuggingFace.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            Loaded VideoMAE encoder model.
        """
        logger.info(f"Loading VideoMAE encoder from: {model_name}")

        try:
            encoder = VideoMAEModel.from_pretrained(model_name)
            logger.info(
                f"Encoder loaded: {encoder.config.num_hidden_layers} layers, "
                f"hidden_size={encoder.config.hidden_size}"
            )
            return encoder
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            raise

    def _configure_freezing(
        self,
        freeze_backbone: bool,
        unfreeze_last_n_blocks: int,
    ) -> None:
        """Configure which parameters are trainable.

        Args:
            freeze_backbone: If True, freeze all encoder parameters.
            unfreeze_last_n_blocks: If freeze_backbone=True and this > 0,
                                    unfreeze the last N transformer blocks.
        """
        if not freeze_backbone:
            # All parameters trainable
            for param in self.encoder.parameters():
                param.requires_grad = True
            logger.info("Encoder fully trainable")
            return

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # Unfreeze last N transformer blocks
            num_layers = self.encoder.config.num_hidden_layers

            if unfreeze_last_n_blocks > num_layers:
                logger.warning(
                    f"unfreeze_last_n_blocks ({unfreeze_last_n_blocks}) > "
                    f"num_layers ({num_layers}), unfreezing all blocks"
                )
                unfreeze_last_n_blocks = num_layers

            # VideoMAE encoder structure: encoder.encoder.layer[i]
            for i in range(num_layers - unfreeze_last_n_blocks, num_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True

            logger.info(f"Unfroze last {unfreeze_last_n_blocks} encoder blocks")
        else:
            logger.info("Encoder fully frozen, only head is trainable")

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through encoder and classification head.

        Args:
            pixel_values: Input video frames.
                          Shape: [batch_size, num_frames, channels, height, width]
                          Example: [8, 16, 3, 224, 224]
                          - batch_size: number of video clips in batch
                          - num_frames: frames per clip (16)
                          - channels: RGB color channels (3)
                          - height: frame height in pixels (224)
                          - width: frame width in pixels (224)

        Returns:
            logits: Classification logits (not softmax probabilities).
                    Shape: [batch_size, num_classes]
                    Example: [8, 3]
                    - Index 0: background score
                    - Index 1: pickup score
                    - Index 2: putdown score
        """
        # Encode video through VideoMAE
        # Input: [batch_size, num_frames, channels, height, width]
        # Output: BaseModelOutput with last_hidden_state [batch_size, sequence_length, hidden_dim]
        encoder_output = self.encoder(pixel_values=pixel_values)

        # Extract hidden states
        # Shape: [batch_size, sequence_length, hidden_dim]
        # Example: [8, 1568, 384] for VideoMAE-small
        hidden_states = encoder_output.last_hidden_state

        # Pool over sequence dimension
        # Shape: [batch_size, hidden_dim]
        # Example: [8, 384]
        pooled_features = self._pool_features(hidden_states)

        # Classify
        # Shape: [batch_size, num_classes]
        # Example: [8, 3]
        logits = self.head(pooled_features)

        return logits

    def _pool_features(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Pool encoder output over sequence dimension using mean pooling.

        Args:
            hidden_states: Encoder output.
                           Shape: [batch_size, sequence_length, hidden_dim]
                           Example: [8, 1568, 384]
                           - sequence_length: num_frames × patches_per_frame
                           - hidden_dim: transformer hidden dimension

        Returns:
            pooled: Mean-pooled features.
                    Shape: [batch_size, hidden_dim]
                    Example: [8, 384]
        """
        # Mean pooling over sequence dimension (dim=1)
        pooled = hidden_states.mean(dim=1)
        return pooled

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get list of trainable parameters for optimizer.

        Returns:
            List of parameters with requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Count total number of parameters (trainable + frozen).

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())


# ============================================================
# FACTORY FUNCTIONS
# ============================================================


def create_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    device: str = "auto",
) -> VideoMAEClassifier:
    """Factory function to create and configure VideoMAE classifier.

    Args:
        model_name: HuggingFace model identifier.
        num_classes: Number of output classes.
        dropout: Dropout probability in classification head.
        freeze_backbone: Whether to freeze encoder weights.
        unfreeze_last_n_blocks: Number of final blocks to unfreeze.
        device: Device to place model on ("auto", "cuda", "mps", "cpu").

    Returns:
        Configured VideoMAEClassifier on specified device.
    """
    # Create model
    model = VideoMAEClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )

    # Move to device
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)

    logger.info(
        f"Model created: {model.get_num_trainable_params():,} trainable params, "
        f"{model.get_num_total_params():,} total params, device={resolved_device}"
    )

    return model


def _resolve_device(device: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu").

    Returns:
        Resolved torch.device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device)


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================


def save_checkpoint(
    model: VideoMAEClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: str | Path,
) -> Path:
    """Save model checkpoint with optimizer state and metrics.

    Args:
        model: Model to save.
        optimizer: Optimizer with state to save.
        epoch: Current epoch number.
        metrics: Dictionary of metrics to store.
        checkpoint_path: Path to save checkpoint.

    Returns:
        Path where checkpoint was saved.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_config": {
            "model_name": model.model_name,
            "num_classes": model.num_classes,
            "hidden_dim": model.hidden_dim,
            "freeze_backbone": model.freeze_backbone,
            "unfreeze_last_n_blocks": model.unfreeze_last_n_blocks,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Optional[VideoMAEClassifier] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "auto",
    strict: bool = True,
) -> dict:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Optional model to load weights into. If None, creates new model.
        optimizer: Optional optimizer to load state into.
        device: Device to load model onto.
        strict: Whether to strictly enforce state dict matching.

    Returns:
        Dictionary with:
            - "model": loaded model
            - "optimizer": optimizer (if provided)
            - "epoch": epoch number
            - "metrics": stored metrics
            - "model_config": model configuration
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

    # Create model if not provided
    if model is None:
        config = checkpoint["model_config"]
        model = VideoMAEClassifier(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            freeze_backbone=config["freeze_backbone"],
            unfreeze_last_n_blocks=config["unfreeze_last_n_blocks"],
        )
        model = model.to(resolved_device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info(f"Model weights loaded from: {checkpoint_path}")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "model_config": checkpoint.get("model_config", {}),
    }


# ============================================================
# INFERENCE UTILITIES
# ============================================================


@torch.no_grad()
def predict_batch(
    model: VideoMAEClassifier,
    pixel_values: torch.Tensor,
    return_probs: bool = True,
) -> dict:
    """Run inference on a batch of video clips.

    Args:
        model: Trained VideoMAE classifier.
        pixel_values: Input batch.
                      Shape: [batch_size, num_frames, channels, height, width]
        return_probs: If True, return softmax probabilities; else return logits.

    Returns:
        Dictionary with:
            - "predictions": predicted class indices [batch_size]
            - "scores": confidence scores for predicted class [batch_size]
            - "probs" or "logits": full output [batch_size, num_classes]
    """
    model.eval()

    # Forward pass
    logits = model(pixel_values)

    if return_probs:
        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        scores = probs.gather(dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)

        return {
            "predictions": predictions,
            "scores": scores,
            "probs": probs,
        }
    else:
        predictions = logits.argmax(dim=-1)
        scores = logits.gather(dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)

        return {
            "predictions": predictions,
            "scores": scores,
            "logits": logits,
        }
