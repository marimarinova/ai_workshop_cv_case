"""Frozen image encoder wrapper for Track A feature extraction.

This module provides an abstract interface for image embedders, allowing
different encoder backends (torchvision, timm, transformers) to be used
interchangeably.

Currently implements:
- TorchVisionEmbedder: MobileNetV3, ResNet variants via torchvision
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

if TYPE_CHECKING:
    from pickup_putdown.config import TrackAFeaturesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class AbstractImageEmbedder(ABC):
    """Abstract base class for image embedding models.

    Subclasses must implement:
    - embed(): Convert image(s) to embedding vector(s)
    - embedding_dim: Property returning the output dimension
    """

    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize the embedder.

        Args:
            model_name: Name/identifier of the model.
            device: Device to run on ("auto", "cuda", "cpu").
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model: nn.Module | None = None

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the embedding vector."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version string for cache invalidation."""
        pass

    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """Embed a single image.

        Args:
            image: Image array (H, W, 3) in BGR format (OpenCV convention).

        Returns:
            Embedding vector as 1D numpy array.
        """
        pass

    def embed_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Embed a batch of images.

        Default implementation calls embed() for each image.
        Subclasses can override for more efficient batched inference.

        Args:
            images: List of image arrays (H, W, 3) in BGR format.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(img) for img in images]


# ---------------------------------------------------------------------------
# TorchVision implementation
# ---------------------------------------------------------------------------

# Supported torchvision models and their embedding dimensions
TORCHVISION_MODELS: dict[str, tuple[type, int]] = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
    "resnet18": (models.resnet18, 512),
    "resnet34": (models.resnet34, 512),
    "resnet50": (models.resnet50, 2048),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
}


class TorchVisionEmbedder(AbstractImageEmbedder):
    """Image embedder using torchvision pretrained models.

    Extracts features from the penultimate layer (before classification head).
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str = "auto",
        weights: str = "DEFAULT",
    ):
        """Initialize the torchvision embedder.

        Args:
            model_name: One of the supported torchvision model names.
            device: Device to run on ("auto", "cuda", "cpu").
            weights: Pretrained weights to use ("DEFAULT" for ImageNet).

        Raises:
            ValueError: If model_name is not supported.
        """
        if model_name not in TORCHVISION_MODELS:
            supported = list(TORCHVISION_MODELS.keys())
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

        super().__init__(model_name, device)

        self._weights = weights
        self._embedding_dim = TORCHVISION_MODELS[model_name][1]

        # Build model
        self._model = self._build_model()
        self._model.to(self.device)
        self._model.eval()

        # Preprocessing transform (ImageNet normalization)
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info(
            f"Loaded {model_name} embedder on {self.device}, embedding_dim={self._embedding_dim}"
        )

    def _build_model(self) -> nn.Module:
        """Build the model with classification head removed."""
        model_class, _ = TORCHVISION_MODELS[self.model_name]

        # Load pretrained model
        if self._weights == "DEFAULT":
            model = model_class(weights="DEFAULT")
        else:
            model = model_class(weights=None)

        # Remove classification head to get feature extractor
        if "mobilenet" in self.model_name:
            # MobileNet: remove classifier, keep avgpool
            model.classifier = nn.Identity()
        elif "resnet" in self.model_name:
            # ResNet: remove fc layer
            model.fc = nn.Identity()
        elif "efficientnet" in self.model_name:
            # EfficientNet: remove classifier
            model.classifier = nn.Identity()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        return model

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the embedding vector."""
        return self._embedding_dim

    @property
    def version(self) -> str:
        """Return version string for cache invalidation."""
        return f"torchvision_{self.model_name}_{self._weights}_v1"

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Embed a single image.

        Args:
            image: Image array (H, W, 3) in BGR format.

        Returns:
            Embedding vector as 1D numpy array.
        """
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1].copy()

        # Preprocess
        tensor = self._transform(image_rgb)
        tensor = tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self._model(tensor)

        # Flatten and convert to numpy
        embedding = features.squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Embed a batch of images efficiently.

        Args:
            images: List of image arrays (H, W, 3) in BGR format.

        Returns:
            List of embedding vectors.
        """
        if not images:
            return []

        # Preprocess all images
        tensors = []
        for img in images:
            img_rgb = img[:, :, ::-1].copy()
            tensor = self._transform(img_rgb)
            tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(tensors).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self._model(batch)

        # Convert to list of numpy arrays
        embeddings = [f.cpu().numpy() for f in features]

        return embeddings


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_embedder(
    config: TrackAFeaturesConfig,
) -> AbstractImageEmbedder:
    """Create an embedder based on configuration.

    Args:
        config: Track A features configuration.

    Returns:
        Configured embedder instance.

    Raises:
        ValueError: If encoder_name is not recognized.
    """
    encoder_name = config.encoder_name

    # Check if it's a torchvision model
    if encoder_name in TORCHVISION_MODELS:
        return TorchVisionEmbedder(
            model_name=encoder_name,
            device=config.encoder_device,
        )

    # Future: add timm, transformers backends here
    # if encoder_name.startswith("timm_"):
    #     return TimmEmbedder(...)
    # if encoder_name.startswith("hf_"):
    #     return TransformersEmbedder(...)

    raise ValueError(
        f"Unknown encoder '{encoder_name}'. "
        f"Supported torchvision models: {list(TORCHVISION_MODELS.keys())}"
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def save_embedding(
    embedding: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Save an embedding to disk as .npy file.

    Args:
        embedding: Embedding vector.
        output_path: Destination path (should end in .npy).

    Returns:
        Path where the embedding was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)
    return output_path


def load_embedding(path: Path | str) -> np.ndarray:
    """Load an embedding from disk.

    Args:
        path: Path to .npy file.

    Returns:
        Embedding vector.
    """
    return np.load(path)
