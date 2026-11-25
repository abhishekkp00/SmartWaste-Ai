"""
ml/custom_cnn.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Custom CNN from Scratch

Implements a modular five-block convolutional network for TrashNet:

    224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 512 features -> 6 logits

The architecture is independent of pretrained weights. All image size,
class count, and dropout settings are reused from ml/config.py.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg


class ConvBlock(nn.Module):
    """Convolution, batch normalization, ReLU, and 2×2 max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class CustomCNN(nn.Module):
    """Five-block custom CNN for 224×224 waste images."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        dropout1: Optional[float] = None,
        dropout2: Optional[float] = None,
    ) -> None:
        super().__init__()
        num_classes = cfg.NUM_CLASSES if num_classes is None else num_classes
        dropout1 = cfg.training.CNN_DROPOUT1 if dropout1 is None else dropout1
        dropout2 = cfg.training.CNN_DROPOUT2 if dropout2 is None else dropout2

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout1 < 1.0 or not 0.0 <= dropout2 < 1.0:
            raise ValueError("dropout values must be in [0, 1)")

        self.num_classes = num_classes

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the 512-dimensional feature vector for each batch item."""
        features = self.features(x)
        return features.flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        """Return unnormalized logits with shape [Batch, num_classes]."""
        features = self.features(x)
        return self.classifier(features)

    def freeze_features(self) -> None:
        """Freeze convolutional feature extractors (not used in Phase 3 training)."""
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_features(self) -> None:
        """Restore gradient updates for convolutional feature extractors."""
        for parameter in self.features.parameters():
            parameter.requires_grad = True


def build_custom_cnn(
    num_classes: Optional[int] = None,
    dropout1: Optional[float] = None,
    dropout2: Optional[float] = None,
) -> CustomCNN:
    """Factory function that returns a CustomCNN using central configuration defaults."""
    return CustomCNN(num_classes=num_classes, dropout1=dropout1, dropout2=dropout2)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_parameters, total_parameters)."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = build_custom_cnn()
    dummy = torch.zeros(2, 3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])

    model.eval()
    with torch.no_grad():
        logits = model(dummy)
        features = model.forward_features(dummy)

    trainable, total = count_parameters(model)
    print(f"Input shape     : {tuple(dummy.shape)}")
    print(f"Feature shape   : {tuple(features.shape)}")
    print(f"Logit shape     : {tuple(logits.shape)}")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params    : {total:,}")

    assert logits.shape == (2, cfg.NUM_CLASSES)
    assert features.shape == (2, 512)
