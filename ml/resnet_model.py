"""
ml/resnet_model.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — ResNet18 transfer-learning model

Provides factory functions for the two transfer-learning stages:
  Stage A: pretrained backbone frozen; train the new classifier head.
  Stage B: layer3, layer4, and classifier head trainable.

No training loop is implemented in this phase.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg


class ResNet18Classifier(nn.Module):
    """ResNet18 backbone with a configurable six-class classifier head."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        num_classes = cfg.NUM_CLASSES if num_classes is None else num_classes
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the replacement classifier head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone and head parameters."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def prepare_stage_b(self) -> None:
        """Freeze layer1/layer2; retain layer3, layer4, and head trainability."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")


def build_resnet18_stage_a(num_classes: Optional[int] = None, pretrained: bool = True) -> ResNet18Classifier:
    """Create Stage A: pretrained backbone frozen, new head trainable."""
    return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=True)


def build_resnet18_stage_b(num_classes: Optional[int] = None, pretrained: bool = True) -> ResNet18Classifier:
    """Create Stage B: pretrained backbone with layer3/layer4 unfrozen for fine-tuning."""
    model = ResNet18Classifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=False)
    model.prepare_stage_b()
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_parameters, total_parameters)."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = build_resnet18_stage_a(pretrained=False)
    dummy = torch.zeros(2, 3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])
    model.eval()
    with torch.no_grad():
        logits = model(dummy)

    trainable, total = count_parameters(model)
    print(f"Input shape      : {tuple(dummy.shape)}")
    print(f"Logit shape      : {tuple(logits.shape)}")
    print(f"Trainable params : {trainable:,}")
    print(f"Total params     : {total:,}")
    assert logits.shape == (2, cfg.NUM_CLASSES)
