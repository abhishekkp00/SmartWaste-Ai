"""
ml/train.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Training engine for CustomCNN and ResNet18

Provides:
  - weighted CrossEntropyLoss from training labels
  - optional WeightedRandomSampler
  - AdamW optimizer parameter groups
  - training/validation loops
  - validation-based early stopping
  - checkpoint saving and training history

This module contains the reusable engine only; experiments decide epochs,
models, learning rates, and whether to use the weighted sampler.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.dataset import compute_class_weights, create_dataloaders
from ml.metrics import accuracy_from_logits, classification_metrics
from ml.utils import EarlyStopping, AverageMeter, set_seed


class Trainer:
    """Reusable training and validation loop with checkpoint support."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        weighted_loss: bool = True,
        checkpoint_path: Optional[Path] = None,
        class_weights: Optional[Tensor] = None,
        parameter_groups: Optional[Iterable[Dict[str, object]]] = None,
        patience: Optional[int] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if parameter_groups is None:
            parameter_groups = [{"params": self.model.parameters(), "lr": learning_rate, "weight_decay": weight_decay}]
        self.optimizer = AdamW(parameter_groups)

        if weighted_loss:
            if class_weights is None:
                raise ValueError("class_weights are required when weighted_loss=True")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.scheduler: Optional[CosineAnnealingLR] = None
        self.early_stopping = EarlyStopping(
            patience=cfg.training.PATIENCE if patience is None else patience,
            mode="max",
            verbose=True,
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else cfg.paths.models_dir / "custom_cnn_best.pth"
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = float("-inf")

    def configure_scheduler(self, t_max: int, eta_min: float = 1e-6) -> None:
        """Configure cosine learning-rate annealing."""
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Run one training epoch and return loss/accuracy."""
        self.model.train()
        loss_meter = AverageMeter("Loss", ":.4f")
        acc_meter = AverageMeter("Acc", ":.4f")

        for images, labels in train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy_from_logits(logits.detach(), labels), batch_size)

        return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    @torch.no_grad()
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Run validation and return loss/accuracy."""
        self.model.eval()
        loss_meter = AverageMeter("Loss", ":.4f")
        acc_meter = AverageMeter("Acc", ":.4f")

        all_predictions: List[int] = []
        all_targets: List[int] = []

        for images, labels in val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy_from_logits(logits, labels), batch_size)

            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

        metrics = classification_metrics(all_predictions, all_targets, cfg.NUM_CLASSES)
        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg,
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }

    def fit(self, train_loader, val_loader, epochs: int) -> Dict[str, List[float]]:
        """Train for the requested number of epochs and return history."""
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(epoch, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping(val_metrics["accuracy"]):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        self.save_history()
        return self.history

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Save model and optimizer state plus validation metrics."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "val_metrics": val_metrics,
            "history": self.history,
            "torch_version": str(torch.__version__),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Best checkpoint saved -> {self.checkpoint_path}")

    def save_history(self) -> Path:
        """Save training history JSON under results/reports."""
        history_path = cfg.paths.reports_dir / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return history_path


def build_weighted_loss(train_dataset) -> torch.Tensor:
    """Compute class weights from the training dataset only."""
    return compute_class_weights(train_dataset, device=torch.device("cpu"))


def create_training_components(
    model: nn.Module,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = False,
    weighted_loss: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> tuple:
    """Create DataLoaders, training weights, and a reusable Trainer."""
    set_seed(cfg.training.RANDOM_SEED)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0 if num_workers is None else num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )

    train_dataset = train_loader.dataset
    weights = build_weighted_loss(train_dataset)
    trainer = Trainer(
        model=model,
        weighted_loss=weighted_loss,
        class_weights=weights,
        checkpoint_path=checkpoint_path,
    )
    return trainer, train_loader, val_loader, test_loader, weights
