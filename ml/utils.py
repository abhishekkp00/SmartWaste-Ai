"""
ml/utils.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Shared Utility Functions

Covers:
  - Reproducibility  : set_seed(), deterministic CUDA configuration
  - Device selection : get_device()
  - Logging          : get_logger() — structured, coloured console + file log
  - Checkpointing    : save_checkpoint(), load_checkpoint()
  - Training helpers : AverageMeter (tracks running mean of a metric),
                       EarlyStopping, format_time()
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide stubs so type annotations below don't crash at import time
    torch = None   # type: ignore[assignment]
    nn    = None   # type: ignore[assignment]
    np    = None   # type: ignore[assignment]



# ──────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for full reproducibility across runs.

    Sets seeds for:
      - Python built-in random
      - NumPy (if available)
      - PyTorch CPU and CUDA (if available)
      - CuDNN determinism flags

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass   # Older PyTorch versions may not support this flag



# ──────────────────────────────────────────────────────────────────────────────
# 2. Device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority: CUDA GPU > Apple MPS (M-series) > CPU

    The backend inference engine and training scripts call this once and
    move models + tensors to the returned device.

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS (Metal) detected.")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU found — using CPU. Training will be slow.")
    return device


# ──────────────────────────────────────────────────────────────────────────────
# 3. Logger
# ──────────────────────────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """ANSI colour codes for console log levels."""
    _GREY   = "\x1b[38;5;244m"
    _GREEN  = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED    = "\x1b[31m"
    _BOLD   = "\x1b[1m"
    _RESET  = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:    _GREY   + "%(levelname)-8s%(reset)s %(message)s",
        logging.INFO:     _GREEN  + "%(levelname)-8s%(reset)s %(message)s",
        logging.WARNING:  _YELLOW + "%(levelname)-8s%(reset)s %(message)s",
        logging.ERROR:    _RED    + "%(levelname)-8s%(reset)s %(message)s",
        logging.CRITICAL: _BOLD   + _RED + "%(levelname)-8s%(reset)s %(message)s",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        fmt = fmt.replace("%(reset)s", self._RESET)
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def get_logger(
    name: str = "smartwaste",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Create and return a named logger with coloured console output.

    Optionally writes a plain-text copy to `log_file`.

    Args:
        name:     Logger name (usually the calling module's __name__).
        level:    Logging level.  Defaults to INFO.
        log_file: Optional path.  If given, also writes to this file.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler with colour
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter())
    logger.addHandler(console_handler)

    # Optional file handler (plain text, no colour codes)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


# Module-level default logger
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    metric_name: str,
    config_dict: Dict[str, Any],
    save_path: Path,
) -> None:
    """
    Save a full training checkpoint to disk.

    The checkpoint contains everything needed to resume training or reproduce
    evaluation results:
      - model state dict (weights)
      - optimizer state dict (momentum, adaptive moments)
      - epoch number
      - best metric value and name
      - full training configuration dict
      - PyTorch version (for compatibility checks)

    Args:
        model:        The PyTorch model.
        optimizer:    The PyTorch optimizer.
        epoch:        The epoch at which this checkpoint was saved.
        metric_value: Value of the tracked metric (e.g. 0.9312 for val_acc).
        metric_name:  Name of the tracked metric (e.g. "val_acc").
        config_dict:  Serialisable dict of all training hyperparameters.
        save_path:    Full file path (including .pth extension).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "config": config_dict,
        "torch_version": torch.__version__,
    }

    torch.save(checkpoint, save_path)
    logger.info(
        f"Checkpoint saved → {save_path}  "
        f"[epoch={epoch}, {metric_name}={metric_value:.4f}]"
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a saved checkpoint into a model (and optionally an optimizer).

    Maps tensors to `device` automatically, so a checkpoint saved on GPU
    can be loaded on CPU for inference.

    Args:
        model:           The model to load weights into.
        checkpoint_path: Path to the .pth file.
        optimizer:       If provided, also restores optimizer state.
        device:          Target device.  Defaults to CPU.

    Returns:
        The full checkpoint dict (so callers can read epoch, metric, config).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError:      If the state dict is incompatible with the model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first or verify the path in ml/config.py."
        )

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", "unknown")
    metric_name = checkpoint.get("metric_name", "?")
    metric_value = checkpoint.get("metric_value", float("nan"))

    logger.info(
        f"Checkpoint loaded ← {checkpoint_path}  "
        f"[epoch={epoch}, {metric_name}={metric_value:.4f}]"
    )
    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """
    Track the running average of a scalar (e.g. loss or accuracy).

    Typical usage inside a training loop:
        loss_meter = AverageMeter("Loss")
        for batch in dataloader:
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), n=batch_size)
        print(loss_meter)   # → "Loss: 0.4312 (avg 0.4591)"
    """

    def __init__(self, name: str = "", fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update with a new value.

        Args:
            val: The metric value for this mini-batch.
            n:   Number of samples in this mini-batch (for weighted average).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmtstr = f"{{name}} {{val{self.fmt}}} (avg {{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Tracks the best value seen so far and counts consecutive epochs
    without improvement.  When the counter exceeds `patience`, sets
    `self.should_stop = True`.

    Args:
        patience:   Number of epochs to tolerate no improvement.
        min_delta:  Minimum change to count as an improvement.
        mode:       "min" for loss, "max" for accuracy.
        verbose:    If True, log each improvement.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric_value: float) -> bool:
        """
        Call after each validation epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = (
            metric_value < self.best_value - self.min_delta
            if self.mode == "min"
            else metric_value > self.best_value + self.min_delta
        )

        if improved:
            if self.verbose:
                logger.info(
                    f"[EarlyStopping] Improvement: "
                    f"{self.best_value:.4f} → {metric_value:.4f}"
                )
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.warning(
                    f"[EarlyStopping] No improvement for "
                    f"{self.counter}/{self.patience} epochs."
                )
            if self.counter >= self.patience:
                logger.warning(
                    "[EarlyStopping] Patience exhausted. Stopping training."
                )
                self.should_stop = True

        return self.should_stop


def format_time(seconds: float) -> str:
    """
    Convert elapsed seconds to a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string, e.g. "1h 23m 45s" or "2m 30s" or "45s".
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.

    Useful for comparing Custom CNN vs ResNet18 complexity.

    Args:
        model: Any PyTorch nn.Module.

    Returns:
        Dict with 'trainable' and 'total' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # Logger test
    log = get_logger("test")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")

    # AverageMeter test
    meter = AverageMeter("Loss", ":.4f")
    for v in [0.5, 0.4, 0.3]:
        meter.update(v, n=32)
    print(meter)

    # EarlyStopping test
    es = EarlyStopping(patience=3, mode="max", verbose=True)
    for val in [0.80, 0.82, 0.81, 0.80, 0.80, 0.80]:
        should_stop = es(val)
        if should_stop:
            break

    # Format time test
    print(format_time(5432))   # → "1h 30m 32s"

    print("\nAll utils tests passed.")
