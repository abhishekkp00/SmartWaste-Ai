"""
tests/validate_phase7.py
─────────────────────────────────────────────────────────────────────────────
Phase 7 validation — Inference engine and recommendation logic.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.custom_cnn import build_custom_cnn
from ml.predict import RECYCLING_RECOMMENDATIONS, WastePredictor

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, message: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {message}" if message else ""))
    RESULTS.append((name, passed, message))


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n {title}\n{'─' * 60}")


def synthetic_image() -> Image.Image:
    return Image.new("RGB", (320, 240), color=(120, 180, 70))


def main() -> int:
    section("1. Recommendation Mapping")
    record("6 recommendation entries", len(RECYCLING_RECOMMENDATIONS) == cfg.NUM_CLASSES)
    record(
        "All classes covered",
        set(RECYCLING_RECOMMENDATIONS.keys()) == set(cfg.CLASS_NAMES),
        f"covered={sorted(RECYCLING_RECOMMENDATIONS.keys())}",
    )

    section("2. Predictor Setup + Preprocessing")
    try:
        predictor = WastePredictor(
            model_name="custom_cnn",
            device=torch.device("cpu"),
            confidence_threshold=0.0,
            top_k=3,
        )
        batch = predictor.preprocess(synthetic_image())
        record("Preprocess shape", tuple(batch.shape) == (1, 3, 224, 224), f"got {tuple(batch.shape)}")
        record("Preprocess dtype", batch.dtype == torch.float32, f"got {batch.dtype}")
    except Exception as exc:
        record("Predictor setup", False, str(exc))
        return print_summary()

    section("3. Single Inference")
    try:
        result = predictor.predict(synthetic_image())
        record("Predicted class valid", result["class_name"] in cfg.CLASS_NAMES, result["class_name"])
        record("Top-k length", len(result["top_k"]) == 3, f"got {len(result['top_k'])}")

        confidences = [item["confidence"] for item in result["top_k"]]
        non_increasing = all(a >= b for a, b in zip(confidences, confidences[1:]))
        record("Top-k sorted by confidence", non_increasing, str(confidences))

        recommendation = result["recommendation"]
        record("Recommendation has disposal_bin", "disposal_bin" in recommendation)
        record("Recommendation has prep", "prep" in recommendation)
    except Exception as exc:
        record("Single inference", False, str(exc))

    section("4. Confidence Threshold Warning")
    try:
        strict_predictor = WastePredictor(
            model_name="custom_cnn",
            device=torch.device("cpu"),
            confidence_threshold=0.9999,
            top_k=2,
        )
        strict_result = strict_predictor.predict(synthetic_image())
        record("Low-confidence warning emitted", strict_result["warning"] is not None)
        record("Low-confidence flag false", strict_result["is_confident"] is False)
    except Exception as exc:
        record("Confidence warning", False, str(exc))

    section("5. Checkpoint Loading")
    temp_state_path = cfg.paths.models_dir / "phase7_temp_state_dict.pth"
    temp_wrapped_path = cfg.paths.models_dir / "phase7_temp_wrapped_checkpoint.pth"
    try:
        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        model = build_custom_cnn()

        torch.save(model.state_dict(), temp_state_path)
        loaded_raw = WastePredictor(
            model_name="custom_cnn",
            checkpoint_path=temp_state_path,
            device=torch.device("cpu"),
        )
        record("Raw state_dict load", loaded_raw.model is not None)

        torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, temp_wrapped_path)
        loaded_raw.load_checkpoint(temp_wrapped_path)
        record("Wrapped checkpoint load", True)

        # Regression: older checkpoints stored torch.__version__ as a
        # TorchVersion object, which PyTorch 2.6's weights-only loader
        # rejects unless the metadata class is explicitly allowlisted.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 1,
                "torch_version": torch.__version__,
            },
            temp_wrapped_path,
        )
        loaded_raw.load_checkpoint(temp_wrapped_path)
        record("Legacy TorchVersion metadata checkpoint load", True)
    except Exception as exc:
        record("Checkpoint loading", False, str(exc))
    finally:
        if temp_state_path.exists():
            temp_state_path.unlink()
        if temp_wrapped_path.exists():
            temp_wrapped_path.unlink()

    section("6. Batch Inference")
    try:
        outputs = predictor.predict_batch([synthetic_image(), synthetic_image()], top_k=2)
        record("Batch output length", len(outputs) == 2, f"got {len(outputs)}")
        record("Batch top-k length respected", all(len(item["top_k"]) == 2 for item in outputs))
    except Exception as exc:
        record("Batch inference", False, str(exc))

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 7 Validation Summary\n{'═' * 60}")
    print(f"  Total checks : {len(RESULTS)}")
    print(f"  ✅ Passed    : {passed}")
    print(f"  ❌ Failed    : {failed}")
    print("═" * 60)

    if failed:
        print("\n Failed checks:")
        for name, ok, message in RESULTS:
            if not ok:
                print(f"  ✗ {name}: {message}")
        return 1

    print("\n ✅ ALL PHASE 7 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
