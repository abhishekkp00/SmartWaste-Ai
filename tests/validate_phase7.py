"""
tests/validate_phase7.py
─────────────────────────────────────────────────────────────────────────────
Phase 7 validation for image inference and recycling recommendations.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.predict import WastePredictor
from ml.recommendations import get_recommendation

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, message: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {message}" if message else ""))
    RESULTS.append((name, passed, message))


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n {title}\n{'─' * 60}")


def main() -> int:
    section("1. Predictor Construction")
    try:
        predictor = WastePredictor(device=torch.device("cpu"), top_k=3)
        record("Predictor device is CPU", predictor.device.type == "cpu")
        record("Top-k is config-compatible", predictor.top_k == cfg.inference.TOP_K,
               f"got {predictor.top_k}")
        record("Transform output contract", predictor.transform(torch.zeros((224, 224, 3), dtype=torch.uint8)).shape == torch.Size([3, 224, 224]))
    except Exception as exc:
        record("Predictor construction", False, str(exc))
        return print_summary()

    section("2. Single-Image Inference")
    try:
        rng = np.random.default_rng(42)
        array = (rng.random((320, 480, 3)) * 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")

        result = predictor.predict(image)
        record("Top-level class name", result["class_name"] in cfg.CLASS_NAMES,
               f"got {result['class_name']}")
        record("Predicted index range", 0 <= result["class_index"] < cfg.NUM_CLASSES,
               f"got {result['class_index']}")
        record("Confidence range", 0.0 <= result["confidence"] <= 1.0,
               f"got {result['confidence']:.4f}")
        record("Top-k count", len(result["top_k"]) == 3,
               f"got {len(result['top_k'])}")
        record("Top-k probabilities sorted", all(
            result["top_k"][i]["probability"] >= result["top_k"][i + 1]["probability"]
            for i in range(len(result["top_k"])-1)
        ))
        record("Threshold used", result["threshold"] == cfg.inference.CONFIDENCE_THRESHOLD)
        record("Uncertainty flag consistent", result["is_uncertain"] == (result["confidence"] < cfg.inference.CONFIDENCE_THRESHOLD))
    except Exception as exc:
        record("Single-image inference", False, str(exc))

    section("3. Batch Inference")
    try:
        images = [
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB"),
            Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8), mode="RGB"),
        ]
        results = predictor.predict_batch(images)
        record("Batch result count", len(results) == 2, f"got {len(results)}")
        record("Batch confidence fields", all("confidence" in result for result in results))
        record("Batch top-k fields", all(len(result["top_k"]) == 3 for result in results))
    except Exception as exc:
        record("Batch inference", False, str(exc))

    section("4. Recycling Recommendations")
    try:
        for class_name in cfg.CLASS_NAMES:
            recommendation = get_recommendation(class_name, confidence=0.80)
            record(f"Recommendation for {class_name}", recommendation["class_name"] == class_name)

        uncertain = get_recommendation("plastic", confidence=0.40)
        record("Low confidence warning", uncertain["is_uncertain"] is True)
        record("Warning text present", bool(uncertain["warning"]))

        try:
            get_recommendation("unknown", confidence=0.90)
            record("Unknown class guard", False, "recommendation did not raise")
        except ValueError:
            record("Unknown class guard", True)
    except Exception as exc:
        record("Recommendation validation", False, str(exc))

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
