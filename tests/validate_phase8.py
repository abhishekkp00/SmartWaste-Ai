"""
tests/validate_phase8.py
─────────────────────────────────────────────────────────────────────────────
Phase 8 validation — FastAPI backend routes, logging, and analytics.
"""

import io
import sys
from pathlib import Path
from typing import List, Tuple

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, message: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {message}" if message else ""))
    RESULTS.append((name, passed, message))


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n {title}\n{'─' * 60}")


def png_bytes(color=(120, 180, 70)) -> bytes:
    image = Image.new("RGB", (320, 240), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    with TestClient(app) as client:
        section("1. Health Endpoint")
        try:
            response = client.get("/api/health")
            data = response.json()
            record("GET /api/health status code", response.status_code == 200, str(response.status_code))
            record("Health status field", data.get("status") == "ok", str(data.get("status")))
            record("Supported classes list", isinstance(data.get("supported_classes"), list))
        except Exception as exc:
            record("Health endpoint", False, str(exc))
            return print_summary()

        section("2. Single Prediction Endpoint")
        try:
            image_data = png_bytes((90, 120, 200))
            response = client.post(
                "/api/predict",
                files={"file": ("single.png", image_data, "image/png")},
                data={"model_name": "custom_cnn", "top_k": "3"},
            )
            payload = response.json()
            record("POST /api/predict status code", response.status_code == 200, str(response.status_code))
            record("Prediction class present", isinstance(payload.get("class_name"), str), str(payload.get("class_name")))
            record("Top-k length == 3", len(payload.get("top_k", [])) == 3, str(len(payload.get("top_k", []))))
            record("Prediction log_id present", isinstance(payload.get("log_id"), int), str(payload.get("log_id")))
        except Exception as exc:
            record("Single prediction endpoint", False, str(exc))

        section("3. Batch Prediction Endpoint")
        try:
            files = [
                ("files", ("batch1.png", png_bytes((200, 100, 120)), "image/png")),
                ("files", ("batch2.png", png_bytes((100, 210, 140)), "image/png")),
            ]
            response = client.post(
                "/api/predict/batch",
                files=files,
                data={"model_name": "custom_cnn", "top_k": "2"},
            )
            payload = response.json()
            record("POST /api/predict/batch status code", response.status_code == 200, str(response.status_code))
            record("Batch total == 2", payload.get("total") == 2, str(payload.get("total")))
            item_topk_ok = all(len(item.get("top_k", [])) == 2 for item in payload.get("items", []))
            record("Batch top-k length respected", item_topk_ok)
        except Exception as exc:
            record("Batch prediction endpoint", False, str(exc))

        section("4. Analytics Endpoint")
        try:
            response = client.get("/api/analytics?limit=5")
            payload = response.json()
            summary = payload.get("summary", {})
            recent = payload.get("recent_predictions", [])

            record("GET /api/analytics status code", response.status_code == 200, str(response.status_code))
            record("Summary total_predictions present", isinstance(summary.get("total_predictions"), int), str(summary.get("total_predictions")))
            record("Summary class_distribution present", isinstance(summary.get("class_distribution"), dict))
            record("Recent predictions list", isinstance(recent, list), f"len={len(recent)}")
            record("Recent predictions non-empty", len(recent) >= 1, f"len={len(recent)}")
        except Exception as exc:
            record("Analytics endpoint", False, str(exc))

    return print_summary()


def print_summary() -> int:
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed

    print(f"\n{'═' * 60}\n Phase 8 Validation Summary\n{'═' * 60}")
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

    print("\n ✅ ALL PHASE 8 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
