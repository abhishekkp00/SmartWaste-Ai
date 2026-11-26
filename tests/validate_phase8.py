"""
Phase 8 validation — FastAPI routes, checkpoint safety, history, and analytics.

Predictions are expected to return HTTP 503 when no trained checkpoint exists.
This prevents accidentally serving outputs from randomly initialized weights.
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
        section("1. Health and CORS")
        response = client.get("/api/health")
        data = response.json()
        record("GET /api/health", response.status_code == 200, str(response.status_code))
        record("Health status field", data.get("status") == "ok", str(data.get("status")))
        record("Supported classes list", isinstance(data.get("supported_classes"), list))
        record("Checkpoint readiness field", isinstance(data.get("model_ready"), bool))
        record("Available models field", isinstance(data.get("available_models"), list))

        cors = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        record(
            "Vite CORS preflight",
            cors.status_code in (200, 204)
            and cors.headers.get("access-control-allow-origin") == "http://localhost:5173",
            str(cors.status_code),
        )

        checkpoint_available = "custom_cnn" in data.get("available_models", [])

        section("2. Single Prediction Endpoint")
        response = client.post(
            "/api/predict",
            files={"file": ("single.png", png_bytes(), "image/png")},
            data={"model_name": "custom_cnn", "top_k": "3"},
        )
        if checkpoint_available:
            payload = response.json()
            record("Single prediction accepted", response.status_code == 200, str(response.status_code))
            record("Prediction class present", isinstance(payload.get("class_name"), str))
            record("Top-k length == 3", len(payload.get("top_k", [])) == 3)
            record("Prediction log_id present", isinstance(payload.get("log_id"), int))
        else:
            record(
                "Missing checkpoint rejected safely",
                response.status_code == 503
                and "No trained checkpoint" in response.json().get("detail", ""),
                str(response.status_code),
            )

        section("3. Batch Prediction Endpoint")
        files = [
            ("files", ("batch1.png", png_bytes((200, 100, 120)), "image/png")),
            ("files", ("batch2.png", png_bytes((100, 210, 140)), "image/png")),
        ]
        response = client.post(
            "/api/predict/batch",
            files=files,
            data={"model_name": "custom_cnn", "top_k": "2"},
        )
        if checkpoint_available:
            payload = response.json()
            record("Batch prediction accepted", response.status_code == 200, str(response.status_code))
            record("Batch total == 2", payload.get("total") == 2)
            record(
                "Batch top-k length respected",
                all(len(item.get("top_k", [])) == 2 for item in payload.get("items", [])),
            )
        else:
            record("Batch rejects missing checkpoint safely", response.status_code == 503, str(response.status_code))

        section("4. Analytics Endpoint")
        response = client.get("/api/analytics?limit=5")
        payload = response.json()
        summary = payload.get("summary", {})
        recent = payload.get("recent_predictions", [])
        record("GET /api/analytics", response.status_code == 200, str(response.status_code))
        record("Summary total_predictions is integer", isinstance(summary.get("total_predictions"), int))
        record("Summary class_distribution is object", isinstance(summary.get("class_distribution"), dict))
        record("Recent predictions is list", isinstance(recent, list))
        record("Recent predictions respect limit", len(recent) <= 5, f"len={len(recent)}")

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
