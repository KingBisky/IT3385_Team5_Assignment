"""Automated evidence for Clifton's Task 3 Flask application.

The tests use a deterministic fake classifier for route behaviour so CI can
validate web logic independently from expensive model deserialisation. The
real `.pkl` artefact is also asserted to exist in the submitted repository.
"""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = (
    ROOT
    / "src"
    / "team5_app"
    / "Clifton"
    / "mental_health_risk_app"
    / "app.py"
)
MODEL_FILE = APP_FILE.parent / "clifton_mental_health_risk_final_model.pkl"


def _load_clifton_module():
    spec = importlib.util.spec_from_file_location("clifton_task3_test_app", APP_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    return module


class FakePipeline:
    """Small classifier stub that exercises the same predict/proba interface."""

    classes_ = np.array([0, 1, 2])

    def predict(self, frame: pd.DataFrame):
        # Deterministic rule purely for route tests; not a replacement for the real model.
        scores = frame["anxiety_score"].astype(float)
        return np.where(scores >= 8, 2, np.where(scores >= 5, 1, 0))

    def predict_proba(self, frame: pd.DataFrame):
        rows = []
        for pred in self.predict(frame):
            probs = np.array([0.04, 0.04, 0.04], dtype=float)
            probs[int(pred)] = 0.92
            rows.append(probs)
        return np.vstack(rows)


@pytest.fixture()
def clifton(monkeypatch, tmp_path):
    module = _load_clifton_module()
    fake = FakePipeline()
    monkeypatch.setattr(module, "_load_pipeline", lambda: (fake, None))
    monkeypatch.setattr(module, "BATCH_RESULT_DIR", tmp_path)
    return module


def valid_payload(**overrides):
    payload = {
        "panic_attack_history": 0,
        "family_history_mental_illness": 0,
        "anxiety_score": 5,
        "depression_score": 5,
        "work_stress_level": 5,
        "financial_stress_level": 5,
        "sleep_hours": 7.0,
        "social_support_score": 6,
        "physical_activity_hours_per_week": 6.0,
    }
    payload.update(overrides)
    return payload



def test_team_portal_mounts_clifton_application():
    portal_source = (ROOT / "src" / "team5_app" / "app.py").read_text(encoding="utf-8")
    assert '"/clifton": clifton_module.app' in portal_source
    assert '"active": True' in portal_source

def test_submission_contains_real_clifton_model():
    assert MODEL_FILE.exists()
    assert MODEL_FILE.stat().st_size > 100_000


def test_single_and_batch_pages_render(clifton):
    client = clifton.app.test_client()
    single = client.get("/")
    batch = client.get("/batch")
    assert single.status_code == 200
    assert batch.status_code == 200
    assert b"Mental-health risk classification" in single.data
    assert b"CSV batch workflow" in batch.data


def test_schema_contract_has_all_nine_features(clifton):
    response = clifton.app.test_client().get("/schema")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["features"]) == 9
    assert body["prediction_modes"] == ["single", "batch_csv"]


def test_health_is_deep_model_check(clifton):
    response = clifton.app.test_client().get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_loadable"] is True
    assert body["feature_count"] == 9


def test_single_prediction_returns_label_confidence_and_probabilities(clifton):
    response = clifton.app.test_client().post("/predict", json=valid_payload(anxiety_score=9))
    assert response.status_code == 200
    body = response.get_json()
    assert body["class"] == 2
    assert body["label"] == "High"
    assert body["confidence"] == pytest.approx(0.92)
    assert set(body["probabilities"]) == {"Low", "Moderate", "High"}


def test_single_prediction_rejects_out_of_range_value(clifton):
    response = clifton.app.test_client().post("/predict", json=valid_payload(sleep_hours=99))
    assert response.status_code == 400
    assert "between 3 and 10" in response.get_json()["error"]


def test_batch_template_has_exact_model_schema(clifton):
    response = clifton.app.test_client().get("/batch/template")
    assert response.status_code == 200
    template_df = pd.read_csv(BytesIO(response.data))
    assert list(template_df.columns) == clifton.FEATURES
    assert len(template_df) >= 3


def test_batch_prediction_and_download(clifton):
    input_df = pd.DataFrame([
        valid_payload(anxiety_score=3),
        valid_payload(anxiety_score=6),
        valid_payload(anxiety_score=9),
    ])
    buffer = BytesIO(input_df.to_csv(index=False).encode("utf-8"))

    client = clifton.app.test_client()
    response = client.post(
        "/batch/predict",
        data={"file": (buffer, "profiles.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["row_count"] == 3
    assert body["risk_counts"] == {"Low": 1, "Moderate": 1, "High": 1}
    assert body["average_confidence"] == pytest.approx(0.92)
    assert len(body["preview"]) == 3

    downloaded = client.get(body["download_url"])
    assert downloaded.status_code == 200
    output_df = pd.read_csv(BytesIO(downloaded.data))
    assert list(output_df["predicted_risk_label"]) == ["Low", "Moderate", "High"]
    assert "prediction_confidence" in output_df.columns


def test_batch_rejects_missing_required_column(clifton):
    broken = pd.DataFrame([valid_payload()]).drop(columns=["anxiety_score"])
    buffer = BytesIO(broken.to_csv(index=False).encode("utf-8"))
    response = clifton.app.test_client().post(
        "/batch/predict",
        data={"file": (buffer, "broken.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "CSV validation failed."
    assert any("anxiety_score" in message for message in body["messages"])


def test_batch_download_rejects_invalid_id(clifton):
    response = clifton.app.test_client().get("/batch/download/../../etc/passwd")
    assert response.status_code in {404, 308}
