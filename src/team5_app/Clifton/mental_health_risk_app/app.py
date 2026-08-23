"""Clifton's Mental Health Risk Predictor.

The application is mounted at ``/clifton`` by the Team 5 portal and can also
run standalone for development. Predictions are produced from the PyCaret
classification pipeline exported by Clifton's Task 2 notebook.
"""

from __future__ import annotations

import re
import tempfile
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = BASE_DIR / "clifton_mental_health_risk_final_model"
APP_NAME = "Clifton Mental Health Risk Predictor"
APP_VERSION = "1.1.0"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

MAX_UPLOAD_MB = 32
BATCH_RESULT_TTL_SECONDS = 6 * 60 * 60
BATCH_CHUNK_SIZE = 5000
BATCH_PREVIEW_ROWS = 20

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

BATCH_RESULT_DIR = Path(tempfile.gettempdir()) / "it3385_team5_clifton_batch_results"
BATCH_RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model schema: the 9 predictors selected in Clifton's Task 1 EDA / Task 2.
# ---------------------------------------------------------------------------
RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}
RISK_DESCRIPTIONS = {
    "Low": "The model detects a comparatively lower mental-health risk pattern for this profile.",
    "Moderate": "The model detects a mixed pattern that falls into the moderate-risk class.",
    "High": "The model detects a stronger combination of signals associated with the high-risk class.",
}

FIELDS = [
    {
        "name": "panic_attack_history",
        "label": "Panic attack history",
        "kind": "binary",
        "group": "History",
        "help": "Whether the person has a history of panic attacks.",
        "default": 0,
    },
    {
        "name": "family_history_mental_illness",
        "label": "Family history of mental illness",
        "kind": "binary",
        "group": "History",
        "help": "Whether mental illness has been reported in the family.",
        "default": 0,
    },
    {
        "name": "anxiety_score",
        "label": "Anxiety score",
        "kind": "int",
        "group": "Emotional signals",
        "help": "Self-reported score from 1 (low) to 10 (high).",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "depression_score",
        "label": "Depression score",
        "kind": "int",
        "group": "Emotional signals",
        "help": "Self-reported score from 1 (low) to 10 (high).",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "work_stress_level",
        "label": "Work stress level",
        "kind": "int",
        "group": "Emotional signals",
        "help": "Work-related stress score from 1 to 10.",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "financial_stress_level",
        "label": "Financial stress level",
        "kind": "int",
        "group": "Emotional signals",
        "help": "Financial stress score from 1 to 10.",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "sleep_hours",
        "label": "Sleep hours",
        "kind": "float",
        "group": "Protective factors",
        "help": "Average hours of sleep per day.",
        "min": 3,
        "max": 10,
        "step": 0.1,
        "default": 6.5,
    },
    {
        "name": "social_support_score",
        "label": "Social support score",
        "kind": "int",
        "group": "Protective factors",
        "help": "Perceived social support from 1 to 10.",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 6,
    },
    {
        "name": "physical_activity_hours_per_week",
        "label": "Physical activity / week",
        "kind": "float",
        "group": "Protective factors",
        "help": "Total physical activity hours in a typical week.",
        "min": 0,
        "max": 15,
        "step": 0.1,
        "default": 7.5,
    },
]

# Keep the dataframe column order identical to the Task 2 training schema.
# The UI can group fields differently for usability, but model inference always
# receives the exact feature order used when the saved PyCaret pipeline was fit.
FEATURES = [
    "panic_attack_history",
    "anxiety_score",
    "depression_score",
    "sleep_hours",
    "family_history_mental_illness",
    "social_support_score",
    "work_stress_level",
    "financial_stress_level",
    "physical_activity_hours_per_week",
]
FIELD_BY_NAME = {field["name"]: field for field in FIELDS}
DEFAULT_VALUES = {field["name"]: field["default"] for field in FIELDS}
GROUP_ORDER = ["History", "Emotional signals", "Protective factors"]


def grouped_fields() -> dict[str, list[dict[str, Any]]]:
    return {
        group: [field for field in FIELDS if field["group"] == group]
        for group in GROUP_ORDER
    }


# ---------------------------------------------------------------------------
# Runtime and lazy model loading
# ---------------------------------------------------------------------------
def configure_runtime(
    max_upload_mb: int = 32,
    batch_result_ttl_seconds: int = 21600,
    batch_chunk_size: int = 5000,
    batch_preview_rows: int = 20,
) -> None:
    """Apply shared Team 5 runtime settings."""

    global MAX_UPLOAD_MB
    global BATCH_RESULT_TTL_SECONDS
    global BATCH_CHUNK_SIZE
    global BATCH_PREVIEW_ROWS

    MAX_UPLOAD_MB = int(max_upload_mb)
    BATCH_RESULT_TTL_SECONDS = int(batch_result_ttl_seconds)
    BATCH_CHUNK_SIZE = int(batch_chunk_size)
    BATCH_PREVIEW_ROWS = int(batch_preview_rows)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


_model_cache: dict[str, Any] = {"pipeline": None, "error": None}


def _load_pipeline():
    """Load Clifton's PyCaret pipeline once and cache it."""

    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"], None

    model_file = MODEL_NAME.with_suffix(".pkl")
    if not model_file.exists():
        return None, (
            "Clifton's trained model has not been exported yet. Run Clifton's "
            "Task 2 notebook so clifton_mental_health_risk_final_model.pkl is "
            "saved inside the mental_health_risk_app folder."
        )

    try:
        from pycaret.classification import load_model

        pipeline = load_model(str(MODEL_NAME), verbose=False)
        _model_cache["pipeline"] = pipeline
        _model_cache["error"] = None
        return pipeline, None
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to load Clifton's model: {exc}"
        _model_cache["error"] = message
        return None, message


def _model_status() -> dict[str, str | bool]:
    model_file = MODEL_NAME.with_suffix(".pkl")
    if model_file.exists():
        return {
            "ready": True,
            "label": "Model artifact present",
            "detail": f"Saved PyCaret pipeline: {model_file.name}",
        }
    return {
        "ready": False,
        "label": "Model file required",
        "detail": "Run Clifton Task 2 to export the PKL",
    }


# ---------------------------------------------------------------------------
# Validation and prediction helpers
# ---------------------------------------------------------------------------
class BatchValidationError(ValueError):
    def __init__(self, messages: list[str]):
        super().__init__(messages[0] if messages else "Invalid CSV upload.")
        self.messages = messages


def _parse_binary(value: Any) -> int | None:
    text = str(value).strip().casefold()
    mapping = {
        "0": 0,
        "0.0": 0,
        "no": 0,
        "false": 0,
        "n": 0,
        "1": 1,
        "1.0": 1,
        "yes": 1,
        "true": 1,
        "y": 1,
    }
    return mapping.get(text)


def _normalise_column_name(name: Any) -> str:
    text = str(name).strip().casefold()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"[^a-z0-9_]", "", text)


def _clean_value(field: dict[str, Any], raw_value: Any) -> int | float:
    label = field["label"]

    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(f"Please enter a value for {label}.")

    if field["kind"] == "binary":
        value = _parse_binary(raw_value)
        if value is None:
            raise ValueError(f"{label} must be Yes/No or 1/0.")
        return value

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None

    if field["kind"] == "int" and not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")

    if value < field["min"] or value > field["max"]:
        raise ValueError(
            f"{label} must be between {field['min']} and {field['max']}."
        )

    return int(value) if field["kind"] == "int" else float(value)


def validate_single_payload(payload: dict[str, Any]) -> pd.DataFrame:
    cleaned = {
        field["name"]: _clean_value(field, payload.get(field["name"]))
        for field in FIELDS
    }
    return pd.DataFrame([[cleaned[name] for name in FEATURES]], columns=FEATURES)


def validate_batch_dataframe(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if uploaded_df.empty:
        raise BatchValidationError(["The CSV does not contain any data rows."])

    original = uploaded_df.copy()
    working = uploaded_df.copy()
    normalised = [_normalise_column_name(column) for column in working.columns]

    duplicates = sorted({name for name in normalised if normalised.count(name) > 1})
    if duplicates:
        raise BatchValidationError(
            ["Duplicate column names after normalisation: " + ", ".join(duplicates)]
        )

    working.columns = normalised
    missing = [name for name in FEATURES if name not in working.columns]
    if missing:
        raise BatchValidationError(
            ["Missing required columns: " + ", ".join(missing) + "."]
        )

    cleaned = pd.DataFrame(index=working.index)
    errors: list[str] = []

    for field in FIELDS:
        name = field["name"]
        label = field["label"]
        raw = working[name]

        if field["kind"] == "binary":
            mapped = raw.astype("string").str.strip().str.casefold().map(
                {
                    "0": 0,
                    "0.0": 0,
                    "no": 0,
                    "false": 0,
                    "n": 0,
                    "1": 1,
                    "1.0": 1,
                    "yes": 1,
                    "true": 1,
                    "y": 1,
                }
            )
            invalid = raw.isna() | mapped.isna()
            if invalid.any():
                rows = ", ".join(str(i + 2) for i in invalid[invalid].index[:5])
                errors.append(
                    f"{label}: invalid or blank value at CSV row(s) {rows}; use 0/1 or No/Yes."
                )
            cleaned[name] = mapped
            continue

        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna()
        out_of_range = numeric.notna() & (
            (numeric < field["min"]) | (numeric > field["max"])
        )
        non_integer = pd.Series(False, index=numeric.index)
        if field["kind"] == "int":
            non_integer = numeric.notna() & ~np.isclose(numeric, np.round(numeric))

        if invalid.any():
            rows = ", ".join(str(i + 2) for i in invalid[invalid].index[:5])
            errors.append(f"{label}: blank or non-numeric value at CSV row(s) {rows}.")
        if out_of_range.any():
            rows = ", ".join(str(i + 2) for i in out_of_range[out_of_range].index[:5])
            errors.append(
                f"{label}: value outside {field['min']}–{field['max']} at CSV row(s) {rows}."
            )
        if non_integer.any():
            rows = ", ".join(str(i + 2) for i in non_integer[non_integer].index[:5])
            errors.append(f"{label}: whole numbers required at CSV row(s) {rows}.")

        cleaned[name] = (
            numeric.round().astype("Int64")
            if field["kind"] == "int"
            else numeric.astype(float)
        )

    if errors:
        raise BatchValidationError(errors[:12])

    for field in FIELDS:
        if field["kind"] in {"binary", "int"}:
            cleaned[field["name"]] = cleaned[field["name"]].astype(int)

    return cleaned[FEATURES], original


def _label_from_prediction(raw_prediction: Any) -> tuple[int | str, str]:
    try:
        class_value: int | str = int(raw_prediction)
    except (TypeError, ValueError):
        class_value = str(raw_prediction)
    return class_value, RISK_LABELS.get(class_value, str(raw_prediction))


def _probability_payload(pipeline: Any, input_df: pd.DataFrame) -> tuple[float | None, dict[str, float]]:
    try:
        matrix = pipeline.predict_proba(input_df)
        row = np.asarray(matrix)[0]
        classes = list(getattr(pipeline, "classes_", range(len(row))))
        probabilities: dict[str, float] = {}
        for cls, probability in zip(classes, row):
            try:
                cls_key: int | str = int(cls)
            except (TypeError, ValueError):
                cls_key = str(cls)
            label = RISK_LABELS.get(cls_key, str(cls))
            probabilities[label] = float(probability)
        return float(np.max(row)), probabilities
    except Exception:  # noqa: BLE001
        return None, {}


def predict_one(input_df: pd.DataFrame) -> dict[str, Any]:
    pipeline, error = _load_pipeline()
    if error:
        raise RuntimeError(error)

    raw_prediction = pipeline.predict(input_df)[0]
    class_value, label = _label_from_prediction(raw_prediction)
    confidence, probabilities = _probability_payload(pipeline, input_df)

    return {
        "class": class_value,
        "label": label,
        "description": RISK_DESCRIPTIONS.get(label, "Prediction completed."),
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_many(input_df: pd.DataFrame) -> tuple[list[Any], list[str], list[float | None]]:
    pipeline, error = _load_pipeline()
    if error:
        raise RuntimeError(error)

    classes: list[Any] = []
    labels: list[str] = []
    confidences: list[float | None] = []

    for start in range(0, len(input_df), BATCH_CHUNK_SIZE):
        chunk = input_df.iloc[start : start + BATCH_CHUNK_SIZE]
        raw_predictions = pipeline.predict(chunk)
        for raw in raw_predictions:
            class_value, label = _label_from_prediction(raw)
            classes.append(class_value)
            labels.append(label)

        try:
            probability_matrix = np.asarray(pipeline.predict_proba(chunk))
            confidences.extend(np.max(probability_matrix, axis=1).astype(float).tolist())
        except Exception:  # noqa: BLE001
            confidences.extend([None] * len(chunk))

    return classes, labels, confidences


# ---------------------------------------------------------------------------
# Temporary batch-result storage
# ---------------------------------------------------------------------------
def _cleanup_old_batch_results() -> None:
    cutoff = time.time() - BATCH_RESULT_TTL_SECONDS
    for path in BATCH_RESULT_DIR.glob("*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not delete expired result %s", path)


def _save_batch_result(result_df: pd.DataFrame) -> str:
    _cleanup_old_batch_results()
    result_id = uuid.uuid4().hex
    result_df.to_csv(BATCH_RESULT_DIR / f"{result_id}.csv", index=False)
    return result_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template(
        "predictor.html",
        groups=grouped_fields(),
        model_status=_model_status(),
        feature_count=len(FEATURES),
    )


@app.post("/predict")
def predict():
    """Validate one profile and return a real-time model prediction as JSON."""

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(payload, dict):
        return jsonify(error="Prediction payload must be a JSON object or form fields."), 400

    try:
        input_df = validate_single_payload(payload)
        result = predict_one(input_df)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Single prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500

    return jsonify(result)


@app.get("/batch")
def batch_form():
    """Render the CSV batch-prediction workflow."""
    return render_template(
        "batch.html",
        fields=FIELDS,
        required_columns=FEATURES,
        model_status=_model_status(),
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.get("/batch/template")
def batch_template():
    """Download a ready-to-use CSV containing the exact model schema."""
    examples = pd.DataFrame(
        [
            {
                "panic_attack_history": 0,
                "family_history_mental_illness": 0,
                "anxiety_score": 3,
                "depression_score": 3,
                "work_stress_level": 4,
                "financial_stress_level": 3,
                "sleep_hours": 8.0,
                "social_support_score": 8,
                "physical_activity_hours_per_week": 9.0,
            },
            DEFAULT_VALUES,
            {
                "panic_attack_history": 1,
                "family_history_mental_illness": 1,
                "anxiety_score": 9,
                "depression_score": 9,
                "work_stress_level": 8,
                "financial_stress_level": 8,
                "sleep_hours": 4.0,
                "social_support_score": 3,
                "physical_activity_hours_per_week": 2.0,
            },
        ],
        columns=FEATURES,
    )
    buffer = StringIO()
    examples.to_csv(buffer, index=False)
    return app.response_class(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=clifton_batch_template.csv"
        },
    )


@app.post("/batch/predict")
def batch_predict():
    """Validate an uploaded CSV and predict every row in bounded chunks."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Please choose a CSV file first."), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify(error="Only .csv files are accepted."), 400

    try:
        uploaded_df = pd.read_csv(upload)
        model_input, original_df = validate_batch_dataframe(uploaded_df)
        classes, labels, confidences = predict_many(model_input)
    except BatchValidationError as exc:
        return jsonify(error="CSV validation failed.", messages=exc.messages), 400
    except pd.errors.EmptyDataError:
        return jsonify(error="The uploaded CSV is empty."), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Batch prediction failed")
        return jsonify(error=f"Batch prediction failed: {exc}"), 500

    result_df = original_df.copy()
    result_df["predicted_mental_health_risk"] = classes
    result_df["predicted_risk_label"] = labels
    result_df["prediction_confidence"] = [
        round(value, 6) if value is not None else np.nan for value in confidences
    ]

    result_id = _save_batch_result(result_df)
    preview_columns = FEATURES + [
        "predicted_mental_health_risk",
        "predicted_risk_label",
        "prediction_confidence",
    ]
    preview_df = result_df[[c for c in preview_columns if c in result_df.columns]].head(
        BATCH_PREVIEW_ROWS
    )

    # Return a compact class distribution as immediate evidence that the batch
    # request completed across all rows. The full row-level output remains in
    # the downloadable CSV.
    risk_counts = {label: int(labels.count(label)) for label in RISK_LABELS.values()}
    usable_confidences = [value for value in confidences if value is not None]
    average_confidence = (
        float(np.mean(usable_confidences)) if usable_confidences else None
    )

    return jsonify(
        result_id=result_id,
        row_count=len(result_df),
        risk_counts=risk_counts,
        average_confidence=average_confidence,
        preview=preview_df.where(pd.notna(preview_df), None).to_dict(orient="records"),
        download_url=url_for("batch_download", result_id=result_id),
    )


@app.get("/batch/download/<result_id>")
def batch_download(result_id: str):
    """Return a previously generated batch result while its TTL is valid."""
    if not re.fullmatch(r"[0-9a-f]{32}", result_id):
        return jsonify(error="Invalid result ID."), 404

    path = BATCH_RESULT_DIR / f"{result_id}.csv"
    if not path.exists():
        return jsonify(error="Result not found or expired."), 404

    if time.time() - path.stat().st_mtime > BATCH_RESULT_TTL_SECONDS:
        path.unlink(missing_ok=True)
        return jsonify(error="Result expired. Please run the batch prediction again."), 410

    return send_file(
        path,
        as_attachment=True,
        download_name="clifton_mental_health_risk_predictions.csv",
        mimetype="text/csv",
    )


@app.get("/schema")
def schema():
    """Expose the deployed model-input contract for testing and documentation."""

    public_fields = [
        {
            key: field[key]
            for key in ("name", "label", "kind", "min", "max", "step", "default")
            if key in field
        }
        for field in FIELDS
    ]
    return jsonify(
        app=APP_NAME,
        version=APP_VERSION,
        target="mental_health_risk",
        classes=RISK_LABELS,
        features=public_fields,
        prediction_modes=["single", "batch_csv"],
    )


@app.get("/health")
def health():
    """Deep health check: verify that the saved model can actually be loaded."""

    status = _model_status()
    pipeline, load_error = _load_pipeline() if status["ready"] else (None, "Model file missing")
    model_loadable = pipeline is not None and load_error is None
    payload = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "status": "ok" if model_loadable else "degraded",
        "model_artifact_present": bool(status["ready"]),
        "model_loadable": model_loadable,
        "model_file": str(MODEL_NAME.with_suffix(".pkl")),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "prediction_modes": ["single", "batch_csv"],
    }
    if load_error:
        payload["detail"] = str(load_error)
    return jsonify(payload), (200 if model_loadable else 503)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error=f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
