"""Employee Burnout Predictor - Flask + PyCaret deployment.

The application exposes two real-time prediction workflows:

1. Single prediction: validate one employee profile submitted from the form.
2. Batch prediction: validate a CSV, predict every valid row, preview the
   results in the browser and provide the complete predictions as a CSV file.

The model, templates and static assets are resolved relative to this file so
that the whole team project remains portable across computers and drive paths.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file
from pycaret.classification import load_model


# ---------------------------------------------------------------------------
# Portable application paths and model loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = BASE_DIR / "employee_burnout_final_model"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# The original coursework dataset is ~16 MB, so the upload limit is large
# enough to accept it while still protecting the local app from huge files.
MAX_UPLOAD_MB = 32
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Results are written to the operating system's temporary directory rather than
# the project folder. This prevents generated batch files from cluttering the
# submission and keeps the application portable.
BATCH_RESULT_DIR = Path(tempfile.gettempdir()) / "it3385_team5_batch_results"
BATCH_RESULT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_RESULT_TTL_SECONDS = 6 * 60 * 60
BATCH_CHUNK_SIZE = 5000
BATCH_PREVIEW_ROWS = 20

model = load_model(str(MODEL_NAME))


# ---------------------------------------------------------------------------
# Model schema
# ---------------------------------------------------------------------------
BURNOUT_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

CATEGORY_OPTIONS = {
    # These values match the categories present in the coursework dataset.
    "gender": ["Female", "Male", "Non-binary"],
    "job_role": [
        "Backend Developer",
        "Data Scientist",
        "DevOps",
        "Frontend Developer",
        "ML Engineer",
        "Product Manager",
        "QA Engineer",
        "Software Engineer",
    ],
    "company_size": ["Startup", "Mid-size", "Large", "MNC"],
    "work_mode": ["Onsite", "Remote", "Hybrid"],
}

NUMERIC_SPECS = {
    "age": {"kind": "int", "min": 22, "max": 54, "step": 1},
    "experience_years": {"kind": "float", "min": 0, "max": 18.5, "step": 0.1},
    "work_hours_per_week": {"kind": "float", "min": 30, "max": 84, "step": 0.1},
    "overtime_hours": {"kind": "float", "min": 0, "max": 24, "step": 0.1},
    "meetings_per_day": {"kind": "int", "min": 0, "max": 12, "step": 1},
    "deadlines_missed": {"kind": "int", "min": 0, "max": 5, "step": 1},
    "job_satisfaction": {"kind": "float", "min": 1, "max": 10, "step": 0.1},
    "manager_support": {"kind": "float", "min": 1, "max": 10, "step": 0.1},
    "work_life_balance": {"kind": "float", "min": 1, "max": 10, "step": 0.1},
    "sleep_hours": {"kind": "float", "min": 3, "max": 10, "step": 0.1},
    "physical_activity_days": {"kind": "int", "min": 0, "max": 7, "step": 1},
    "screen_time_hours": {"kind": "float", "min": 3, "max": 16, "step": 0.1},
    "caffeine_intake": {"kind": "int", "min": 0, "max": 5, "step": 1},
    "social_support_score": {"kind": "float", "min": 1, "max": 10, "step": 0.1},
    "has_therapy": {"kind": "int", "min": 0, "max": 1, "step": 1},
    "stress_level": {"kind": "float", "min": 1, "max": 10, "step": 0.1},
    "anxiety_score": {"kind": "float", "min": 1, "max": 9.6, "step": 0.1},
    "depression_score": {"kind": "float", "min": 1, "max": 8.1, "step": 0.1},
    "seeks_professional_help": {"kind": "int", "min": 0, "max": 1, "step": 1},
}

BINARY_FEATURES = {"has_therapy", "seeks_professional_help"}

FORM_SECTIONS = [
    {
        "title": "Employee Profile",
        "kicker": "01 · Profile",
        "description": "A quick snapshot of the employee and their working setup.",
        "image": "images/profile.svg",
        "fields": [
            {"name": "age", "label": "Age", "type": "number"},
            {"name": "gender", "label": "Gender", "type": "select"},
            {"name": "job_role", "label": "Job Role", "type": "select"},
            {"name": "experience_years", "label": "Experience (years)", "type": "number"},
            {"name": "company_size", "label": "Company Size", "type": "select"},
            {"name": "work_mode", "label": "Work Mode", "type": "select"},
        ],
    },
    {
        "title": "Workload & Environment",
        "kicker": "02 · Work",
        "description": "How demanding the week feels, including support, deadlines and balance.",
        "image": "images/workload.svg",
        "fields": [
            {"name": "work_hours_per_week", "label": "Work Hours / Week", "type": "number"},
            {"name": "overtime_hours", "label": "Overtime Hours", "type": "number"},
            {"name": "meetings_per_day", "label": "Meetings / Day", "type": "number"},
            {"name": "deadlines_missed", "label": "Deadlines Missed", "type": "number"},
            {"name": "job_satisfaction", "label": "Job Satisfaction", "type": "number"},
            {"name": "manager_support", "label": "Manager Support", "type": "number"},
            {"name": "work_life_balance", "label": "Work-Life Balance", "type": "number"},
        ],
    },
    {
        "title": "Lifestyle & Wellbeing",
        "kicker": "03 · Wellbeing",
        "description": "Daily habits and self-reported wellbeing signals used by the model.",
        "image": "images/wellbeing.svg",
        "fields": [
            {"name": "sleep_hours", "label": "Sleep Hours", "type": "number"},
            {"name": "physical_activity_days", "label": "Physical Activity Days / Week", "type": "number"},
            {"name": "screen_time_hours", "label": "Screen Time Hours", "type": "number"},
            {"name": "caffeine_intake", "label": "Caffeine Intake", "type": "number"},
            {"name": "social_support_score", "label": "Social Support Score", "type": "number"},
            {"name": "has_therapy", "label": "Currently Has Therapy", "type": "binary_select"},
            {"name": "stress_level", "label": "Stress Level", "type": "number"},
            {"name": "anxiety_score", "label": "Anxiety Score", "type": "number"},
            {"name": "depression_score", "label": "Depression Score", "type": "number"},
            {"name": "seeks_professional_help", "label": "Seeks Professional Help", "type": "binary_select"},
        ],
    },
]

FEATURES = [field["name"] for section in FORM_SECTIONS for field in section["fields"]]
FIELD_LABELS = {field["name"]: field["label"] for section in FORM_SECTIONS for field in section["fields"]}

DEFAULT_VALUES = {
    "age": 38,
    "gender": "Male",
    "job_role": "Software Engineer",
    "experience_years": 5.0,
    "company_size": "Mid-size",
    "work_mode": "Hybrid",
    "work_hours_per_week": 47.0,
    "overtime_hours": 6.0,
    "meetings_per_day": 4,
    "deadlines_missed": 1,
    "job_satisfaction": 5.5,
    "manager_support": 5.5,
    "work_life_balance": 5.0,
    "sleep_hours": 6.5,
    "physical_activity_days": 2,
    "screen_time_hours": 8.0,
    "caffeine_intake": 2,
    "social_support_score": 5.5,
    "has_therapy": 0,
    "stress_level": 5.8,
    "anxiety_score": 4.5,
    "depression_score": 3.1,
    "seeks_professional_help": 0,
}


# ---------------------------------------------------------------------------
# Validation helpers shared by single and batch prediction
# ---------------------------------------------------------------------------
class BatchValidationError(ValueError):
    """Raised when an uploaded CSV cannot safely be passed to the model."""

    def __init__(self, messages: list[str]):
        super().__init__(messages[0] if messages else "Invalid CSV upload.")
        self.messages = messages


def _canonical_category(feature: str, raw_value: Any) -> Optional[str]:
    """Return the dataset's canonical category while accepting case variation."""
    value = str(raw_value).strip()
    lookup = {option.casefold(): option for option in CATEGORY_OPTIONS[feature]}
    return lookup.get(value.casefold())


def _parse_binary(raw_value: Any) -> Optional[int]:
    """Accept common CSV representations for yes/no fields."""
    text = str(raw_value).strip().casefold()
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


def validate_and_build_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate one form submission and return a one-row model DataFrame."""
    cleaned: dict[str, Any] = {}

    for feature in FEATURES:
        label = FIELD_LABELS[feature]
        if feature not in payload or str(payload[feature]).strip() == "":
            raise ValueError(f"Please enter a value for {label}.")

        raw_value = payload[feature]

        if feature in CATEGORY_OPTIONS:
            value = _canonical_category(feature, raw_value)
            if value is None:
                choices = ", ".join(CATEGORY_OPTIONS[feature])
                raise ValueError(f"{label} must be one of: {choices}.")
            cleaned[feature] = value
            continue

        if feature in BINARY_FEATURES:
            binary_value = _parse_binary(raw_value)
            if binary_value is None:
                raise ValueError(f"{label} must be No/Yes (0/1).")
            cleaned[feature] = binary_value
            continue

        spec = NUMERIC_SPECS[feature]
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number.") from None

        if spec["kind"] == "int" and not numeric_value.is_integer():
            raise ValueError(f"{label} must be a whole number.")
        if numeric_value < spec["min"] or numeric_value > spec["max"]:
            raise ValueError(f"{label} must be between {spec['min']} and {spec['max']}.")

        cleaned[feature] = int(numeric_value) if spec["kind"] == "int" else numeric_value

    return pd.DataFrame([[cleaned[f] for f in FEATURES]], columns=FEATURES)


def _normalise_column_name(name: Any) -> str:
    """Make uploaded headers tolerant of spaces/case while preserving meaning."""
    text = str(name).strip().casefold()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"[^a-z0-9_]", "", text)


def validate_batch_dataframe(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete CSV efficiently and return model-ready features.

    Validation is vectorised instead of looping through every row, which keeps
    large batch requests responsive. Error messages include example CSV row
    numbers so users can correct the file without guessing where the issue is.
    """
    if uploaded_df.empty:
        raise BatchValidationError(["The CSV does not contain any data rows."])

    normalised_columns = [_normalise_column_name(c) for c in uploaded_df.columns]
    duplicates = sorted({c for c in normalised_columns if normalised_columns.count(c) > 1})
    if duplicates:
        raise BatchValidationError([
            "Some column names become duplicates after normalisation: " + ", ".join(duplicates)
        ])

    working = uploaded_df.copy()
    working.columns = normalised_columns

    missing = [feature for feature in FEATURES if feature not in working.columns]
    if missing:
        raise BatchValidationError([
            "Missing required columns: " + ", ".join(missing) + ".",
            "Tip: download the CSV template from the Batch Prediction section and copy your data into it.",
        ])

    cleaned = pd.DataFrame(index=working.index)
    errors: list[str] = []

    for feature in FEATURES:
        label = FIELD_LABELS[feature]
        raw = working[feature]

        if feature in CATEGORY_OPTIONS:
            canonical_lookup = {option.casefold(): option for option in CATEGORY_OPTIONS[feature]}
            text = raw.astype("string").str.strip()
            canonical = text.str.casefold().map(canonical_lookup)
            invalid = raw.isna() | canonical.isna()
            if invalid.any():
                rows = ", ".join(str(i + 2) for i in invalid[invalid].index[:5])
                choices = ", ".join(CATEGORY_OPTIONS[feature])
                errors.append(f"{label}: invalid or blank value at CSV row(s) {rows}. Allowed: {choices}.")
            cleaned[feature] = canonical
            continue

        if feature in BINARY_FEATURES:
            text = raw.astype("string").str.strip().str.casefold()
            binary_map = {
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
            numeric = text.map(binary_map)
            invalid = raw.isna() | numeric.isna()
            if invalid.any():
                rows = ", ".join(str(i + 2) for i in invalid[invalid].index[:5])
                errors.append(f"{label}: invalid or blank value at CSV row(s) {rows}. Use 0/1 or No/Yes.")
            cleaned[feature] = numeric
            continue

        spec = NUMERIC_SPECS[feature]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid_numeric = numeric.isna()
        out_of_range = numeric.notna() & ((numeric < spec["min"]) | (numeric > spec["max"]))
        non_integer = pd.Series(False, index=numeric.index)
        if spec["kind"] == "int":
            non_integer = numeric.notna() & ~np.isclose(numeric, np.round(numeric))

        if invalid_numeric.any():
            rows = ", ".join(str(i + 2) for i in invalid_numeric[invalid_numeric].index[:5])
            errors.append(f"{label}: blank or non-numeric value at CSV row(s) {rows}.")
        if out_of_range.any():
            rows = ", ".join(str(i + 2) for i in out_of_range[out_of_range].index[:5])
            errors.append(
                f"{label}: value outside {spec['min']}–{spec['max']} at CSV row(s) {rows}."
            )
        if non_integer.any():
            rows = ", ".join(str(i + 2) for i in non_integer[non_integer].index[:5])
            errors.append(f"{label}: whole numbers are required at CSV row(s) {rows}.")

        cleaned[feature] = numeric.round().astype("Int64") if spec["kind"] == "int" else numeric.astype(float)

    if errors:
        # Showing every repeated issue from a very large dataset can overwhelm
        # the page. The first issues are enough to identify the columns to fix.
        raise BatchValidationError(errors[:12])

    # Convert nullable integer dtype to ordinary int only after validation has
    # confirmed there are no missing values.
    for feature, spec in NUMERIC_SPECS.items():
        if spec["kind"] == "int":
            cleaned[feature] = cleaned[feature].astype(int)

    return cleaned[FEATURES]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _prediction_label(raw_prediction: Any) -> tuple[Any, str]:
    """Convert model output to the class number and human-readable label."""
    try:
        class_value = int(raw_prediction)
    except (TypeError, ValueError):
        class_value = raw_prediction
    return class_value, BURNOUT_LABELS.get(class_value, str(raw_prediction))


def predict_burnout(data_unseen: pd.DataFrame) -> dict[str, Any]:
    """Run a single real-time prediction and return class, label and confidence."""
    raw_prediction = model.predict(data_unseen)[0]
    predicted_class, predicted_label = _prediction_label(raw_prediction)

    confidence = None
    try:
        probabilities = model.predict_proba(data_unseen)[0]
        confidence = float(np.max(probabilities))
    except Exception:
        # Some estimators do not expose predict_proba. Prediction still works.
        pass

    return {
        "class": predicted_class,
        "label": predicted_label,
        "confidence": confidence,
    }


def predict_batch(data_unseen: pd.DataFrame) -> tuple[list[Any], list[str], list[Optional[float]]]:
    """Predict a large CSV in chunks to avoid unnecessary memory spikes."""
    classes: list[Any] = []
    labels: list[str] = []
    confidences: list[Optional[float]] = []

    for start in range(0, len(data_unseen), BATCH_CHUNK_SIZE):
        chunk = data_unseen.iloc[start : start + BATCH_CHUNK_SIZE]
        raw_predictions = model.predict(chunk)

        for raw_prediction in raw_predictions:
            class_value, label = _prediction_label(raw_prediction)
            classes.append(class_value)
            labels.append(label)

        try:
            probability_matrix = model.predict_proba(chunk)
            confidences.extend(np.max(probability_matrix, axis=1).astype(float).tolist())
        except Exception:
            confidences.extend([None] * len(chunk))

    return classes, labels, confidences


# ---------------------------------------------------------------------------
# Batch-result file management
# ---------------------------------------------------------------------------
def _cleanup_old_batch_results() -> None:
    """Remove old generated CSV files so temporary storage stays bounded."""
    cutoff = time.time() - BATCH_RESULT_TTL_SECONDS
    for path in BATCH_RESULT_DIR.glob("*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not clean temporary batch result: %s", path)


def _save_batch_result(result_df: pd.DataFrame) -> str:
    """Store a prediction CSV temporarily and return an unguessable token."""
    _cleanup_old_batch_results()
    token = uuid.uuid4().hex
    result_path = BATCH_RESULT_DIR / f"{token}.csv"
    result_df.to_csv(result_path, index=False)
    return token


# ---------------------------------------------------------------------------
# Page rendering and routes
# ---------------------------------------------------------------------------
def render_page(
    form_values: dict[str, Any],
    prediction: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
    batch_error: Optional[list[str]] = None,
    batch_result: Optional[dict[str, Any]] = None,
):
    """Render the single shared page so both prediction modes feel integrated."""
    return render_template(
        "home.html",
        form_sections=FORM_SECTIONS,
        category_options=CATEGORY_OPTIONS,
        numeric_specs=NUMERIC_SPECS,
        form_values=form_values,
        prediction=prediction,
        error=error,
        factor_count=len(FEATURES),
        required_columns=FEATURES,
        max_upload_mb=MAX_UPLOAD_MB,
        batch_error=batch_error,
        batch_result=batch_result,
    )


@app.route("/")
def home():
    return render_page(DEFAULT_VALUES.copy())


@app.route("/team-home")
def team_home():
    """Return to the Team 5 portal when mounted under /kang-bin/."""
    return redirect(request.host_url)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle one employee profile and return a prediction immediately."""
    submitted_values = DEFAULT_VALUES.copy()
    submitted_values.update(request.form.to_dict())

    try:
        data_unseen = validate_and_build_dataframe(request.form.to_dict())
        prediction = predict_burnout(data_unseen)
        return render_page(submitted_values, prediction=prediction)
    except ValueError as exc:
        return render_page(submitted_values, error=str(exc)), 400
    except Exception:
        app.logger.exception("Single prediction failed")
        return render_page(
            submitted_values,
            error="The prediction could not be completed. Please check the inputs and try again.",
        ), 500


@app.route("/batch-template")
def batch_template():
    """Download a model-ready CSV template with two editable example rows."""
    second_row = DEFAULT_VALUES.copy()
    second_row.update(
        {
            "age": 29,
            "gender": "Female",
            "job_role": "Data Scientist",
            "experience_years": 3.5,
            "company_size": "Startup",
            "work_mode": "Remote",
            "work_hours_per_week": 52.0,
            "overtime_hours": 8.0,
            "stress_level": 7.2,
            "sleep_hours": 5.8,
        }
    )
    template_df = pd.DataFrame([DEFAULT_VALUES, second_row], columns=FEATURES)
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")

    from io import BytesIO

    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="employee_burnout_batch_template.csv",
    )


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Validate an uploaded CSV and generate real-time predictions for all rows."""
    uploaded_file = request.files.get("batch_file")

    if uploaded_file is None or not uploaded_file.filename:
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=["Choose a CSV file before starting the batch prediction."],
        ), 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=["Only .csv files are supported for batch prediction."],
        ), 400

    try:
        # utf-8-sig transparently handles CSVs exported by Excel with a BOM.
        text_stream = StringIO(uploaded_file.stream.read().decode("utf-8-sig"))
        original_df = pd.read_csv(text_stream)
        model_df = validate_batch_dataframe(original_df)

        classes, labels, confidences = predict_batch(model_df)

        # Preserve every original uploaded column. This means users may upload
        # the full coursework dataset (including burnout_score/burnout_level);
        # only the 23 required predictor columns are sent to the model.
        result_df = original_df.copy()
        result_df["predicted_burnout_class"] = classes
        result_df["predicted_burnout_level"] = labels
        if any(value is not None for value in confidences):
            result_df["prediction_confidence"] = [
                round(float(value), 6) if value is not None else np.nan for value in confidences
            ]

        token = _save_batch_result(result_df)
        counts = pd.Series(labels).value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)
        valid_confidences = [value for value in confidences if value is not None]

        preview_columns = [
            c
            for c in [
                "age",
                "job_role",
                "work_mode",
                "work_hours_per_week",
                "stress_level",
                "predicted_burnout_level",
                "prediction_confidence",
            ]
            if c in result_df.columns
        ]
        preview_df = result_df[preview_columns].head(BATCH_PREVIEW_ROWS).copy()
        if "prediction_confidence" in preview_df.columns:
            preview_df["prediction_confidence"] = preview_df["prediction_confidence"].map(
                lambda value: "—" if pd.isna(value) else f"{float(value) * 100:.1f}%"
            )

        batch_result = {
            "filename": uploaded_file.filename,
            "rows": len(result_df),
            "low": int(counts["Low"]),
            "moderate": int(counts["Moderate"]),
            "high": int(counts["High"]),
            "average_confidence": (
                float(np.mean(valid_confidences)) if valid_confidences else None
            ),
            "preview_columns": preview_columns,
            "preview_rows": preview_df.to_dict(orient="records"),
            "preview_limit": BATCH_PREVIEW_ROWS,
            "download_token": token,
        }
        return render_page(DEFAULT_VALUES.copy(), batch_result=batch_result)

    except UnicodeDecodeError:
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=["The CSV could not be read as UTF-8 text. Re-save it as a standard UTF-8 CSV and try again."],
        ), 400
    except pd.errors.EmptyDataError:
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=["The uploaded CSV is empty."],
        ), 400
    except pd.errors.ParserError as exc:
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=[f"The CSV structure could not be parsed: {exc}"],
        ), 400
    except BatchValidationError as exc:
        return render_page(DEFAULT_VALUES.copy(), batch_error=exc.messages), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return render_page(
            DEFAULT_VALUES.copy(),
            batch_error=[
                "The batch prediction could not be completed. Check the CSV format and try again."
            ],
        ), 500


@app.route("/batch-results/<token>.csv")
def download_batch_result(token: str):
    """Download a previously generated batch result using its random token."""
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        abort(404)

    result_path = BATCH_RESULT_DIR / f"{token}.csv"
    if not result_path.exists():
        abort(404)

    return send_file(
        result_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="employee_burnout_batch_predictions.csv",
    )


@app.errorhandler(413)
def upload_too_large(_error):
    """Return a friendly message instead of Flask's default 413 response."""
    return render_page(
        DEFAULT_VALUES.copy(),
        batch_error=[f"The CSV is too large. The maximum upload size is {MAX_UPLOAD_MB} MB."],
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
