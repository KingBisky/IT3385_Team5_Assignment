"""Long Chen's Employee Salary Predictor — mounted at /long-chen by the team portal.
Also runs standalone: `python app.py` here serves it directly on its own port.
"""
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import io
import time
import uuid
import tempfile
from pathlib import Path

from schema import FIELDS, grouped_fields

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "employee_salary_final_model")  # pycaret appends .pkl

app = Flask(__name__)

MAX_UPLOAD_MB = 32
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

BATCH_RESULT_DIR = Path(tempfile.gettempdir()) / "it3385_team5_long_chen_batch_results"
BATCH_RESULT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CHUNK_SIZE = 5000
BATCH_PREVIEW_ROWS = 20
BATCH_RESULT_TTL_SECONDS = 3600  # 1 hour

_model_cache = {"pipeline": None, "error": None}

def _load_pipeline():
    if _model_cache["pipeline"] is not None or _model_cache["error"] is not None:
        return _model_cache["pipeline"], _model_cache["error"]

    if not os.path.exists(MODEL_PATH + ".pkl"):
        _model_cache["error"] = (
            f"Model file not found at {MODEL_PATH}.pkl — make sure "
            f"employee_salary_final_model.pkl sits next to this app.py."
        )
        return None, _model_cache["error"]

    try:
        from pycaret.regression import load_model
        pipeline = load_model(MODEL_PATH, verbose=False)
        _model_cache["pipeline"] = pipeline
        return pipeline, None
    except Exception as exc:  # noqa: BLE001
        _model_cache["error"] = f"Failed to load model: {exc}"
        return None, _model_cache["error"]


@app.route("/", methods=["GET"])
def form():
    return render_template("salary_predictor.html", groups=grouped_fields())


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}

    row = {}
    for f in FIELDS:
        raw = payload.get(f["name"], f["default"])
        if f["kind"] == "number":
            try:
                row[f["name"]] = float(raw)
            except (TypeError, ValueError):
                return jsonify(error=f"'{f['label']}' must be a number."), 400
        else:
            row[f["name"]] = str(raw)

    pipeline, error = _load_pipeline()
    if error:
        return jsonify(error=error), 503

    input_df = pd.DataFrame([row])
    try:
        prediction = float(pipeline.predict(input_df)[0])
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Prediction failed: {exc}"), 500

    return jsonify(prediction=prediction, formatted=f"${prediction:,.0f}")

@app.route("/batch", methods=["GET"])
def batch_form():
    return render_template("batch.html")


@app.route("/batch/predict", methods=["POST"])
def batch_predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="No CSV file uploaded."), 400

    pipeline, error = _load_pipeline()
    if error:
        return jsonify(error=error), 503

    try:
        input_df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Could not read CSV: {exc}"), 400

    missing = [f["name"] for f in FIELDS if f["name"] not in input_df.columns]
    if missing:
        return jsonify(error=f"CSV is missing required columns: {missing}"), 400

    # Fill any blank cells with each field's default rather than failing the whole batch
    for f in FIELDS:
        input_df[f["name"]] = input_df[f["name"]].fillna(f["default"])

    predictions = []
    for start in range(0, len(input_df), BATCH_CHUNK_SIZE):
        chunk = input_df.iloc[start:start + BATCH_CHUNK_SIZE]
        try:
            predictions.extend(pipeline.predict(chunk).tolist())
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Prediction failed on rows {start}-{start + len(chunk)}: {exc}"), 500

    input_df["predicted_salary_usd"] = predictions

    result_id = uuid.uuid4().hex
    result_path = BATCH_RESULT_DIR / f"{result_id}.csv"
    input_df.to_csv(result_path, index=False)

    return jsonify(
        result_id=result_id,
        row_count=len(input_df),
        preview=input_df.head(BATCH_PREVIEW_ROWS).to_dict(orient="records"),
        download_url=f"/batch/download/{result_id}",
    )


@app.route("/batch/download/<result_id>", methods=["GET"])
def batch_download(result_id):
    result_path = BATCH_RESULT_DIR / f"{result_id}.csv"
    if not result_path.exists():
        return jsonify(error="Result not found or expired."), 404

    if time.time() - result_path.stat().st_mtime > BATCH_RESULT_TTL_SECONDS:
        result_path.unlink(missing_ok=True)
        return jsonify(error="Result expired."), 410

    return send_file(result_path, as_attachment=True, download_name="salary_predictions.csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)