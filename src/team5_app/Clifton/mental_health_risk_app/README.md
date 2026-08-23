# Clifton — Mental Health Risk Predictor (Task 3)

This Flask component implements Clifton's IT3385 Task 3 web application and is mounted by the integrated Team 5 portal at `/clifton/`. It loads the complete PyCaret pipeline exported in Task 2 and supports both real-time single-profile prediction and CSV batch prediction.

## Task 3 requirement coverage

The assignment asks for a web front end, a Python back end connected to the trained model, real-time prediction, local testing, cloud/PaaS deployment, a modern user-friendly interface, and an integrated team application. The Advanced rubric additionally expects flawless, organised, well-commented code, good programming practices, and deployed **single and batch** prediction.

| Requirement / Advanced evidence | Clifton implementation |
|---|---|
| Web front end | Responsive Jinja templates in `templates/` with dedicated single and batch workflows |
| Python back end | Flask application in `app.py` |
| Connect to trained pipeline | Lazy-loads `clifton_mental_health_risk_final_model.pkl` exported by Task 2 |
| Real-time single prediction | `POST /clifton/predict` |
| Batch prediction | `POST /clifton/batch/predict`, downloadable output CSV |
| Input validation | Server-side type/range/schema validation for all nine model inputs |
| User-friendly UI | Responsive form groups, inline range guidance, confidence/probability display, batch drag/drop, preview and summary |
| Good code structure | Schema metadata, validation helpers, prediction helpers, temporary-result management, routes and runtime configuration are separated in the source |
| Comments / documentation | Module docstring, function docstrings, comments, this guide and automated tests |
| Team integration | Mounted by `src/team5_app/app.py` under `/clifton` alongside the other team applications |
| Production readiness | Dockerfile + Gunicorn WSGI entrypoint at team-project level; Hydra-managed upload/chunk settings |
| Health evidence | `GET /clifton/health` loads the saved pipeline and reports `model_loadable` |
| Model schema evidence | `GET /clifton/schema` exposes the exact deployed feature contract |

## Saved model

The deployed application uses:

`clifton_mental_health_risk_final_model.pkl`

The file must remain in this directory. The model is loaded lazily through PyCaret so importing the web application does not repeatedly deserialize the pipeline.

## Model inputs

1. `panic_attack_history` — 0/1
2. `anxiety_score` — 1 to 10
3. `depression_score` — 1 to 10
4. `sleep_hours` — 3.0 to 10.0
5. `family_history_mental_illness` — 0/1
6. `social_support_score` — 1 to 10
7. `work_stress_level` — 1 to 10
8. `financial_stress_level` — 1 to 10
9. `physical_activity_hours_per_week` — 0.0 to 15.0

These limits match the Clifton dataset used in Task 1/2.

## Routes

| Route | Method | Purpose |
|---|---:|---|
| `/clifton/` | GET | Single-profile web UI |
| `/clifton/predict` | POST | Real-time single prediction API |
| `/clifton/batch` | GET | Batch CSV web UI |
| `/clifton/batch/template` | GET | Download valid example CSV |
| `/clifton/batch/predict` | POST | Validate and predict uploaded CSV |
| `/clifton/batch/download/<id>` | GET | Download complete batch result |
| `/clifton/schema` | GET | Deployed input/schema contract |
| `/clifton/health` | GET | Deep model-load health check |

## Single-prediction output

The API returns the predicted numerical class, human-readable label (`Low`, `Moderate`, `High`), explanation, confidence and class probabilities when the pipeline exposes `predict_proba`. The browser presents these values in the result panel.

## Batch workflow

1. Open `/clifton/batch`.
2. Download the template if required.
3. Upload a CSV containing the nine required fields. Extra columns are preserved.
4. The server normalises column names and validates every required value before model inference.
5. Prediction runs in configurable chunks.
6. The UI shows row count, Low/Moderate/High distribution, average confidence and a preview.
7. Download the full CSV containing `predicted_mental_health_risk`, `predicted_risk_label`, and `prediction_confidence`.

Generated result files are stored temporarily and expire according to the Hydra TTL setting.

## Local verification

From the project root after `poetry install`:

```bash
poetry run pytest tests/test_clifton_task3.py -v
poetry run python src/team5_app/app.py
```

Then open:

```text
http://127.0.0.1:5000/clifton/
http://127.0.0.1:5000/clifton/batch
http://127.0.0.1:5000/clifton/health
```

A successful `/clifton/health` response should report `status: ok`, `model_artifact_present: true`, and `model_loadable: true`.

## Cloud deployment / Advanced-rubric evidence

The integrated project is containerised with the repository-level `Dockerfile` and production WSGI entry point. After any source update, the final container **must be redeployed** before submission so the live Cloud Run revision contains this Clifton code.

After redeployment, run:

```bash
python scripts/verify_clifton_deployment.py \
  https://team5-app-873480729550.asia-southeast1.run.app
```

The verifier checks the live Clifton page, deep health endpoint, single prediction, batch template, batch prediction and batch-result download. Keep its successful terminal output/screenshots as presentation evidence for the Advanced rubric.

> Educational-use note: this application predicts the class learned from the assignment dataset. It is not a medical diagnosis or clinical decision tool.
