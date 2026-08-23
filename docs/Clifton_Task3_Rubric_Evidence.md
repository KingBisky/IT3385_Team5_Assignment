# Clifton Task 3 — Rubric Evidence Map

## Assignment requirement mapping

- **Front end:** `src/team5_app/Clifton/mental_health_risk_app/templates/`
- **Back end:** `src/team5_app/Clifton/mental_health_risk_app/app.py`
- **Saved ML pipeline:** `clifton_mental_health_risk_final_model.pkl`
- **Real-time single prediction:** `/clifton/predict`
- **Batch prediction:** `/clifton/batch/predict` + downloadable CSV
- **Modern UI:** responsive CSS, grouped input cards, confidence/probability presentation, batch drag/drop and result summary
- **Local testing:** `tests/test_clifton_task3.py`
- **Integrated app:** Clifton is mounted by `src/team5_app/app.py` at `/clifton`
- **Cloud deployment:** repository `Dockerfile` + `src/team5_app/wsgi.py` + Cloud Run instructions in root README

## Advanced rubric mapping

### 1. Source code works flawlessly / organised / detailed comments

Evidence includes:

- central schema (`FIELDS`) used by UI and validators
- strict validation before inference
- lazy/cached model loading
- single and chunked batch inference helpers
- controlled temporary-result TTL
- deterministic download IDs with traversal protection
- safe DOM error rendering
- dedicated deep health and schema endpoints
- route/helper docstrings and explanatory comments
- automated unit/integration tests

### 2. Good programming practices

- `pathlib` for portable paths
- no hard-coded user-specific filesystem paths
- Hydra runtime settings for upload size/chunk/preview/TTL
- model is not reloaded on every request
- input validation occurs server-side rather than trusting HTML/JavaScript
- batch files are bounded by `MAX_CONTENT_LENGTH` and predicted in chunks
- temporary result filenames use random UUIDs and are TTL-cleaned
- output is explicitly labelled educational/non-diagnostic

### 3. Deployed real-time prediction for single and batch requests

Source support is complete. For final marking, redeploy the latest container and capture evidence from the production URL showing:

1. Team portal → Clifton app
2. successful single profile prediction
3. successful batch CSV prediction using `samples/clifton_batch_demo.csv`
4. batch output CSV download
5. `/clifton/health` showing `model_loadable: true`

Use `scripts/verify_clifton_deployment.py <production-base-url>` as a repeatable deployment smoke test.
