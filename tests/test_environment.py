from pathlib import Path


def test_project_structure():
    root = Path(__file__).resolve().parents[1]

    assert (root / "pyproject.toml").exists()
    assert (root / "poetry.lock").exists()
    assert (root / "config" / "main.yaml").exists()
    assert (root / "src" / "team5_app").exists()

    dvc_files = [
        root / "data" / "raw" / "Kang Bin" / "tech_mental_health_burnout.csv.dvc",
        root / "data" / "raw" / "Clifton" / "mental_health_risk_dataset.csv.dvc",
        root / "data" / "raw" / "Long Chen" / "global_ai_jobs.csv.dvc",
    ]

    for dvc_file in dvc_files:
        assert dvc_file.exists()