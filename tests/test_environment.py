from pathlib import Path

from hydra import compose, initialize_config_dir


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


def test_hydra_configuration():
    root = Path(__file__).resolve().parents[1]

    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "config"),
    ):
        cfg = compose(config_name="main")

    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 5000
    assert cfg.app.max_upload_mb == 32
    assert cfg.app.batch.result_ttl_seconds == 21600
    assert cfg.app.batch.chunk_size == 5000
    assert cfg.app.batch.preview_rows == 20


def test_hydra_overrides():
    root = Path(__file__).resolve().parents[1]

    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "config"),
    ):
        cfg = compose(
            config_name="main",
            overrides=[
                "server.port=5050",
                "app.batch.chunk_size=10000",
            ],
        )

    assert cfg.server.port == 5050
    assert cfg.app.batch.chunk_size == 10000