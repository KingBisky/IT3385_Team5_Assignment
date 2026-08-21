from pathlib import Path


def test_project_structure():
    root = Path(__file__).resolve().parents[1]

    assert (root / "pyproject.toml").exists()
    assert (root / "poetry.lock").exists()
    assert (root / "config" / "main.yaml").exists()
    assert (root / "data" / "raw.dvc").exists()
    assert (root / "src" / "team5_app").exists()