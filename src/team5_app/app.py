"""IT3385 Team 5 integrated Flask application.

This file is the single entry point for the whole team project.  The shared
portal is served at ``/`` and each team member's Flask application can be
mounted under its own route.  Keeping the apps independent means every member
can use a different dataset, preprocessing pipeline and trained model without
mixing files or assumptions between projects.

Run from the project root with::

    python app.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from flask import Flask, render_template
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple


ROOT = Path(__file__).resolve().parent

# Shared team portal. Explicit absolute folders keep the application portable
# when the whole IT3385_Team5 directory is moved to another computer/drive.
portal = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)


# One metadata record per project.  Teammates can later replace their
# placeholder route with a mounted Flask app without changing Kang Bin's data
# or model.  This is the integration layer that lets the team use different
# datasets while still presenting one coherent web application.
TEAM_PROJECTS = [
    {
        "name": "Kang Bin",
        "initials": "KB",
        "status": "Live",
        "title": "Employee Burnout Predictor",
        "description": (
            "Explore employee burnout risk with a trained classification model, "
            "using either one profile or an uploaded CSV for batch prediction."
        ),
        "dataset": "Tech mental-health & burnout dataset",
        "capabilities": ["Single prediction", "Batch CSV", "23 model inputs"],
        "href": "/kang-bin/",
        "active": True,
    },
    {
        "name": "Clifton",
        "initials": "CL",
        "status": "Live",
        "title": "Mental Health Risk Predictor",
        "description": (
            "Classifies mental-health risk as Low, Moderate or High using nine "
            "EDA-selected signals, with both single-profile and batch CSV workflows."
        ),
        "dataset": "Mental health risk dataset",
        "capabilities": ["Single prediction", "Batch CSV", "9 model inputs"],
        "href": "/clifton/",
        "active": True,
    },
    {
        "name": "Long Chen",
        "initials": "LC",
        "status": "Live",
        "title": "Employee Salary Predictor",
        "description": (
            "Estimates an employee's expected salary in real time from role, "
            "company, and market signals using a trained regression pipeline."
        ),
        "dataset": "Employee salary dataset",
        "capabilities": ["Real-time prediction", "Auto re-appraise on input change"],
        "href": "/long-chen/",
        "active": True,
    },
]


@portal.route("/")
def home():
    """Render the shared landing page for all team projects."""
    return render_template("index.html", members=TEAM_PROJECTS)


def load_flask_module(module_name: str, app_file: Path):
    """Load a teammate Flask app from disk without requiring a Python package.

    ``importlib`` is used because team folders contain spaces and should remain
    portable.  Adding the app folder to ``sys.path`` also allows local imports
    inside a teammate's project if they decide to split their code into modules.
    """
    app_dir = app_file.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    spec = importlib.util.spec_from_file_location(module_name, app_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Flask application from {app_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def build_application(cfg: DictConfig):
    kang_bin_module = load_flask_module(
        "kang_bin_employee_burnout",
        ROOT / "Kang Bin" / "employee_burnout_app" / "app.py",
    )

    kang_bin_module.configure_runtime(
        max_upload_mb=cfg.app.max_upload_mb,
        batch_result_ttl_seconds=cfg.app.batch.result_ttl_seconds,
        batch_chunk_size=cfg.app.batch.chunk_size,
        batch_preview_rows=cfg.app.batch.preview_rows,
    )

    clifton_module = load_flask_module(
        "clifton_mental_health_risk",
        ROOT / "Clifton" / "mental_health_risk_app" / "app.py",
    )

    clifton_module.configure_runtime(
        max_upload_mb=cfg.app.max_upload_mb,
        batch_result_ttl_seconds=cfg.app.batch.result_ttl_seconds,
        batch_chunk_size=cfg.app.batch.chunk_size,
        batch_preview_rows=cfg.app.batch.preview_rows,
    )

    long_chen_module = load_flask_module(
        "long_chen_salary_predictor",
        ROOT / "Long Chen" / "salary_predictor_app" / "app.py",
    )

    return DispatcherMiddleware(
        portal,
        {
            "/kang-bin": kang_bin_module.app,
            "/clifton": clifton_module.app,
            "/long-chen": long_chen_module.app,
        },
    )


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    application = build_application(cfg)

    run_simple(
        str(cfg.server.host),
        int(cfg.server.port),
        application,
        use_reloader=bool(cfg.server.use_reloader),
        use_debugger=bool(cfg.server.use_debugger),
        threaded=bool(cfg.server.threaded),
    )


if __name__ == "__main__":
    main()
