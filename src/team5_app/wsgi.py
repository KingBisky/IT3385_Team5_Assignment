"""WSGI entrypoint for production (gunicorn / Cloud Run).

app.py's main() is decorated with @hydra.main, which parses sys.argv as
Hydra overrides — fine for `python app.py` locally, but gunicorn imports
this module directly with no CLI parsing. This builds the same app via
Hydra's Compose API instead, and exposes a plain WSGI `application`:

    gunicorn --chdir src/team5_app wsgi:application
"""
from pathlib import Path
from hydra import initialize_config_dir, compose

from app import build_application

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
    cfg = compose(config_name="main")  

application = build_application(cfg)