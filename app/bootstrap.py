from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .constants import APP_TITLE, APP_VERSION
from .data_store import project_paths

paths = project_paths()
templates = Jinja2Templates(directory=str(paths["templates_dir"]))


def create_app() -> FastAPI:
    application = FastAPI(title=APP_TITLE, version=APP_VERSION)
    application.mount("/static", StaticFiles(directory=paths["static_dir"]), name="static")
    return application
