"""Paths, model IDs and API settings shared by the pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
SPARK_OPEN_MODEL = os.environ.get("BASETEN_MODEL", "").strip() or "meta-models/muse-spark-open-1026"
"""The hackathon's Baseten keys unlock the -1026 deployment only; BASETEN_MODEL overrides it."""
SPARK_PROVIDERS = ("meta", "baseten")


def meta_api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MODEL_API_KEY is not set. Copy .env.example to .env and paste your key there."
        )
    return key


def spark_provider() -> str:
    """meta: Muse Spark 1.3 through the Meta Model API. baseten: Muse Spark Open (the open weights) hosted by Baseten."""
    provider = os.environ.get("SPARK_PROVIDER", "").strip().lower() or "meta"
    if provider not in SPARK_PROVIDERS:
        raise RuntimeError(f"SPARK_PROVIDER must be one of {', '.join(SPARK_PROVIDERS)}, not {provider!r}.")
    return provider


def spark_settings() -> tuple[str, str, str]:
    """(base URL, model id, API key) for whichever Spark is in use."""
    if spark_provider() == "baseten":
        key = os.environ.get("BASETEN_API_KEY", "").strip()
        if not key:
            raise RuntimeError("BASETEN_API_KEY is not set; SPARK_PROVIDER=baseten needs it in .env.")
        return BASETEN_BASE_URL, SPARK_OPEN_MODEL, key
    return META_BASE_URL, SPARK_MODEL, meta_api_key()


def find_tool(name: str) -> str:
    """Locate ffmpeg or ffprobe on PATH, falling back to winget's install folder."""
    found = shutil.which(name)
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for candidate in packages.glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"):
        return str(candidate)
    raise RuntimeError(f"{name} not found. Install it with: winget install --id Gyan.FFmpeg -e")
