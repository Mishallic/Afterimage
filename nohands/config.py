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


def meta_api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MODEL_API_KEY is not set. Copy .env.example to .env and paste your key there."
        )
    return key


def find_tool(name: str) -> str:
    """Locate ffmpeg or ffprobe on PATH, falling back to winget's install folder."""
    found = shutil.which(name)
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for candidate in packages.glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"):
        return str(candidate)
    raise RuntimeError(f"{name} not found. Install it with: winget install --id Gyan.FFmpeg -e")
