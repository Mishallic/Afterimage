"""Every call to the Meta Model API, appended to a ledger as it happens.

A run's ledger shows its work: which model was asked what, when, how long it took, what came
back and roughly what it cost. Media is recorded by SHA-256, never as bytes, and the API key
never appears. Calls made while no run is recording go to out/ledger.jsonl, so none go unlogged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

DEFAULT = ROOT / "out" / "ledger.jsonl"
_active: ContextVar[Path] = ContextVar("ledger", default=DEFAULT)


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def recording(path: Path) -> Iterator[Path]:
    """Send every call made inside this block, including from asyncio tasks it runs, to `path`."""
    token = _active.set(path)
    try:
        yield path
    finally:
        _active.reset(token)


def _scrub(text: str) -> str:
    for name in ("MODEL_API_KEY", "BASETEN_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            text = text.replace(key, "[key]")
    return text


@contextmanager
def call(endpoint: str, purpose: str, *, model: str | None = None, **request) -> Iterator[dict]:
    """Time one API call and append it to the ledger, whether it succeeds or fails.

    `request` describes what was sent; fill the yielded dict with what came back.
    """
    entry: dict = {
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "model": model,
        "endpoint": endpoint,
        "purpose": purpose,
        "request": request,
    }
    response: dict = {}
    started = time.perf_counter()
    try:
        yield response
        entry["status"] = "ok"
    except BaseException as error:
        entry["status"] = "error"
        entry["error"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        entry["ms"] = round(1000 * (time.perf_counter() - started))
        entry["response"] = response
        path = _active.get()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_scrub(json.dumps(entry)) + "\n")


def entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(path: Path) -> dict:
    """What the receipt says about the ledger: its hash, and calls, errors and cost per model.

    Failed calls aren't billed, so they don't count towards the cost.
    """
    calls = entries(path)
    by_model: dict[str, dict] = {}
    for entry in calls:
        totals = by_model.setdefault(
            entry["model"] or entry["endpoint"], {"calls": 0, "errors": 0, "est_cost_usd": 0.0}
        )
        totals["calls"] += 1
        if entry["status"] != "ok":
            totals["errors"] += 1
        else:
            totals["est_cost_usd"] = round(totals["est_cost_usd"] + (entry["request"].get("est_cost_usd") or 0.0), 4)
    return {
        "file": path.name,
        "sha256": file_sha256(path) if path.exists() else None,
        "calls": len(calls),
        "by_model": by_model,
    }
