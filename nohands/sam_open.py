"""SAM 3.1 from its open weights, running on a Modal GPU instead of the Meta Model API.

The deployed app (scripts/sam_modal.py) takes the same proxy video SAM would get through the
API and the same prompts, and returns the masks as a maskfile. Results are cached next to the
run and hashed into the receipt, so verify re-derives everything without a GPU.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import ledger, maskfile
from .occluders import Objects

APP = "nohands-sam"
CLASS = "Tracker"
FUNCTION = "track"
MODEL = "facebook/sam3.1 (open weights, Modal)"
GPU_USD_PER_HOUR = 1.10
"""Rough A10 rate, for the cost column of the ledger; Modal bills per second."""
RETRIES = 6


def _modal():
    try:
        import modal
    except ImportError as error:
        raise RuntimeError("Open-weight masks need the Modal client: pip install modal, then modal setup.") from error
    return modal


def _result(proxy: Path, prompts: list[dict], call_file: Path) -> bytes:
    """Start the remote call, or pick up one started earlier, and wait for its result.

    The call id is kept next to the cache, so a dropped connection (this machine's DNS does
    that) never loses a tracking run: re-running the pipeline resumes waiting for it.
    """
    modal = _modal()
    if call_file.exists():
        call = modal.FunctionCall.from_id(call_file.read_text(encoding="utf-8").strip())
    else:
        call = getattr(modal.Cls.from_name(APP, CLASS)(), FUNCTION).spawn(proxy.read_bytes(), prompts)
        call_file.parent.mkdir(parents=True, exist_ok=True)
        call_file.write_text(call.object_id, encoding="utf-8")
    for attempt in range(1, RETRIES + 1):
        try:
            data = call.get()
            break
        except modal.exception.ConnectionError:
            if attempt == RETRIES:
                raise
            time.sleep(10 * attempt)
    call_file.unlink(missing_ok=True)
    return data


def _track(proxy: Path, prompts: list[dict], out: Path, *, purpose: str, frames: int | None) -> Objects:
    if out.exists():
        return maskfile.load(out)[0]
    with ledger.call(
        f"modal:{APP}/{CLASS}.{FUNCTION}", purpose, model=MODEL,
        file=proxy.name, sha256=ledger.file_sha256(proxy), frames=frames, prompts=prompts,
    ) as logged:
        started = time.monotonic()
        data = _result(proxy, prompts, out.with_suffix(".call"))
        seconds = time.monotonic() - started
        logged["seconds"] = round(seconds, 1)
        logged["est_cost_usd"] = round(seconds / 3600 * GPU_USD_PER_HOUR, 4)
        objects, meta = maskfile.unpack(data)
        logged["objects"] = sum(len(blobs) for blobs in objects.values())
        logged["gpu"] = meta.get("gpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return objects


def track_concepts(proxy: Path, concepts: list[str], out: Path, *, frames: int | None = None) -> Objects:
    """Every instance of each noun phrase, through the whole video; keys are concept:<id>."""
    return _track(
        proxy, [{"text": concept} for concept in concepts], out,
        purpose=f"track {', '.join(repr(c) for c in concepts)}", frames=frames,
    )


def track_boxes(
    proxy: Path, boxes: list[tuple[int, tuple[int, int, int, int]]], out: Path, *, frames: int | None = None
) -> Objects:
    """Specific things from a box on one frame each, both ways through the video; keys are box:<n>."""
    prompts = [{"frame": frame, "box": [int(v) for v in box]} for frame, box in boxes]
    return _track(proxy, prompts, out, purpose=f"track {len(prompts)} boxed object(s)", frames=frames)
