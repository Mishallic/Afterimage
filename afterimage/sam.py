"""SAM 3.1 video segmentation through the Meta Model API, with an on-disk event cache.

Every streamed event is saved as NDJSON before parsing, so a session is paid for once
and can be re-parsed offline as often as we like.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    SegmentationMaskRecord,
    VideoSegmentationResult,
    build_object_prompt,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)
import openai
from openai import AsyncOpenAI

from . import ledger
from .config import META_BASE_URL, SAM_MODEL, meta_api_key
from .masks import Blob, resize_nearest

try:  # the parser's decoder without its canonical re-encode (see _raster)
    from meta_sam_parser._mask_codec import _decode_raster
except ImportError:  # pragma: no cover - older or newer parser layouts
    _decode_raster = None

try:
    from meta_sam_parser import InvalidSegmentationMaskError
    from meta_sam_parser import _segmentation as _parser_segmentation
except ImportError:  # pragma: no cover - older or newer parser layouts
    _parser_segmentation = None


def _mask_shape_only(mask) -> bytes:
    """Stands in for the parser's per-mask validation while a stream is read.

    The parser decodes every mask, then re-encodes it to check it is canonical, and throws the
    pixels away: pure Python, a context-coded bit per pixel, an hour and a half for one concept
    over an 84-minute session, before any mask is used. The masks are decoded once in
    frame_objects, in parallel, where a bad payload fails instead. The stream file itself is
    hashed into the receipt.
    """
    payload = getattr(mask, "payload", None)
    if not isinstance(payload, str) or mask.width <= 0 or mask.height <= 0:
        raise InvalidSegmentationMaskError("Mask payload must be a string with positive dimensions.")
    prefix = {"one_bit": "!", "lossless": "~"}.get(mask.encoding)
    if prefix is None or not payload.startswith(prefix):
        raise InvalidSegmentationMaskError(f"Unsupported or malformed {mask.encoding} mask payload.")
    return b""


if _parser_segmentation is not None:
    _parser_segmentation.decode_mask_to_raster = _mask_shape_only

PRICE_PER_FRAME = 0.20 / 1000
MIN_BILLED_FRAMES = 50
MAX_FRAMES_PER_CALL = 15_000
MAX_OBJECTS_PER_CALL = 16
STREAM_READ_TIMEOUT = 300.0
"""Seconds without a streamed event before a SAM call counts as stalled. A full session streams
an event every few seconds; on Oct 30 one call sat silent for 36 minutes and would have for ever."""
STREAM_ATTEMPTS = 5
"""Tries per stream; waits of 30, 60, 90 and 120 s between them ride out a dropped connection."""
"""A stalled or dropped stream is started over, this many times at most; each attempt is in the ledger."""


def client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=META_BASE_URL, api_key=meta_api_key(),
        timeout=STREAM_READ_TIMEOUT, max_retries=2,
    )


class StreamIncomplete(RuntimeError):
    """The stream ended without SAM's final response: the connection went, not the job."""


def estimated_cost(frames: int) -> float:
    return max(frames, MIN_BILLED_FRAMES) * PRICE_PER_FRAME


def slug(concept: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", concept.lower()).strip("-")


async def upload_cached(
    api: AsyncOpenAI, video: Path, id_file: Path, *, expires_days: int = 7
) -> str:
    """Upload once and remember the file id next to the outputs."""
    if id_file.exists():
        return id_file.read_text(encoding="utf-8").strip()
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        try:
            with ledger.call(
                "files.create", f"upload {video.name} for SAM" + ("" if attempt == 1 else f" (attempt {attempt})"),
                file=video.name, bytes=video.stat().st_size, sha256=ledger.file_sha256(video),
            ) as logged, video.open("rb") as handle:
                uploaded = await api.files.create(
                    purpose="user_data",
                    file=handle,
                    expires_after={"anchor": "created_at", "seconds": expires_days * 86_400},
                )
                logged["file_id"] = uploaded.id
            break
        except (openai.APITimeoutError, openai.APIConnectionError):
            if attempt == STREAM_ATTEMPTS:
                raise
            await asyncio.sleep(min(120, 30 * attempt))
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(uploaded.id, encoding="utf-8")
    return uploaded.id


async def _stream_events(
    api: AsyncOpenAI, file_id: str, prompt: str, direction: str
) -> AsyncIterator[dict]:
    stream = await api.responses.create(
        model=SAM_MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
        stream=True,
        metadata={"propagation_direction": direction, "mask_encoding": "one_bit"},
    )
    async for event in stream:
        yield event.model_dump(mode="json")


async def _replay(events_file: Path) -> AsyncIterator[dict]:
    with events_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


async def parse_cached(events_file: Path) -> VideoSegmentationResult:
    return await parse_responses_stream(_replay(events_file), video_segmentation_format()).final_result()


async def _segment(
    api: AsyncOpenAI, file_id: str, prompt: str, events_file: Path, direction: str,
    *, purpose: str, frames: int | None,
) -> VideoSegmentationResult:
    if not events_file.exists():
        events_file.parent.mkdir(parents=True, exist_ok=True)
        partial = events_file.with_suffix(".partial")
        for attempt in range(1, STREAM_ATTEMPTS + 1):
            try:
                await _record(api, file_id, prompt, direction, partial, purpose=purpose, frames=frames, attempt=attempt)
                break
            except (openai.APITimeoutError, openai.APIConnectionError, StreamIncomplete):
                partial.unlink(missing_ok=True)
                if attempt == STREAM_ATTEMPTS:
                    raise
                await asyncio.sleep(min(120, 30 * attempt))
        partial.replace(events_file)
    return await parse_cached(events_file)


async def _record(
    api: AsyncOpenAI, file_id: str, prompt: str, direction: str, partial: Path,
    *, purpose: str, frames: int | None, attempt: int,
) -> None:
    """One streamed SAM call written to `partial`, event by event, with its own ledger entry."""
    with ledger.call(
        "responses.create", purpose if attempt == 1 else f"{purpose} (attempt {attempt})", model=SAM_MODEL,
        prompt=prompt, file_id=file_id, direction=direction, frames=frames,
        est_cost_usd=None if frames is None else round(estimated_cost(frames), 4),
    ) as logged:
        count = 0
        with partial.open("w", encoding="utf-8") as handle:
            async for event in _stream_events(api, file_id, prompt, direction):
                handle.write(json.dumps(event) + "\n")
                count += 1
                body = event.get("response")
                if isinstance(body, dict):
                    logged["id"] = body.get("id") or logged.get("id")
                    if body.get("usage"):
                        logged["usage"] = body["usage"]
        logged["events"] = count
        logged["events_sha256"] = ledger.file_sha256(partial)
        if "usage" not in logged:
            raise StreamIncomplete(f"{purpose}: {count} events, then the stream ended without the final response")


async def segment_concept(
    api: AsyncOpenAI,
    file_id: str,
    concept: str,
    events_file: Path,
    *,
    direction: str = "forward",
    frames: int | None = None,
) -> VideoSegmentationResult:
    """Track every instance of one noun phrase through the video (SAM takes one concept per call)."""
    return await _segment(
        api, file_id, concept, events_file, direction, purpose=f"track {concept!r}", frames=frames
    )


def box_prompt(size: tuple[int, int], boxes: list[tuple[int, tuple[int, int, int, int]]]) -> str:
    """SAM's object-prompt text for (frame, (left, top, right, bottom)) boxes, one object each."""
    objects = [
        PromptObject(id=number, prompts=[FramePrompt(frame=frame, box=PromptBox(*box))])
        for number, (frame, box) in enumerate(boxes[:MAX_OBJECTS_PER_CALL], start=1)
    ]
    return build_object_prompt(size=size, objects=objects)


async def segment_boxes(
    api: AsyncOpenAI,
    file_id: str,
    size: tuple[int, int],
    boxes: list[tuple[int, tuple[int, int, int, int]]],
    events_file: Path,
    *,
    frames: int | None = None,
) -> VideoSegmentationResult:
    """Track specific things from a box on one frame each, both ways through the video.

    This is how the critic's findings come back: Spark says where a leftover is,
    SAM follows it through every frame it appears in.
    """
    return await _segment(
        api, file_id, box_prompt(size, boxes), events_file, "both",
        purpose=f"track {min(len(boxes), MAX_OBJECTS_PER_CALL)} boxed object(s)", frames=frames,
    )


DECODE_WORKERS = 6
"""Processes decoding masks side by side. The reference decoder is pure Python, a context-coded bit
per pixel: a person's masks over an 84-minute session took one core an hour and a half."""
PARALLEL_FROM = 64
"""Fewer records than this are decoded in-process; a pool is not worth its start-up."""


def frame_objects(result: VideoSegmentationResult, *, prefix: str = "") -> dict[int, dict[str, Blob]]:
    """Each object's mask per frame, keyed by `prefix` + SAM's object id."""
    latest: dict[tuple[int, str], SegmentationMaskRecord] = {}
    for record in result.records:
        if not isinstance(record, SegmentationMaskRecord) or record.frame is None:
            continue
        key = (record.frame.frame_index, record.object_id)
        if key not in latest or record.revision > latest[key].revision:
            latest[key] = record
    keys = list(latest)
    objects: dict[int, dict[str, Blob]] = {}
    for (frame_index, object_id), blob in zip(keys, _decode_all([latest[key] for key in keys])):
        if blob is not None and blob.area:
            objects.setdefault(frame_index, {})[prefix + object_id] = blob
    return objects


def _decode_all(records: list[SegmentationMaskRecord]) -> list[Blob | None]:
    """The records' masks as blobs, decoded in parallel when there are many."""
    if len(records) < PARALLEL_FROM:
        return [_blob(record) for record in records]
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, min(os.cpu_count() or 1, DECODE_WORKERS))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_blob, records, chunksize=32))


def frame_masks(result: VideoSegmentationResult, size: tuple[int, int]) -> dict[int, np.ndarray]:
    """Union of every object's mask per frame, as bool arrays of shape (height, width)."""
    width, height = size
    masks: dict[int, np.ndarray] = {}
    for frame_index, objects in frame_objects(result).items():
        target = np.zeros((height, width), dtype=bool)
        for blob in objects.values():
            blob.paste_into(target)
        masks[frame_index] = target
    return masks


def _raster(mask) -> bytes:
    """The mask's pixels. The strict decoder re-encodes every mask to check it is canonical, which
    doubles the cost; the payload is already hashed as part of the event stream, so decode only."""
    if mask.encoding == "one_bit" and _decode_raster is not None and isinstance(mask.payload, str):
        return _decode_raster(mask.payload, int(mask.width), int(mask.height))
    return decode_mask_to_raster(mask)


def _blob(record: SegmentationMaskRecord) -> Blob | None:
    # Masks come box-local: a small raster plus the box it belongs in, in source pixels.
    raster = np.frombuffer(_raster(record.mask), dtype=np.uint8).astype(bool)
    raster = raster.reshape(record.mask.height, record.mask.width)
    bounds = record.bounds
    left, top = round(bounds.left), round(bounds.top)
    right, bottom = round(bounds.right), round(bounds.bottom)
    if right <= left or bottom <= top:
        return None
    return Blob(left, top, resize_nearest(raster, (bottom - top, right - left)))
