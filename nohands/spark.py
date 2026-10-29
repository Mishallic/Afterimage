"""Muse Spark's jobs: scan the session, criticise the reconstruction, name the chapters.

Spark decides *what* matters and *where*; SAM 3.1 turns that into pixel-accurate masks.
That split (reason with Spark, segment with SAM) is the pattern Meta's docs recommend.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import imaging, ledger, media
from .config import META_BASE_URL, SPARK_MODEL, meta_api_key

GRID = 1000
"""Spark's grounding coordinates run 0..1000 across whatever image or video it was sent."""
SCAN_FRAMES = 96
SCAN_FPS = 2.0
CRITIC_WIDTH = 1280
OCCLUDER_KINDS = ["body", "hand", "brush", "tool", "animal", "other"]
TRACKED_ANYWAY = ("body", "hand", "brush")
"""Kinds the default SAM concepts (person, hand, paintbrush) already cover."""
MAX_EXTRA_CONCEPTS = 3
"""Each extra concept is another SAM pass over the whole session: $0.72 per footage-hour at 1 fps."""

_BOX = {"type": "array", "items": {"type": "number"}}

SCAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["occluders", "canvas_corners"],
    "properties": {
        "occluders": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phrase", "kind", "sightings"],
                "properties": {
                    "phrase": {"type": "string"},
                    "kind": {"type": "string", "enum": OCCLUDER_KINDS},
                    "sightings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["second", "box"],
                            "properties": {"second": {"type": "number"}, "box": _BOX},
                        },
                    },
                },
            },
        },
        "canvas_corners": {"type": "array", "items": _BOX},
    },
}

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["defects"],
    "properties": {
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["image", "box", "kind", "confidence"],
                "properties": {
                    "image": {"type": "integer"},
                    "box": _BOX,
                    "kind": {
                        "type": "string",
                        "enum": ["hand", "arm", "body", "brush", "tool", "shadow", "glare", "smear", "other"],
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}

CHAPTERS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chapters"],
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "start_second", "end_second"],
                "properties": {
                    "title": {"type": "string"},
                    "start_second": {"type": "number"},
                    "end_second": {"type": "number"},
                },
            },
        },
    },
}

SCAN_PROMPT = """You are watching a timelapse of one painting session, filmed from a fixed camera.
Find every physical thing that, at any moment, is between the camera and the painting surface:
the painter's body and hands, tools (brushes, palette knife, rag, palette, cup) and anything else
(a pet, a phone). Do NOT include anything painted on the canvas, even if it shows hands or people.
For each, give a short noun phrase a segmentation model can use (like "person", "paintbrush",
"rag"), its kind (body: the painter's body, arms or clothes; hand; brush; tool: any other object
they hold or put down; animal; other) and up to 3 sightings: the second in this video and a box
[x_min, y_min, x_max, y_max] on a 0-1000 grid with (0,0) at the top-left. Also give the four
corners of the painting surface as [x, y] points on the same grid, ordered top-left, top-right,
bottom-right, bottom-left."""

CRITIC_PROMPT = """You are an object grounding expert checking a video reconstruction.
Each image shows the camera frame on the LEFT and our reconstruction of the painting on the RIGHT.
The reconstruction must contain only paint: no part of a person, hand, arm, brush or tool, and no
shadow, glare or smear that isn't paint. For every leftover on the RIGHT half, report the image
number, a box [x_min, y_min, x_max, y_max] on a 0-1000 grid relative to the RIGHT half only, what
it is, and your confidence from 0 to 1. Things genuinely painted on the canvas (including painted
hands or figures) are not leftovers. Don't miss small fragments like a brush tip. If an image is
clean, report nothing for it."""

CHAPTERS_PROMPT = """This timelapse shows one painting session from start to finish.
Split it into 3 to 6 phases a painter would recognise (for example sketch, block-in, background,
details, final touches). For each, give a short title and its start and end second in this video."""


@dataclass(frozen=True)
class Sighting:
    frame: int
    """Frame index in the video Spark's answer is mapped back to."""
    box: tuple[int, int, int, int]
    """(left, top, right, bottom) in that video's pixels, half-open."""


@dataclass(frozen=True)
class Occluder:
    phrase: str
    kind: str
    sightings: list[Sighting]


@dataclass(frozen=True)
class Scan:
    occluders: list[Occluder]
    canvas_corners: np.ndarray | None
    """Four (x, y) points as fractions of the frame, or None if Spark didn't find a canvas."""


@dataclass(frozen=True)
class Defect:
    keyframe: int
    """Position in the list of keyframes that was sent."""
    box: tuple[int, int, int, int]
    """(left, top, right, bottom) in the clean frame's pixels, half-open."""
    kind: str
    confidence: float

    @property
    def is_object(self) -> bool:
        """Objects can be tracked by SAM; shadows, glare and smears are handled by holding the area."""
        return self.kind in ("hand", "arm", "body", "brush", "tool")


def client() -> OpenAI:
    return OpenAI(base_url=META_BASE_URL, api_key=meta_api_key())


def upload(api: OpenAI, path: Path, *, expires_days: int = 7) -> str:
    with ledger.call(
        "files.create", f"upload {path.name} for Spark",
        file=path.name, bytes=path.stat().st_size, sha256=ledger.file_sha256(path),
    ) as logged, path.open("rb") as handle:
        uploaded = api.files.create(
            purpose="user_data",
            file=handle,
            expires_after={"anchor": "created_at", "seconds": expires_days * 86_400},
        )
        logged["file_id"] = uploaded.id
    return uploaded.id


def image_block(frame: np.ndarray) -> dict:
    data = base64.b64encode(media.encode_jpeg(frame)).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}


def _described(content: list[dict]) -> dict:
    """What the ledger keeps of a request: the instructions, and media only by hash or file id."""
    texts = [block["text"] for block in content if block["type"] == "input_text"]
    return {
        "prompt": texts[0] if texts else "",
        "images": [ledger.sha256(block["image_url"]) for block in content if block["type"] == "input_image"],
        "files": [block["file_id"] for block in content if block["type"] in ("input_video", "input_file")],
    }


def ask_json(
    api: OpenAI, content: list[dict], schema: dict, name: str, *, effort: str = "medium",
    max_output_tokens: int = 8000,
) -> dict:
    with ledger.call(
        "responses.create", name, model=SPARK_MODEL,
        effort=effort, max_output_tokens=max_output_tokens, **_described(content),
    ) as logged:
        response = api.responses.create(
            model=SPARK_MODEL,
            input=[{"type": "message", "role": "user", "content": content}],
            reasoning={"effort": effort},
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        )
        logged["id"] = response.id
        usage = getattr(response, "usage", None)
        logged["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        answer = json.loads(response.output_text)
        logged["answer"] = answer
    return answer


def to_pixels(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A 0..1000 box from Spark as a whole-pixel, half-open box inside an image of `size`."""
    width, height = size
    x0, y0, x1, y1 = (min(max(float(value), 0.0), GRID) for value in box[:4])
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    left, top = min(int(x0 * width / GRID), width - 1), min(int(y0 * height / GRID), height - 1)
    right = min(max(left + 1, round(x1 * width / GRID)), width)
    bottom = min(max(top + 1, round(y1 * height / GRID)), height)
    return left, top, right, bottom


def scan_picks(frames: int, count: int = SCAN_FRAMES) -> list[int]:
    """Which session frames go into the scan video: evenly spread, first and last included."""
    if frames <= count:
        return list(range(frames))
    return sorted(set(np.linspace(0, frames - 1, count).round().astype(int).tolist()))


def make_scan_video(source: Path, out: Path, *, fps: float, size: tuple[int, int], frames: int) -> list[int]:
    """A short video of evenly spread frames, short enough for Spark to see every one of them.

    Spark's video sampling isn't documented; its open-weight sibling reads 2 fps and about 96
    frames, so 96 frames at 2 fps (48 seconds) keeps every picked frame in view.
    """
    picks = scan_picks(frames)
    with media.FrameWriter(out, size, SCAN_FPS) as writer:
        for frame in media.read_indices(source, picks, fps=fps, size=size):
            writer.write(frame)
    return picks


def _scan_frame(second: float, picks: list[int]) -> int:
    return picks[min(max(round(second * SCAN_FPS), 0), len(picks) - 1)]


def scan_answer(api: OpenAI, scan_video: Path) -> dict:
    """Spark's raw answer about a scan video, for caching; parse_scan turns it into a Scan."""
    return ask_json(
        api,
        [{"type": "input_text", "text": SCAN_PROMPT}, {"type": "input_video", "file_id": upload(api, scan_video)}],
        SCAN_SCHEMA,
        "occluder_scan",
    )


def parse_scan(answer: dict, picks: list[int], size: tuple[int, int]) -> Scan:
    """A scan answer in session frames (`picks` made the scan video) and pixels of `size`."""
    occluders = [
        Occluder(
            phrase=" ".join(item["phrase"].lower().split()),
            kind=item["kind"],
            sightings=[
                Sighting(_scan_frame(sighting["second"], picks), to_pixels(sighting["box"], size))
                for sighting in item["sightings"][:3]
                if len(sighting["box"]) >= 4
            ],
        )
        for item in answer["occluders"]
        if item["phrase"].strip()
    ]
    corners = [point[:2] for point in answer["canvas_corners"] if len(point) >= 2]
    canvas = np.clip(np.array(corners, dtype=np.float64) / GRID, 0, 1) if len(corners) == 4 else None
    return Scan(occluders, canvas)


def extra_concepts(scan: Scan, existing: tuple[str, ...], limit: int = MAX_EXTRA_CONCEPTS) -> list[str]:
    """What Spark saw that's worth a SAM pass of its own, beyond the painter, hands and brushes."""
    seen = {phrase.replace(" ", "") for phrase in existing}
    extra = []
    for occluder in scan.occluders:
        key = occluder.phrase.replace(" ", "")
        if occluder.kind not in TRACKED_ANYWAY and key not in seen:
            seen.add(key)
            extra.append(occluder.phrase)
    return extra[:limit]


def critic_answer(api: OpenAI, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Show Spark (camera | reconstruction) pairs; its raw answer, for caching. parse_critic reads it."""
    content: list[dict] = [{"type": "input_text", "text": CRITIC_PROMPT}]
    for number, (raw, clean) in enumerate(pairs):
        side_by_side = np.hstack([raw, clean])
        height = round(side_by_side.shape[0] * CRITIC_WIDTH / side_by_side.shape[1] / 2) * 2
        content.append({"type": "input_text", "text": f"Image {number}:"})
        content.append(image_block(imaging.resize(side_by_side, (CRITIC_WIDTH, height))))
    return ask_json(api, content, CRITIC_SCHEMA, "reconstruction_critic", effort="high")


def parse_critic(answer: dict, count: int, size: tuple[int, int]) -> list[Defect]:
    """Leftovers from a critic answer about `count` pairs, boxed in pixels of a `size` reconstruction."""
    return [
        Defect(item["image"], to_pixels(item["box"], size), item["kind"], float(item["confidence"]))
        for item in answer["defects"]
        if 0 <= item["image"] < count and len(item["box"]) >= 4
    ]


def chapters(api: OpenAI, scan_video: Path, picks: list[int]) -> list[dict]:
    answer = ask_json(
        api,
        [{"type": "input_text", "text": CHAPTERS_PROMPT}, {"type": "input_video", "file_id": upload(api, scan_video)}],
        CHAPTERS_SCHEMA,
        "session_chapters",
        effort="low",
    )
    return [
        {"title": item["title"], "start": _scan_frame(item["start_second"], picks),
         "end": _scan_frame(item["end_second"], picks)}
        for item in answer["chapters"]
    ]
