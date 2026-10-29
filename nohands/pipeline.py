"""The NoHands pipeline, and the receipt that makes its output checkable.

masks → clean plate (pass 1, recording every change) → transient veto → replay (pass 2) → outputs.
Every step is deterministic given the source footage, the cached SAM 3.1 mask streams and the
parameters, and none of them generates pixels. So anyone holding those three can re-derive every
output frame and compare it with the receipt: `scripts/nohands_verify.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, fields, replace
from itertools import islice
from pathlib import Path

import numpy as np

from . import ledger, media, sam, spark
from .autocut import auto_edl
from .cleanplate import CleanPlate, CleanPlateConfig, first_visible_canvas
from .config import find_tool
from .devmasks import MotionMasker
from .geometry import Warper, canvas_size, order_corners, polygon_mask
from .ledger import file_sha256
from .masks import resize_nearest
from .occluders import occluders, painted
from .receipt import receipt as pixel_receipt, time_map
from .render import Sources, render
from .report import write_report
from .stats import StatsTracker
from .timeline import Timeline

RECEIPT_VERSION = 1
START_SCAN_FRAMES = 120
LEDGER = "ledger.jsonl"
MOMENTS_WIDTH = 960
"""Width of the natural-speed copy the pixel receipt plays from."""
MAX_LAYERS = 48
DIVE_SECONDS = 4.0
UNDER_STILLS = 8


@dataclass(frozen=True)
class Params:
    fps: float = 1.0
    """Frames sampled per second of footage."""
    width: int = 1280
    sam_width: int = 640
    seconds: float | None = None
    corners: tuple[tuple[float, float], ...] | None = None
    """Canvas corners as fractions of the camera frame (top-left, top-right, bottom-right, bottom-left)."""
    aspect: float | None = None
    dilate: int = 12
    persist_seconds: float = 8.0
    """How long new paint must hold before it counts; longer than SAM's ~7.5 s object confirmation."""
    revert_window: float = 300.0
    """Seconds within which an undone change counts as transient (shadow, glare, resting hand)."""
    concepts: tuple[str, ...] = ("person", "hand", "paintbrush")
    masks: str = "sam"
    """sam (the product), motion (offline development stand-in) or none."""
    scan: bool = True
    """With SAM masks: Spark watches the session first, adds what else to track and finds the canvas."""
    output_fps: float = 30.0

    @property
    def persist_frames(self) -> int:
        return max(2, round(self.persist_seconds * self.fps))

    @property
    def revert_frames(self) -> int:
        return round(self.revert_window * self.fps)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict) -> Params:
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        if values.get("corners") is not None:
            values["corners"] = tuple(tuple(point) for point in values["corners"])
        if "concepts" in values:
            values["concepts"] = tuple(values["concepts"])
        return cls(**values)


@dataclass(frozen=True)
class Geometry:
    frame_size: tuple[int, int]
    """Camera frames at working size."""
    size: tuple[int, int]
    """Output size: the straightened canvas, or the camera frame if there are no corners."""
    warper: Warper | None

    @classmethod
    def of(cls, info: media.VideoInfo, params: Params) -> Geometry:
        frame_size = media.output_size(info, params.width)
        if not params.corners:
            return cls(frame_size, frame_size, None)
        corners = np.array(params.corners, dtype=np.float64) * frame_size
        size = canvas_size(corners, params.width, params.aspect)
        return cls(frame_size, size, Warper.facing(corners, size, (frame_size[1], frame_size[0])))

    def view(self, frame: np.ndarray) -> np.ndarray:
        return self.warper.image(frame) if self.warper else frame

    def view_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        mask = resize_nearest(mask, (self.frame_size[1], self.frame_size[0]))
        return self.warper.mask(mask) if self.warper else mask


@dataclass
class Cleaned:
    start: np.ndarray
    timeline: Timeline
    kept: Timeline
    vetoed: int
    occlusion: list[float]


def toolchain() -> dict:
    ffmpeg = subprocess.run([find_tool("ffmpeg"), "-version"], capture_output=True, text=True).stdout
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ffmpeg": ffmpeg.splitlines()[0] if ffmpeg else "unknown",
    }


def _sam_work(out: Path, params: Params, proxy_size: tuple[int, int]) -> Path:
    tag = f"{params.fps:g}fps-{proxy_size[0]}px" + (f"-{params.seconds:g}s" if params.seconds else "")
    return out / "sam" / tag


def _proxy(video: Path, params: Params, proxy_size: tuple[int, int], work: Path) -> tuple[Path, int]:
    """The low-fps copy that SAM and Spark see, made once, and its number of frames."""
    proxy = work / "proxy.mp4"
    if not proxy.exists():
        media.make_proxy(video, proxy, fps=params.fps, size=proxy_size, seconds=params.seconds)
    info = media.probe(proxy)
    return proxy, round(info.duration * info.fps)


def spark_scan(video: Path, params: Params, info: media.VideoInfo, out: Path) -> tuple[Params, dict]:
    """Spark watches a 48-second digest of the session: what else to track, and where the canvas is.

    One Spark call decides which extra SAM passes are worth paying for, instead of running SAM on
    every concept that might matter. The answer is cached, so re-runs never ask again, and the
    receipt records the parameters it led to, so verify doesn't need it.
    """
    proxy_size = media.output_size(info, params.sam_width)
    proxy, frames = _proxy(video, params, proxy_size, _sam_work(out, params, proxy_size))
    picks = spark.scan_picks(frames)
    answer_file = out / "spark" / "scan.json"
    if not answer_file.exists():
        answer_file.parent.mkdir(parents=True, exist_ok=True)
        digest = answer_file.with_name("scan.mp4")
        spark.make_scan_video(proxy, digest, fps=params.fps, size=proxy_size, frames=frames)
        answer = spark.scan_answer(spark.client(), digest)
        answer_file.write_text(json.dumps(answer, indent=2), encoding="utf-8")
    scan = spark.parse_scan(json.loads(answer_file.read_text(encoding="utf-8")), picks, proxy_size)
    added = spark.extra_concepts(scan, params.concepts)
    corners = params.corners
    if corners is None and scan.canvas_corners is not None:
        corners = tuple((round(float(x), 4), round(float(y), 4)) for x, y in order_corners(scan.canvas_corners))
    found = {
        "file": answer_file.relative_to(out).as_posix(),
        "sha256": file_sha256(answer_file),
        "seen": [{"phrase": o.phrase, "kind": o.kind} for o in scan.occluders],
        "added_concepts": added,
        "canvas_corners": corners if corners != params.corners else None,
    }
    return replace(params, concepts=params.concepts + tuple(added), corners=corners), found


async def _sam_objects(
    video: Path, params: Params, proxy_size: tuple[int, int], events: dict[str, Path]
) -> dict[int, dict]:
    if all(path.exists() for path in events.values()):
        # Everything is cached: re-parse without touching the API (this is how verify runs).
        results = {concept: await sam.parse_cached(path) for concept, path in events.items()}
    else:
        work = next(iter(events.values())).parent
        proxy, frames = _proxy(video, params, proxy_size, work)
        api = sam.client()
        file_id = await sam.upload_cached(api, proxy, work / "file_id.txt")
        results = {
            concept: await sam.segment_concept(api, file_id, concept, path, frames=frames)
            for concept, path in events.items()
        }
    objects: dict[int, dict] = {}
    for concept, result in results.items():
        for index, blobs in sam.frame_objects(result, prefix=f"{concept}:").items():
            objects.setdefault(index, {}).update(blobs)
    return objects


def sam_occluders(
    video: Path, params: Params, info: media.VideoInfo, out: Path
) -> tuple[dict[int, np.ndarray], list[dict], list[str]]:
    """Per-frame painter masks (camera view), the SAM streams they came from, and what was kept as painted."""
    proxy_size = media.output_size(info, params.sam_width)
    work = _sam_work(out, params, proxy_size)
    events = {concept: work / f"{sam.slug(concept)}.events.ndjson" for concept in params.concepts}
    objects = asyncio.run(_sam_objects(video, params, proxy_size, events))
    shape = (proxy_size[1], proxy_size[0])
    if params.corners:
        region = polygon_mask(np.array(params.corners, dtype=np.float64) * proxy_size, shape)
    else:
        region = np.ones(shape, dtype=bool)
    keys = painted(objects, region, params.fps)
    streams = [
        {"concept": concept, "file": path.relative_to(out).as_posix(), "sha256": file_sha256(path)}
        for concept, path in events.items()
    ]
    return occluders(objects, keys, shape), streams, sorted(keys)


def clean(
    video: Path, params: Params, geometry: Geometry, masks: dict[int, np.ndarray] | None,
    *, raw_out: Path | None, hidden: frozenset[int] = frozenset(),
) -> Cleaned:
    """Pass 1: run the clean plate forward recording every change, then veto the transient ones.

    Frames in `hidden` count as fully covered, so nothing in them is used: the evaluation holds
    them out to score against.
    """

    def frames():
        return (
            geometry.view(frame)
            for frame in media.read_frames(video, fps=params.fps, size=geometry.frame_size, seconds=params.seconds)
        )

    everything = np.ones((geometry.size[1], geometry.size[0]), dtype=bool)
    motion = MotionMasker() if params.masks == "motion" else None
    if masks is not None:
        with closing(frames()) as head:
            start = first_visible_canvas(
                islice(head, START_SCAN_FRAMES),
                (everything if index in hidden else geometry.view_mask(masks.get(index))
                 for index in range(START_SCAN_FRAMES)),
                dilate_px=params.dilate,
                steady_frames=params.persist_frames,
            )
    else:
        with closing(frames()) as head:
            start = next(head)

    plate = CleanPlate(start, CleanPlateConfig(dilate_px=params.dilate, persist_frames=params.persist_frames, record=True))
    occlusion: list[float] = []
    with ExitStack() as stack:
        raw = stack.enter_context(media.FrameWriter(raw_out, geometry.size, params.output_fps)) if raw_out else None
        for index, frame in enumerate(stack.enter_context(closing(frames()))):
            if masks is not None:
                occluder = geometry.view_mask(masks.get(index))
            elif motion is not None:
                occluder = motion.mask(frame)
            else:
                occluder = None
            if index in hidden:
                occluder = everything
            plate.update(frame, occluder)
            occlusion.append(0.0 if occluder is None else float(occluder.mean()))
            if raw is not None:
                raw.write(frame)

    timeline = plate.timeline()
    vetoed = timeline.transients(threshold=plate.config.change_threshold, window=params.revert_frames)
    return Cleaned(start, timeline, timeline.without(vetoed), int(vetoed.sum()), occlusion)


def replay(
    cleaned: Cleaned, params: Params, size: tuple[int, int], *,
    out: Path | None = None, tracker: StatsTracker | None = None,
) -> tuple[np.ndarray, str]:
    """Pass 2: replay only the lasting changes. Returns the final canvas and a SHA-256 over every frame."""
    chain = hashlib.sha256()
    canvas = cleaned.start
    with ExitStack() as stack:
        if out is not None:
            timelapse = stack.enter_context(media.FrameWriter(out / "timelapse.mp4", size, params.output_fps))
            compare = stack.enter_context(media.FrameWriter(out / "compare.mp4", (size[0] * 2, size[1]), params.output_fps))
            raws = stack.enter_context(closing(media.read_frames(out / "raw.mp4", fps=params.output_fps, size=size)))
        for index, (canvas, changed) in enumerate(cleaned.kept.replay(cleaned.start, len(cleaned.occlusion))):
            chain.update(canvas.tobytes())
            if tracker is not None:
                tracker.update(canvas, changed, cleaned.occlusion[index])
            if out is not None:
                timelapse.write(canvas)
                compare.write(np.hstack([next(raws), canvas]))
    return canvas.copy(), chain.hexdigest()


def underneath(cleaned: Cleaned, final: np.ndarray, out: Path, fps: float) -> list[dict]:
    """Peel the painting back to the bare canvas: underneath.mp4, plus a few stills for the page.

    The dive stops once fewer than 0.1% of the pixels have anything left to peel (or at
    MAX_LAYERS), so one spot reworked fifty times doesn't stretch it out.
    """
    depth = cleaned.kept.depth()
    counts = np.bincount(depth.ravel())
    deeper = depth.size - np.cumsum(counts)
    layers = min(MAX_LAYERS, max(1, int(np.flatnonzero(deeper < depth.size * 0.001)[0])))
    stills = set(np.linspace(1, layers, min(UNDER_STILLS, layers)).round().astype(int).tolist())
    hold = max(1, round(DIVE_SECONDS * fps / layers))
    files = []
    with media.FrameWriter(out / "underneath.mp4", (final.shape[1], final.shape[0]), fps) as writer:
        for _ in range(round(fps / 2)):
            writer.write(final)
        for k, canvas in enumerate(cleaned.kept.peel(cleaned.start, final, layers), start=1):
            for _ in range(hold if k < layers else round(fps)):
                writer.write(canvas)
            if k in stills:
                files.append({"k": k, "file": media.write_image(canvas, out / f"underneath-{k}.jpg").name})
    return files


def run(video: Path, out: Path, params: Params, *, title: str = "", reels: bool = True) -> dict:
    """Process one session into out/: videos, stats, reels, match page, receipt.json and the ledger of Meta calls."""
    out.mkdir(parents=True, exist_ok=True)
    info = media.probe(video)
    masks, streams, kept_as_painted, scan = None, [], [], None
    if params.masks == "sam":
        with ledger.recording(out / LEDGER):
            if params.scan:
                params, scan = spark_scan(video, params, info, out)
                print(f"  Spark saw: {', '.join(item['phrase'] for item in scan['seen']) or 'nothing'}; "
                      f"tracking {', '.join(params.concepts)}")
            masks, streams, kept_as_painted = sam_occluders(video, params, info, out)
        if kept_as_painted:
            print(f"  kept as painted, not removed: {', '.join(kept_as_painted)}")

    geometry = Geometry.of(info, params)
    cleaned = clean(video, params, geometry, masks, raw_out=out / "raw.mp4")
    print(f"{len(cleaned.timeline)} pixel changes, {cleaned.vetoed} vetoed as transient")
    moments = out / "moments.mp4"
    if not moments.exists():
        media.make_moments(video, moments, size=media.output_size(info, MOMENTS_WIDTH), seconds=params.seconds)
    tracker = StatsTracker(cleaned.start, params.fps)
    final, chain = replay(cleaned, params, geometry.size, out=out, tracker=tracker)

    stats = tracker.finish(final).to_dict()
    stats["vetoed_changes"] = cleaned.vetoed
    stats["kept_changes"] = len(cleaned.kept)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if reels:
        sources = Sources(clean=out / "timelapse.mp4", raw=out / "raw.mp4", size=geometry.size, fps=params.output_fps)
        for layout, name in (("vertical", "reel.mp4"), ("landscape", "reel-landscape.mp4")):
            edl = auto_edl(stats, layout=layout)
            edl.validate(stats["frames"])
            if layout == "vertical":
                (out / "reel.edl.json").write_text(edl.to_json(), encoding="utf-8")
            render(edl, sources, out / name)

    receipt = {
        "version": RECEIPT_VERSION,
        "tool": "nohands",
        "source": {
            "file": video.name,
            "sha256": file_sha256(video),
            "bytes": video.stat().st_size,
            "duration_s": info.duration,
        },
        "params": params.to_dict(),
        "spark_scan": scan,
        "sam_streams": streams,
        "meta_calls": ledger.summary(out / LEDGER),
        "kept_as_painted": kept_as_painted,
        "timeline": {"changes": len(cleaned.timeline), "vetoed": cleaned.vetoed, "kept": len(cleaned.kept)},
        "output": {
            "frames": len(cleaned.occlusion),
            "size": list(geometry.size),
            "canvas_sha256": chain,
            "final_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
        },
        "generative_steps": [],
        "toolchain": toolchain(),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    painted_at = cleaned.kept.painted_at()
    final_image = media.write_image(final, out / "final.jpg")
    timemap = media.write_image(time_map(painted_at, stats["frames"]), out / "timemap.jpg")
    lookup = pixel_receipt(painted_at, sample_fps=params.fps, corners=params.corners)
    stills = underneath(cleaned, final, out, params.output_fps)
    write_report(out, stats, title or video.stem, final_image, timemap, lookup, receipt, stills)
    return receipt


@dataclass(frozen=True)
class Verdict:
    ok: bool
    checks: list[tuple[str, bool, str]]


def verify(receipt_path: Path, video: Path) -> Verdict:
    """Re-derive every output frame from the footage and the cached masks; compare with the receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    folder = receipt_path.parent
    checks: list[tuple[str, bool, str]] = []

    source_ok = file_sha256(video) == receipt["source"]["sha256"]
    checks.append(("source footage matches the receipt", source_ok, receipt["source"]["sha256"][:16]))
    streams_ok = all(
        (folder / stream["file"]).exists() and file_sha256(folder / stream["file"]) == stream["sha256"]
        for stream in receipt["sam_streams"]
    )
    checks.append(("SAM 3.1 mask streams match the receipt", streams_ok, f"{len(receipt['sam_streams'])} streams"))
    if not (source_ok and streams_ok):
        return Verdict(False, checks)

    params = Params.from_dict(receipt["params"])
    info = media.probe(video)
    geometry = Geometry.of(info, params)
    masks = sam_occluders(video, params, info, folder)[0] if params.masks == "sam" else None
    cleaned = clean(video, params, geometry, masks, raw_out=None)
    _, chain = replay(cleaned, params, geometry.size)
    same = chain == receipt["output"]["canvas_sha256"]
    detail = f"{len(cleaned.occlusion)} frames"
    if not same and toolchain() != receipt.get("toolchain"):
        detail += " (toolchain differs from the receipt's, so bit-identical output isn't expected)"
    checks.append(("every output frame re-derived from the footage", same, detail))
    return Verdict(all(ok for _, ok, _ in checks), checks)
