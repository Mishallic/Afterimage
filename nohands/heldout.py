"""Score a session against frames it never saw: the moments the painter stepped back.

While nobody is over the canvas, the camera sees the painting as it really is. Hide those frames,
rebuild the canvas without them and compare with what the camera saw. The baselines get the same
frames hidden: a masked median per segment (Loomis Painter style, with the same masks) and a
plain temporal median. Separately, stretches where the painter is in view but not painting score
false commits: nothing committed in them is paint.
"""

from __future__ import annotations

from collections import deque
from contextlib import closing
from itertools import count
from pathlib import Path

import numpy as np

from . import media
from .cleanplate import exposure_gain
from .evaluate import colour_error, masked_median, psnr
from .pipeline import Geometry, Params, clean
from .timeline import Timeline

CLEARLY_DIFFERENT = 5.0
"""ΔE00 above which two colours look clearly different side by side."""
STILL_LEVELS = 6.0
"""Most pixels of a held-out window must vary less than this (0-255), or something moved in it."""
METHODS = ("NoHands", "masked median (Loomis-style)", "temporal median (no masks)")


def clear_windows(occlusion: list[float], *, min_frames: int, trim: int = 1) -> list[tuple[int, int]]:
    """Stretches of at least `min_frames` frames with nothing over the canvas, as [start, stop).

    `trim` frames come off each end, where the painter is still stepping out or back in.
    """
    windows, start = [], None
    for index, share in enumerate([*occlusion, 1.0]):
        if share == 0 and start is None:
            start = index
        elif share > 0 and start is not None:
            first, stop = start + trim, index - trim
            if stop - first >= min_frames:
                windows.append((first, stop))
            start = None
    return windows


def false_commit_rate(timeline: Timeline, windows: list[tuple[int, int]], fps: float) -> float:
    """Share of the canvas committed per minute during stretches with no painting, in percent."""
    minutes = sum(stop - start for start, stop in windows) / fps / 60
    if not minutes:
        return 0.0
    committed = sum(int(((timeline.frame >= start) & (timeline.frame < stop)).sum()) for start, stop in windows)
    return 100 * committed / (timeline.shape[0] * timeline.shape[1]) / minutes


def _matched(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`frame` brought to `reference`'s exposure, so only content differences are scored."""
    current = frame.astype(np.float32)
    gain = exposure_gain(reference.astype(np.float32), current, np.ones(frame.shape[:2], dtype=bool))
    return current if gain is None else current * gain


def _scores(output: np.ndarray, truth: np.ndarray) -> dict:
    matched = np.clip(_matched(output, truth) + 0.5, 0, 255).astype(np.uint8)
    error = colour_error(matched, truth)
    return {
        "delta_e": round(float(error.mean()), 2),
        "clearly_off_pct": round(100 * float((error > CLEARLY_DIFFERENT).mean()), 2),
        "psnr": round(psnr(matched, truth), 2),
    }


def heldout_report(
    video: Path, params: Params, masks: dict[int, np.ndarray] | None, *,
    windows: list[tuple[int, int]] | None = None, static: list[tuple[int, int]] = (),
    min_seconds: float = 3.0, segment_seconds: float = 10.0,
) -> dict:
    """Rebuild the session with its clear windows hidden, and score every method on them.

    `masks` are the camera-view occluder masks the run used. Windows and static stretches are
    [start, stop) frame ranges; without `windows`, they're found in the masks. A window needs a
    full segment of frames before it, or there is nothing to rebuild it from.
    """
    info = media.probe(video)
    geometry = Geometry.of(info, params)
    segment = max(1, round(segment_seconds * params.fps))
    if windows is None:
        if masks is None:
            raise ValueError("finding clear windows needs masks; give them by hand instead")
        seconds = min(info.duration, params.seconds) if params.seconds else info.duration
        occlusion = [
            0.0 if masks.get(index) is None else float(geometry.view_mask(masks[index]).mean())
            for index in range(int(seconds * params.fps))
        ]
        windows = clear_windows(occlusion, min_frames=max(1, round(min_seconds * params.fps)))
    windows = [(start, stop) for start, stop in windows if start >= segment]
    hidden = frozenset(index for start, stop in windows for index in range(start, stop))
    cleaned = clean(video, params, geometry, masks, raw_out=None, hidden=hidden)
    frame_count = len(cleaned.occlusion)
    windows = [(start, min(stop, frame_count)) for start, stop in windows if start < frame_count]
    starts = {start for start, _ in windows}
    within = {index: start for start, stop in windows for index in range(start, stop)}

    outputs: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHODS}
    for index, (canvas, _) in enumerate(cleaned.kept.replay(cleaned.start, len(cleaned.occlusion))):
        if index in starts:
            outputs["NoHands"][index] = canvas.copy()

    seen: dict[int, list[np.ndarray]] = {start: [] for start in starts}
    recent: deque[np.ndarray] = deque(maxlen=segment)
    everything = np.ones((geometry.size[1], geometry.size[0]), dtype=bool)
    stream = (
        geometry.view(frame)
        for frame in media.read_frames(video, fps=params.fps, size=geometry.frame_size, seconds=params.seconds)
    )

    def observed():
        for index, frame in enumerate(stream):
            if index in starts:
                outputs["temporal median (no masks)"][index] = np.median(np.stack(recent), axis=0).astype(np.uint8)
            if index in within:
                seen[within[index]].append(frame)
            else:
                recent.append(frame)
            yield frame

    view_masks = (
        everything if index in hidden else None if masks is None else geometry.view_mask(masks.get(index))
        for index in count()
    )
    with closing(stream):
        for index, output in enumerate(masked_median(observed(), view_masks, segment, params.dilate)):
            if index in starts:
                outputs["masked median (Loomis-style)"][index] = output

    report: dict = {"fps": params.fps, "frames": frame_count, "windows": [], "rejected": []}
    for start, stop in windows:
        frames = seen[start]
        matched = np.stack([_matched(frame, frames[0]) for frame in frames])
        spread = float(np.percentile(matched.std(axis=0).max(axis=2), 95))
        entry = {"start_s": round(start / params.fps, 1), "stop_s": round(stop / params.fps, 1),
                 "frames": len(frames), "spread": round(spread, 2)}
        if spread > STILL_LEVELS:
            report["rejected"].append(entry)
            continue
        truth = np.clip(np.median(matched, axis=0) + 0.5, 0, 255).astype(np.uint8)
        entry["scores"] = {method: _scores(outputs[method][start], truth) for method in METHODS}
        report["windows"].append(entry)

    scored = report["windows"]
    report["summary"] = {
        method: {
            key: round(float(np.mean([entry["scores"][method][key] for entry in scored])), 2)
            for key in ("delta_e", "clearly_off_pct", "psnr")
        }
        for method in METHODS
    } if scored else {}
    report["static"] = {
        "minutes": round(sum(stop - start for start, stop in static) / params.fps / 60, 2),
        "false_commit_pct_per_min": round(false_commit_rate(cleaned.kept, list(static), params.fps), 4),
    } if static else None
    return report


def markdown(report: dict, title: str) -> str:
    lines = [
        f"# Held-out evaluation: {title}",
        "",
        f"{len(report['windows'])} stretches where the painter stepped back, {report['frames']} frames at "
        f"{report['fps']:g} fps. Each stretch was hidden from every method, rebuilt from the rest of the session "
        "and compared with what the camera saw. Lower ΔE00 is better; higher PSNR is better. "
        f"\"Clearly off\" counts pixels with ΔE00 above {CLEARLY_DIFFERENT:g}.",
        "",
        "| Method | ΔE00 | Clearly off % | PSNR (dB) |",
        "|---|---|---|---|",
    ]
    for method, scores in report["summary"].items():
        lines.append(f"| {method} | {scores['delta_e']} | {scores['clearly_off_pct']} | {scores['psnr']} |")
    if report["windows"]:
        lines += ["", "| Stretch (s) | Frames | Spread | " + " | ".join(f"{m} ΔE00" for m in METHODS) + " |",
                  "|---|---|---|" + "---|" * len(METHODS)]
        for entry in report["windows"]:
            scores = " | ".join(str(entry["scores"][method]["delta_e"]) for method in METHODS)
            lines.append(f"| {entry['start_s']}–{entry['stop_s']} | {entry['frames']} | {entry['spread']} | {scores} |")
    for entry in report["rejected"]:
        lines += ["", f"Not scored: {entry['start_s']}–{entry['stop_s']} s moved too much to be a still "
                      f"canvas (spread {entry['spread']}), so something was likely there that the masks missed."]
    if report.get("static"):
        static = report["static"]
        lines += ["", f"False commits while not painting: {static['false_commit_pct_per_min']}% of the canvas "
                      f"per minute, over {static['minutes']} min."]
    return "\n".join(lines) + "\n"
