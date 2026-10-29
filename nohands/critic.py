"""The critic loop: Spark looks at the result, and SAM fixes what it finds.

After the clean plate has run, Spark sees (camera | reconstruction) pairs at the moments most
likely to have gone wrong and boxes anything on the reconstruction that isn't paint. Objects
go back to SAM as box prompts and become occluders wherever they appear; shadows, glare and
smears are held, which drops the change that put them there. Every decision is cached and
recorded, so a verify run replays it without asking again.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import imaging, media
from .geometry import Warper, homography, order_corners
from .sam import MAX_OBJECTS_PER_CALL
from .spark import Defect
from .timeline import Timeline

MIN_CONFIDENCE = 0.5
STILL_IOU = 0.9
"""Consecutive occluder masks overlapping this much count as holding still."""
KEYFRAMES = 6
GAP_SECONDS = 30.0
"""Keyframes are at least this far apart, so one long rest doesn't take every slot."""


def pick_keyframes(
    occlusion: list[float], stillness: list[float], kept: Timeline, persist_frames: int, fps: float,
    *, count: int = KEYFRAMES,
) -> list[int]:
    """Where to look: the end, the biggest single change, and the ends of the longest rests.

    A leftover comes from something that held still over the canvas for at least the
    persistence, so the last frame of each such rest shows it next to the thing that caused it.
    """
    frames = len(occlusion)
    if frames == 0:
        return []
    still = np.array(stillness) >= STILL_IOU
    run = np.zeros(frames, dtype=np.int64)
    for index in range(frames):
        run[index] = run[index - 1] + 1 if still[index] and index else int(still[index])
    rests = [index for index in range(frames)
             if run[index] >= persist_frames and (index == frames - 1 or not still[index + 1])]
    rests.sort(key=lambda index: -run[index])
    picks = [frames - 1]
    if len(kept):
        picks.append(int(np.bincount(kept.frame, minlength=frames).argmax()))
    gap = round(GAP_SECONDS * fps)
    for index in rests:
        if len(picks) >= count:
            break
        if all(abs(index - pick) >= gap for pick in picks):
            picks.append(index)
    return sorted(set(picks))


def pairs(
    proxy: Path, keyframes: list[int], start: np.ndarray, kept: Timeline, frames: int,
    *, fps: float, corners: tuple[tuple[float, float], ...] | None, size: tuple[int, int],
    proxy_size: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(camera view, reconstruction) at each keyframe, both facing the canvas at `size`."""
    if corners is None:
        def view(frame: np.ndarray) -> np.ndarray:
            return imaging.resize(frame, size)
    else:
        warper = Warper.facing(np.asarray(corners, dtype=np.float64) * proxy_size, size, (proxy_size[1], proxy_size[0]))
        view = warper.image
    raws = [view(frame) for frame in media.read_indices(proxy, keyframes, fps=fps, size=proxy_size)]
    wanted, rebuilt = set(keyframes), {}
    for index, (canvas, _) in enumerate(kept.replay(start, frames)):
        if index in wanted:
            rebuilt[index] = canvas.copy()
            if len(rebuilt) == len(wanted):
                break
    return [(raw, rebuilt[index]) for raw, index in zip(raws, keyframes)]


def _canvas_to_proxy(
    corners: tuple[tuple[float, float], ...] | None, size: tuple[int, int], proxy_size: tuple[int, int]
) -> np.ndarray:
    width, height = size
    if corners is None:
        return np.diag([proxy_size[0] / width, proxy_size[1] / height, 1.0])
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    return homography(rect, order_corners(np.asarray(corners, dtype=np.float64) * proxy_size))


def proxy_boxes(
    defects: Iterable[Defect], keyframes: list[int], corners: tuple[tuple[float, float], ...] | None,
    size: tuple[int, int], proxy_size: tuple[int, int],
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Object defects as SAM box prompts: (session frame, box) in the proxy's camera-view pixels."""
    matrix = _canvas_to_proxy(corners, size, proxy_size)
    boxes = []
    for defect in defects:
        if not defect.is_object:
            continue
        left, top, right, bottom = defect.box
        points = np.array([[left, top, 1], [right, top, 1], [right, bottom, 1], [left, bottom, 1]], float) @ matrix.T
        points = points[:, :2] / points[:, 2:3]
        x0, y0 = (max(int(v), 0) for v in np.floor(points.min(axis=0)))
        x1, y1 = (min(int(v), limit) for v, limit in zip(np.ceil(points.max(axis=0)), proxy_size))
        if x1 > x0 and y1 > y0:
            boxes.append((keyframes[defect.keyframe], (x0, y0, x1, y1)))
    return boxes[:MAX_OBJECTS_PER_CALL]


def holds(kept: Timeline, defects: Iterable[Defect], keyframes: list[int], size: tuple[int, int]) -> np.ndarray:
    """Changes to drop for the non-object defects: in each box, the latest change at or before its keyframe."""
    drop = np.zeros(len(kept), dtype=bool)
    x, y = kept.pixel % size[0], kept.pixel // size[0]
    for defect in defects:
        if defect.is_object:
            continue
        left, top, right, bottom = defect.box
        inside = np.flatnonzero(
            (x >= left) & (x < right) & (y >= top) & (y < bottom) & (kept.frame <= keyframes[defect.keyframe])
        )
        if not len(inside):
            continue
        inside = inside[np.lexsort((kept.frame[inside], kept.pixel[inside]))]
        pixel = kept.pixel[inside]
        drop[inside[np.r_[pixel[1:] != pixel[:-1], True]]] = True
    return drop
