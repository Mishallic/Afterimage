"""Rebuild the canvas behind the painter, frame by frame.

Each frame updates a running "clean" canvas from the pixels nobody is covering.
New paint is committed only once it holds steady for a few visible frames, so
moving shadows, glare and motion blur never make it into the result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import numpy as np

from .masks import dilate, resize_nearest
from .timeline import Timeline


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(values + 0.5, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class CleanPlateConfig:
    dilate_px: int = 12
    """Grow occluder masks to cover soft edges, motion blur and the brush tip."""
    change_threshold: float = 20.0
    """Per-channel difference (0-255) that counts as a real change on the canvas."""
    persist_frames: int = 3
    """Visible frames a change must hold before it's committed as paint."""
    drift_rate: float = 0.05
    """How quickly unchanged pixels follow slow lighting drift."""
    normalize_exposure: bool = True
    record: bool = False
    """Keep every commit, so a Timeline can later veto the ones that turn out to be transient.

    Commits are dated to the update their value was first seen, not the one that confirmed it,
    so replaying the Timeline shows paint as soon as it appeared, without the persistence delay.
    """


def visible_mask(occluder: np.ndarray | None, shape: tuple[int, int], dilate_px: int) -> np.ndarray:
    if occluder is None:
        return np.ones(shape, dtype=bool)
    return ~dilate(resize_nearest(occluder.astype(bool), shape), dilate_px)


def exposure_gain(current: np.ndarray, reference: np.ndarray, usable: np.ndarray) -> np.ndarray | None:
    """Per-channel gain from `reference` to `current`: phone auto-exposure and white-balance drift.

    Measured on the `usable` pixels; None when too few of them are well lit to tell.
    """
    sample, base = current[usable], reference[usable]
    lit = base.min(axis=1) > 20
    if lit.sum() < 500:
        return None
    # The median ignores the small share of pixels that changed because of new paint.
    return np.clip(np.median(sample[lit] / base[lit], axis=0), 0.6, 1.6)


def first_visible_canvas(
    frames: Iterable[np.ndarray], occluders: Iterable[np.ndarray | None], *, dilate_px: int,
    steady_frames: int = 3, change_threshold: float = 20.0,
) -> np.ndarray:
    """The canvas before anything happened: what each pixel showed once the painter was off it.

    The masks keep the painter out but miss their shadow. A pixel's first uncovered value is
    settled once it holds for `steady_frames` uncovered frames. If it changes sooner, it gives
    way to the next value only if it could have been a shadow over it, i.e. was nowhere
    brighter. So the hand's shadow in the first frame is skipped, while paint added in the first
    moments still arrives as new paint. Frames are matched to the first one's exposure; pixels
    never uncovered fall back to the first frame.
    """
    pairs = zip(frames, occluders)
    try:
        frame, occluder = next(pairs)
    except StopIteration:
        raise ValueError("no frames") from None
    shape = frame.shape[:2]
    anchor = frame.astype(np.float32)
    anchor_visible = visible_mask(occluder, shape, dilate_px)
    start, candidate = anchor.copy(), anchor.copy()
    streak = np.zeros(shape, dtype=np.int32)
    following = np.ones(shape, dtype=bool)
    """Start still follows the latest run of values: everything before it was shadow-like."""
    done = np.zeros(shape, dtype=bool)
    for frame, occluder in chain([(frame, occluder)], pairs):
        current = frame.astype(np.float32)
        visible = visible_mask(occluder, shape, dilate_px)
        gain = exposure_gain(current, anchor, visible & anchor_visible)
        if gain is not None:
            current /= gain

        looking = visible & ~done
        steady = np.abs(current - candidate).max(axis=2) <= change_threshold
        holding = looking & steady & (streak > 0)
        fresh = looking & ~holding
        ended = fresh & (streak > 0) & following
        following[ended] = (start[ended] <= current[ended] + change_threshold / 4).all(axis=1)
        streak[holding] += 1
        streak[fresh] = 1
        candidate[fresh] = current[fresh]
        candidate[holding] += (current[holding] - candidate[holding]) / streak[holding][:, None]
        tracking = looking & following
        start[tracking] = candidate[tracking]

        done |= looking & (streak >= steady_frames)
        if done.all():
            break
    return _to_uint8(start)


class CleanPlate:
    """Feed frames in order; each update returns the canvas with occluders removed.

    Start it from first_visible_canvas() when the masks are known up front, or from the
    first frame when they aren't.
    """

    def __init__(self, initial_canvas: np.ndarray, config: CleanPlateConfig | None = None) -> None:
        self.config = config or CleanPlateConfig()
        self._clean = initial_canvas.astype(np.float32)
        self._candidate = self._clean.copy()
        self._streak = np.zeros(initial_canvas.shape[:2], dtype=np.int32)
        self._since = np.zeros(initial_canvas.shape[:2], dtype=np.int32)
        """Update at which each pixel's current run of a new value started."""
        self.last_commit = np.zeros(initial_canvas.shape[:2], dtype=bool)
        """Pixels that became new paint in the most recent update."""
        self.painted_at = np.full(initial_canvas.shape[:2], -1, dtype=np.int32)
        """Update at which each pixel's latest paint was first seen; -1 if it never changed."""
        self._updates = 0
        self._commits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def timeline(self) -> Timeline:
        return Timeline.concat(self.painted_at.shape, self._commits)

    @property
    def canvas(self) -> np.ndarray:
        return np.clip(self._clean, 0, 255).astype(np.uint8)

    def update(self, frame: np.ndarray, occluder: np.ndarray | None = None) -> np.ndarray:
        cfg = self.config
        current = frame.astype(np.float32)
        visible = visible_mask(occluder, frame.shape[:2], cfg.dilate_px)
        if cfg.normalize_exposure:
            current = self._match_exposure(current, visible)

        differs = np.abs(current - self._clean).max(axis=2) > cfg.change_threshold
        steady = np.abs(current - self._candidate).max(axis=2) <= cfg.change_threshold
        changing = visible & differs

        # Count how long each changed pixel has shown the same new value. Covered
        # pixels keep their streak: we just didn't get to see them this frame.
        continuing = changing & steady & (self._streak > 0)
        restarting = changing & ~continuing
        self._streak[continuing] += 1
        self._streak[restarting] = 1
        self._since[restarting] = self._updates
        self._candidate[restarting] = current[restarting]
        self._candidate[continuing] = 0.5 * (self._candidate[continuing] + current[continuing])

        settled = visible & ~differs
        self._streak[settled] = 0
        self._clean[settled] += cfg.drift_rate * (current[settled] - self._clean[settled])

        commit = self._streak >= cfg.persist_frames
        if cfg.record and commit.any():
            pixels = np.flatnonzero(commit)
            self._commits.append((
                pixels.astype(np.int32),
                self._since.reshape(-1)[pixels],
                _to_uint8(self._clean.reshape(-1, 3)[pixels]),
                _to_uint8(self._candidate.reshape(-1, 3)[pixels]),
            ))
        self._clean[commit] = self._candidate[commit]
        self._streak[commit] = 0
        self.last_commit = commit
        self.painted_at[commit] = self._since[commit]
        self._updates += 1
        return self.canvas

    def _match_exposure(self, current: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Undo phone auto-exposure and white-balance drift, using the uncovered canvas."""
        gain = exposure_gain(current, self._clean, visible)
        return current if gain is None else current / gain
