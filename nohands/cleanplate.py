"""Rebuild the canvas behind the painter, frame by frame.

Each frame updates a running "clean" canvas from the pixels nobody is covering.
New paint is committed only once it holds steady for a few visible frames, so
moving shadows, glare and motion blur never make it into the result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
    """Keep every commit, so a Timeline can later veto the ones that turn out to be transient."""


def visible_mask(occluder: np.ndarray | None, shape: tuple[int, int], dilate_px: int) -> np.ndarray:
    if occluder is None:
        return np.ones(shape, dtype=bool)
    return ~dilate(resize_nearest(occluder.astype(bool), shape), dilate_px)


def first_visible_canvas(
    frames: Iterable[np.ndarray], occluders: Iterable[np.ndarray | None], *, dilate_px: int
) -> np.ndarray:
    """Each pixel's earliest uncovered value, so a painter in the first frame doesn't linger.

    Pixels that are never uncovered fall back to the first frame.
    """
    canvas: np.ndarray | None = None
    seen: np.ndarray | None = None
    for frame, occluder in zip(frames, occluders):
        if canvas is None or seen is None:
            canvas = frame.copy()
            seen = np.zeros(frame.shape[:2], dtype=bool)
        fresh = visible_mask(occluder, frame.shape[:2], dilate_px) & ~seen
        canvas[fresh] = frame[fresh]
        seen |= fresh
        if seen.all():
            break
    if canvas is None:
        raise ValueError("no frames")
    return canvas


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
        self.last_commit = np.zeros(initial_canvas.shape[:2], dtype=bool)
        """Pixels that became new paint in the most recent update."""
        self.painted_at = np.full(initial_canvas.shape[:2], -1, dtype=np.int32)
        """Update number at which each pixel last became new paint; -1 if it never did."""
        self._updates = 0
        self._commits: list[tuple[np.ndarray, int, np.ndarray, np.ndarray]] = []

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
                self._updates,
                _to_uint8(self._clean.reshape(-1, 3)[pixels]),
                _to_uint8(self._candidate.reshape(-1, 3)[pixels]),
            ))
        self._clean[commit] = self._candidate[commit]
        self._streak[commit] = 0
        self.last_commit = commit
        self.painted_at[commit] = self._updates
        self._updates += 1
        return self.canvas

    def _match_exposure(self, current: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Undo phone auto-exposure and white-balance drift, using the uncovered canvas."""
        reference = self._clean[visible]
        sample = current[visible]
        lit = reference.min(axis=1) > 20
        if lit.sum() < 500:
            return current
        # The median ignores the small share of pixels that changed because of new paint.
        gain = np.clip(np.median(sample[lit] / reference[lit], axis=0), 0.6, 1.6)
        return current / gain
