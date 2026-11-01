"""Revocable commits: offline, the future can veto the past.

A shadow, a glint or a resting hand that holds still long enough gets committed like paint.
But it leaves, and the pixel goes back to what it was. Seen over the whole session that's not
paint: both changes are dropped, and the timelapse never shows them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Timeline:
    """Every committed change: which pixel, from which update on, from what colour to what colour."""

    shape: tuple[int, int]
    pixel: np.ndarray
    frame: np.ndarray
    before: np.ndarray
    after: np.ndarray
    drift: np.ndarray | None = None
    """True for a change that records slow drift on a settled pixel rather than a stroke (see
    cleanplate.DRIFT_THRESHOLD). The replay applies it, so the colour on screen keeps up with
    what the camera shows; it doesn't date the pixel for the receipt, count as a layer
    Underneath or count as painting. None means every change is a stroke."""

    @classmethod
    def concat(cls, shape: tuple[int, int], chunks: list[tuple]) -> Timeline:
        """Join (pixels, update per pixel, before, after[, is drift]) chunks."""
        if not chunks:
            empty = np.zeros(0, dtype=np.int32)
            return cls(shape, empty, empty, np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.uint8))
        drift = np.concatenate([np.full(len(chunk[0]), len(chunk) > 4 and bool(chunk[4])) for chunk in chunks])
        return cls(
            shape,
            np.concatenate([chunk[0] for chunk in chunks]).astype(np.int32),
            np.concatenate([chunk[1] for chunk in chunks]).astype(np.int32),
            np.concatenate([chunk[2] for chunk in chunks]),
            np.concatenate([chunk[3] for chunk in chunks]),
            drift if drift.any() else None,
        )

    def __len__(self) -> int:
        return len(self.pixel)

    def strokes(self) -> Timeline:
        """The changes that are strokes, without the drift records."""
        if self.drift is None:
            return self
        return self.without(self.drift)

    def transients(self, *, threshold: float, window: int) -> np.ndarray:
        """Changes that a later change undid within `window` updates, plus the undoing change.

        Matched like brackets, scanning each pixel's history from the end: a change is undone
        when the next change still standing after it returns the pixel to how it was before.
        Going backwards pairs short detours first, so a shadow passing over freshly revealed
        canvas is dropped without also dropping the reveal.
        """
        order = np.lexsort((self.frame, self.pixel))
        pixel, frame = self.pixel[order], self.frame[order]
        before = self.before[order].astype(np.int16)
        after = self.after[order].astype(np.int16)
        starts = np.flatnonzero(np.r_[True, pixel[1:] != pixel[:-1]])
        ends = np.r_[starts[1:], len(pixel)]
        dropped = np.zeros(len(pixel), dtype=bool)
        for start, end in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
            standing: list[int] = []
            for change in range(end - 1, start - 1, -1):
                if standing:
                    later = standing[-1]
                    if (frame[later] - frame[change] <= window
                            and np.abs(after[later] - before[change]).max() <= threshold):
                        standing.pop()
                        dropped[change] = dropped[later] = True
                        continue
                standing.append(change)
        result = np.zeros(len(pixel), dtype=bool)
        result[order] = dropped
        return result

    def without(self, dropped: np.ndarray) -> Timeline:
        keep = ~dropped
        drift = None if self.drift is None else self.drift[keep]
        if drift is not None and not drift.any():
            drift = None
        return Timeline(self.shape, self.pixel[keep], self.frame[keep], self.before[keep], self.after[keep], drift)

    def painted_at(self) -> np.ndarray:
        """Update at which each pixel's last stroke began; -1 if it never had one. Drift doesn't count."""
        strokes = self.strokes()
        result = np.full(self.shape[0] * self.shape[1], -1, dtype=np.int32)
        np.maximum.at(result, strokes.pixel, strokes.frame)
        return result.reshape(self.shape)

    def depth(self) -> np.ndarray:
        """How many strokes each pixel went through."""
        counts = np.zeros(self.shape[0] * self.shape[1], dtype=np.int32)
        np.add.at(counts, self.strokes().pixel, 1)
        return counts.reshape(self.shape)

    def peel(self, initial: np.ndarray, final: np.ndarray, layers: int) -> Iterator[np.ndarray]:
        """The painting with its last k strokes undone, for k = 1..layers: what's underneath.

        Each step takes every pixel back to the colour it had just before its next stroke from
        the end (drift included), so one layer of strokes comes off everywhere at once; a pixel
        with k strokes or fewer is down to the bare canvas. Conservators find this under old
        paintings with X-rays and call it pentimento; here it's a by-product of the record. The
        array yielded is reused.
        """
        order = np.lexsort((self.frame, self.pixel))
        pixel, after = self.pixel[order], self.after[order]
        stroke = np.ones(len(pixel), dtype=bool) if self.drift is None else ~self.drift[order]
        first = np.r_[True, pixel[1:] != pixel[:-1]]
        starts = np.flatnonzero(first)
        group_end = np.repeat(np.r_[starts[1:], len(pixel)], np.diff(np.r_[starts, len(pixel)]))
        to_come = np.r_[np.cumsum(stroke[::-1])[::-1], 0]
        from_end = to_come[:-1] - to_come[group_end] - 1  # for a stroke: strokes after it on its pixel
        canvas = final.copy()
        flat, bare = canvas.reshape(-1, 3), initial.reshape(-1, 3)
        for k in range(1, layers + 1):
            peeled = np.flatnonzero(stroke & (from_end == k - 1))
            down = first[peeled]
            flat[pixel[peeled[down]]] = bare[pixel[peeled[down]]]
            flat[pixel[peeled[~down]]] = after[peeled[~down] - 1]
            yield canvas

    def replay(self, initial: np.ndarray, frames: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(canvas, pixels a stroke changed this update) for each update. Drift records are applied
        to the canvas but not marked as changed. The canvas array is reused."""
        canvas = initial.copy()
        flat = canvas.reshape(-1, 3)
        order = np.argsort(self.frame, kind="stable")
        pixel, frame, after = self.pixel[order], self.frame[order], self.after[order]
        stroke = np.ones(len(pixel), dtype=bool) if self.drift is None else ~self.drift[order]
        bounds = np.searchsorted(frame, np.arange(frames + 1))
        for update in range(frames):
            changed = np.zeros(self.shape[0] * self.shape[1], dtype=bool)
            low, high = bounds[update], bounds[update + 1]
            if high > low:
                flat[pixel[low:high]] = after[low:high]
                changed[pixel[low:high][stroke[low:high]]] = True
            yield canvas, changed.reshape(self.shape)
