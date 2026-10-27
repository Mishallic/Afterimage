"""Binary mask helpers in plain numpy, so the core runs anywhere (including Windows N)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape[:2] == (height, width):
        return mask
    rows = np.minimum(((np.arange(height) + 0.5) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)
    return mask[rows[:, None], cols[None, :]]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by `radius` pixels in every direction (square window, via running sums)."""
    if radius <= 0:
        return mask.astype(bool)
    window = 2 * radius + 1
    counts = np.cumsum(np.pad(mask.astype(np.int32), ((radius + 1, radius), (0, 0))), axis=0)
    counts = counts[window:] - counts[:-window]
    counts = np.cumsum(np.pad(counts, ((0, 0), (radius + 1, radius))), axis=1)
    counts = counts[:, window:] - counts[:, :-window]
    return counts > 0


def soften(mask: np.ndarray, radius: int) -> np.ndarray:
    """Feather a mask's edges into a 0..1 float alpha (a square box average)."""
    if radius <= 0:
        return mask.astype(np.float32)
    window = 2 * radius + 1
    total = np.cumsum(np.pad(mask.astype(np.float32), ((radius + 1, radius), (0, 0)), mode="edge"), axis=0)
    rows = (total[window:] - total[:-window]) / window
    total = np.cumsum(np.pad(rows, ((0, 0), (radius + 1, radius)), mode="edge"), axis=1)
    return (total[:, window:] - total[:, :-window]) / window


@dataclass(frozen=True)
class Blob:
    """One object's mask in one frame, stored only within its bounding box.

    Hours of footage times several objects would not fit in memory as full-frame masks.
    """

    left: int
    top: int
    mask: np.ndarray

    @classmethod
    def of(cls, mask: np.ndarray) -> Blob | None:
        rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        if not len(rows):
            return None
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return cls(int(left), int(top), mask[top:bottom, left:right].copy())

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def overlap(self, other: Blob) -> int:
        x0, y0 = max(self.left, other.left), max(self.top, other.top)
        x1 = min(self.left + self.mask.shape[1], other.left + other.mask.shape[1])
        y1 = min(self.top + self.mask.shape[0], other.top + other.mask.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 0
        mine = self.mask[y0 - self.top:y1 - self.top, x0 - self.left:x1 - self.left]
        theirs = other.mask[y0 - other.top:y1 - other.top, x0 - other.left:x1 - other.left]
        return int((mine & theirs).sum())

    def iou(self, other: Blob) -> float:
        shared = self.overlap(other)
        union = self.area + other.area - shared
        return shared / union if union else 1.0

    def grown(self, radius: int) -> Blob:
        return Blob(self.left - radius, self.top - radius, dilate(np.pad(self.mask, radius), radius))

    def _clipped(self, shape: tuple[int, int]) -> tuple[slice, slice, np.ndarray] | None:
        height, width = shape
        x0, y0 = max(self.left, 0), max(self.top, 0)
        x1 = min(self.left + self.mask.shape[1], width)
        y1 = min(self.top + self.mask.shape[0], height)
        if x1 <= x0 or y1 <= y0:
            return None
        part = self.mask[y0 - self.top:y1 - self.top, x0 - self.left:x1 - self.left]
        return slice(y0, y1), slice(x0, x1), part

    def paste_into(self, target: np.ndarray) -> None:
        clipped = self._clipped(target.shape)
        if clipped is not None:
            rows, cols, part = clipped
            target[rows, cols] |= part

    def share_outside(self, region: np.ndarray) -> float:
        """Share of this blob that falls outside `region`, a full-frame bool mask."""
        area = self.area
        if area == 0:
            return 0.0
        clipped = self._clipped(region.shape)
        inside = 0 if clipped is None else int((clipped[2] & region[clipped[0], clipped[1]]).sum())
        return 1 - inside / area
