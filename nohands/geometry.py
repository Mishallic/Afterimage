"""Canvas geometry: find the canvas corners and straighten the view to face it head-on."""

from __future__ import annotations

import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    """Sort four (x, y) points into top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    total = points.sum(axis=1)
    spread = points[:, 0] - points[:, 1]
    return np.array([
        points[np.argmin(total)], points[np.argmax(spread)],
        points[np.argmax(total)], points[np.argmin(spread)],
    ])


def corners_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Corners of a roughly rectangular mask (such as SAM's "canvas"), taken as extreme points."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return None
    total, spread = xs + ys, xs - ys
    picks = [np.argmin(total), np.argmax(spread), np.argmax(total), np.argmin(spread)]
    return np.array([[xs[i] + 0.5, ys[i] + 0.5] for i in picks], dtype=np.float64)


def polygon_mask(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels inside the quadrilateral `corners` (any order), as a bool mask of `shape`."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    xs, ys = xs + 0.5, ys + 0.5
    points = order_corners(corners)
    inside = np.ones(shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(points, np.roll(points, -1, axis=0)):
        inside &= (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0) >= 0
    return inside


def homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3x3 matrix mapping four source points onto four target points."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(np.asarray(source, float), np.asarray(target, float)):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    solution = np.linalg.solve(np.array(rows), np.array(rhs))
    return np.append(solution, 1.0).reshape(3, 3)


def canvas_size(corners: np.ndarray, width: int, aspect: float | None = None) -> tuple[int, int]:
    """Even output size facing the canvas: `width` wide, height from `aspect` (height / width).

    Without an aspect it's estimated from the corners, which perspective skews a little,
    so pass the real canvas proportions when they're known (40x60 cm portrait is 1.5).
    """
    if aspect is None:
        top_left, top_right, bottom_right, bottom_left = corners
        across = (np.linalg.norm(top_right - top_left) + np.linalg.norm(bottom_right - bottom_left)) / 2
        down = (np.linalg.norm(bottom_left - top_left) + np.linalg.norm(bottom_right - top_right)) / 2
        aspect = down / across
    width = width // 2 * 2
    return width, max(2, round(width * aspect / 2) * 2)


class Warper:
    """Perspective resampling with the mapping precomputed, so every frame costs a few gathers."""

    def __init__(
        self, to_source: np.ndarray, size: tuple[int, int], source_shape: tuple[int, int]
    ) -> None:
        width, height = size
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        w = to_source[2, 0] * u + to_source[2, 1] * v + to_source[2, 2]
        x = (to_source[0, 0] * u + to_source[0, 1] * v + to_source[0, 2]) / w - 0.5
        y = (to_source[1, 0] * u + to_source[1, 1] * v + to_source[1, 2]) / w - 0.5
        source_height, source_width = source_shape
        self._outside = (x < -0.5) | (x > source_width - 0.5) | (y < -0.5) | (y > source_height - 0.5)
        x0, y0 = np.floor(x), np.floor(y)
        self._fx = (x - x0).astype(np.float32)[..., None]
        self._fy = (y - y0).astype(np.float32)[..., None]
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        self._x0 = np.clip(x0, 0, source_width - 1)
        self._x1 = np.clip(x0 + 1, 0, source_width - 1)
        self._y0 = np.clip(y0, 0, source_height - 1)
        self._y1 = np.clip(y0 + 1, 0, source_height - 1)
        self._xn = np.clip(np.rint(x), 0, source_width - 1).astype(np.int64)
        self._yn = np.clip(np.rint(y), 0, source_height - 1).astype(np.int64)
        self.size = size

    @classmethod
    def facing(
        cls, corners: np.ndarray, size: tuple[int, int], source_shape: tuple[int, int]
    ) -> Warper:
        """Straighten the quadrilateral `corners` (TL, TR, BR, BL in source pixels) to `size`."""
        width, height = size
        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        return cls(homography(target, order_corners(corners)), size, source_shape)

    def image(self, frame: np.ndarray) -> np.ndarray:
        """Resampled frame; anything that maps outside the source comes out black."""
        source = frame.astype(np.float32)
        top = source[self._y0, self._x0] * (1 - self._fx) + source[self._y0, self._x1] * self._fx
        bottom = source[self._y1, self._x0] * (1 - self._fx) + source[self._y1, self._x1] * self._fx
        out = np.clip(top * (1 - self._fy) + bottom * self._fy + 0.5, 0, 255).astype(np.uint8)
        out[self._outside] = 0
        return out

    def mask(self, mask: np.ndarray) -> np.ndarray:
        return mask[self._yn, self._xn]
