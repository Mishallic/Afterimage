"""Snap rough canvas corners onto the canvas's real edges.

Spark places the corners within a few percent of the frame. A few percent of the frame is the
easel clamp and a strip of wall across the top of every output, and the bottom of the painting
cut off. The canvas is the one bright, straight-edged thing in the shot, so each edge is found
as the strongest bright-inside, dark-outside step near the rough edge, and a robust line
through those steps ignores whatever crosses the edge: the clamp, a brush, the painter's hand.
"""

from __future__ import annotations

import numpy as np

from .geometry import order_corners

BAND = 0.12
"""How far either side of a rough edge to look, as a share of the frame's shorter side."""
SAMPLES = 48
"""Points along each edge at which the step is sought."""
WINDOW = 6
"""Pixels averaged on either side of a candidate step."""
MIN_STEP = 12.0
"""Smallest inside-minus-outside brightness step (0-255) that counts as an edge."""
MIN_KEPT = 0.4
"""Share of an edge's samples the fitted line must agree with."""
MAX_SHIFT = 1.5
"""A corner may move at most this many bands; further is not the same canvas."""


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z of the cross product of 2-D vectors (numpy 2 dropped it from np.cross)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def snap_corners(frame: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """`corners` (four (x, y) in pixels of `frame`, any order) moved onto the canvas edges.

    Returns them ordered top-left, top-right, bottom-right, bottom-left, or None when no canvas
    fits: an edge without a clear step, a corner that would move too far, or a quadrilateral
    that isn't convex.
    """
    # A white canvas is bright in every channel; walls, wood and paint are not.
    brightness = frame.min(axis=2).astype(np.float32)
    height, width = brightness.shape
    reach = int(round(BAND * min(width, height)))
    points = order_corners(corners)
    lines = []
    for index in range(4):
        line = _edge_line(brightness, points[index], points[(index + 1) % 4], reach)
        if line is None:
            return None
        lines.append(line)
    snapped = []
    for index in range(4):
        corner = _intersection(lines[index - 1], lines[index])
        if corner is None:
            return None
        snapped.append(corner)
    moved = np.array(snapped)
    if np.hypot(*(moved - points).T).max() > MAX_SHIFT * reach or not _convex(moved):
        return None
    return order_corners(moved)


def _edge_line(
    brightness: np.ndarray, start: np.ndarray, end: np.ndarray, reach: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """The line of the strongest step from bright (inside) to darker (outside) near the edge start->end."""
    height, width = brightness.shape
    direction = end - start
    length = float(np.hypot(*direction))
    if length < 4 * WINDOW:
        return None
    tangent = direction / length
    inward = np.array([-tangent[1], tangent[0]])  # right-hand normal: into a TL, TR, BR, BL quadrilateral
    along = np.linspace(0.1, 0.9, SAMPLES)
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    base = start[None, :] + along[:, None] * direction[None, :]
    coords = base[:, None, :] + offsets[None, :, None] * inward[None, None, :]
    xs = np.clip(np.floor(coords[..., 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.floor(coords[..., 1]).astype(np.int64), 0, height - 1)
    profile = brightness[ys, xs]
    sums = np.cumsum(np.pad(profile, ((0, 0), (1, 0))), axis=1)
    k = np.arange(WINDOW, len(offsets) - WINDOW + 1)
    inside = (sums[:, k + WINDOW] - sums[:, k]) / WINDOW
    outside = (sums[:, k] - sums[:, k - WINDOW]) / WINDOW
    step = inside - outside
    best = step.argmax(axis=1)
    strength = step[np.arange(SAMPLES), best]
    good = strength >= MIN_STEP
    if good.sum() < MIN_KEPT * SAMPLES:
        return None
    # The step sits between the samples at offsets[k - 1] and offsets[k].
    found = base[good] + (offsets[k[best[good]]] - 0.5)[:, None] * inward[None, :]
    return _robust_line(found, SAMPLES)


def _robust_line(points: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Total-least-squares line through `points`, refitted without the points far off it."""
    keep = np.ones(len(points), dtype=bool)
    centre = direction = None
    for _ in range(4):
        if keep.sum() < max(2, MIN_KEPT * total):
            return None
        chosen = points[keep]
        centre = chosen.mean(axis=0)
        _, _, axes = np.linalg.svd(chosen - centre, full_matrices=False)
        direction = axes[0]
        residual = np.abs(_cross(points - centre, direction))
        limit = max(2.0, 3.0 * float(np.median(residual[keep])))
        refined = residual <= limit
        if np.array_equal(refined, keep):
            break
        keep = refined
    return centre, direction


def _intersection(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (point_a, direction_a), (point_b, direction_b) = a, b
    denominator = float(_cross(direction_a, direction_b))
    if abs(denominator) < 1e-9:
        return None
    t = float(_cross(point_b - point_a, direction_b)) / denominator
    return point_a + t * direction_a


def _convex(points: np.ndarray) -> bool:
    edges = np.roll(points, -1, axis=0) - points
    turns = _cross(edges, np.roll(edges, -1, axis=0))
    return bool((turns > 0).all() or (turns < 0).all())
