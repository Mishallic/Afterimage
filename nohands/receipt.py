"""The pixel receipt: when each part of the painting was last painted, as a lookup grid and a map."""

from __future__ import annotations

import numpy as np

from .geometry import homography, order_corners

STOPS = np.array([[24, 32, 92], [116, 44, 140], [222, 82, 92], [250, 204, 84]], dtype=np.float32)
"""Early strokes are deep blue, the latest are gold."""
UNPAINTED = (46, 46, 52)
LEAD_SECONDS = 6.0
"""The paint lands while the hand is still there, so the replay starts this long before the change was seen."""
TAIL_SECONDS = 2.0


def time_map(painted_at: np.ndarray, frames: int) -> np.ndarray:
    """The painting coloured by when each pixel was last painted."""
    progress = np.clip(painted_at / max(frames - 1, 1), 0, 1) * (len(STOPS) - 1)
    low = np.clip(np.floor(progress).astype(int), 0, len(STOPS) - 2)
    fraction = (progress - low)[..., None]
    colours = STOPS[low] * (1 - fraction) + STOPS[low + 1] * fraction
    colours[painted_at < 0] = UNPAINTED
    return colours.astype(np.uint8)


def to_camera(corners: tuple[tuple[float, float], ...] | None) -> list[float]:
    """Row-major 3x3 homography from canvas fractions to camera-frame fractions.

    The page shows the whole camera frame when a spot is tapped (the hand is the point),
    so the tapped spot has to be put back where it sits in that frame.
    """
    if corners is None:
        return np.eye(3).ravel().tolist()
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    return homography(unit, order_corners(np.asarray(corners, dtype=np.float64))).ravel().tolist()


def receipt(
    painted_at: np.ndarray, *, sample_fps: float, corners: tuple[tuple[float, float], ...] | None = None,
    width: int = 160,
) -> dict:
    """A small grid of "painted at" frame numbers for the match page's tap-to-see-the-moment."""
    height = max(1, round(painted_at.shape[0] * width / painted_at.shape[1]))
    rows = ((np.arange(height) + 0.5) * painted_at.shape[0] / height).astype(int)
    cols = ((np.arange(width) + 0.5) * painted_at.shape[1] / width).astype(int)
    grid = painted_at[rows[:, None], cols[None, :]]
    return {
        "width": width,
        "height": height,
        "sample_fps": sample_fps,
        "lead_s": LEAD_SECONDS,
        "tail_s": TAIL_SECONDS,
        "to_camera": to_camera(corners),
        "grid": grid.ravel().tolist(),
    }
