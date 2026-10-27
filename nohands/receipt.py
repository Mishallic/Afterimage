"""Proof of Hand: when each part of the painting was painted, as a lookup grid and a map."""

from __future__ import annotations

import numpy as np

STOPS = np.array([[24, 32, 92], [116, 44, 140], [222, 82, 92], [250, 204, 84]], dtype=np.float32)
"""Early strokes are deep blue, the latest are gold."""
UNPAINTED = (46, 46, 52)


def time_map(painted_at: np.ndarray, frames: int) -> np.ndarray:
    """The painting coloured by when each pixel was last painted."""
    progress = np.clip(painted_at / max(frames - 1, 1), 0, 1) * (len(STOPS) - 1)
    low = np.clip(np.floor(progress).astype(int), 0, len(STOPS) - 2)
    fraction = (progress - low)[..., None]
    colours = STOPS[low] * (1 - fraction) + STOPS[low + 1] * fraction
    colours[painted_at < 0] = UNPAINTED
    return colours.astype(np.uint8)


def receipt(painted_at: np.ndarray, *, sample_fps: float, video_fps: float, width: int = 160) -> dict:
    """A small grid of "painted at" frame numbers for the match page's tap-to-see-the-moment."""
    height = max(1, round(painted_at.shape[0] * width / painted_at.shape[1]))
    rows = ((np.arange(height) + 0.5) * painted_at.shape[0] / height).astype(int)
    cols = ((np.arange(width) + 0.5) * painted_at.shape[1] / width).astype(int)
    grid = painted_at[rows[:, None], cols[None, :]]
    return {
        "width": width,
        "height": height,
        "sample_fps": sample_fps,
        "video_fps": video_fps,
        "grid": grid.ravel().tolist(),
    }
