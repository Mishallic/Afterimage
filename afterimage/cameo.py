"""Cameo mode: you, sitting next to your canvas, while it paints itself.

Film a short clip of yourself beside the easel (not in front of it), mark the canvas corners,
and the clean timelapse is projected onto that canvas: real pixels in a real room.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import media
from .geometry import Warper, homography, order_corners, polygon_mask
from .masks import soften
from .render import ease


def _looped(clip: Path, fps: float, size: tuple[int, int]) -> Iterator[np.ndarray]:
    while True:
        produced = False
        for frame in media.read_frames(clip, fps=fps, size=size):
            produced = True
            yield frame
        if not produced:
            raise RuntimeError(f"no frames in {clip}")


def _white_balance(clip_frame: np.ndarray, region: np.ndarray, canvas_frame: np.ndarray) -> np.ndarray:
    """Per-channel gain so the timelapse's paper white matches the canvas white in the clip."""
    def bright(pixels: np.ndarray) -> np.ndarray:
        luminance = pixels.mean(axis=1)
        return pixels[luminance >= np.percentile(luminance, 95)].mean(axis=0)

    gain = bright(clip_frame[region].astype(np.float32)) / np.maximum(
        bright(canvas_frame.reshape(-1, 3).astype(np.float32)), 1.0
    )
    return np.clip(gain, 0.5, 2.0)


def render_cameo(
    clip: Path,
    corners: np.ndarray,
    timelapse: Path,
    out: Path,
    *,
    seconds: float = 8.0,
    hold: float = 1.5,
    width: int = 1280,
    fps: float = 30.0,
) -> Path:
    """`corners` are the canvas corners in the clip, as fractions of its width and height."""
    size = media.output_size(media.probe(clip), width)
    info = media.probe(timelapse)
    canvas_size = (info.width, info.height)
    total = max(1, round(info.duration * info.fps))

    quad = order_corners(np.asarray(corners, dtype=np.float64) * size)
    rect = np.array([[0, 0], [canvas_size[0], 0], [canvas_size[0], canvas_size[1]], [0, canvas_size[1]]], float)
    project = Warper(homography(quad, rect), size, (canvas_size[1], canvas_size[0]))
    region = polygon_mask(quad, (size[1], size[0]))
    alpha = soften(region, 2)[..., None]

    count = max(1, round(seconds * fps))
    painting = max(1, count - round(hold * fps))
    indices = [round(ease(min(k / max(painting - 1, 1), 1.0), "inout") * (total - 1)) for k in range(count)]

    room = _looped(clip, fps, size)
    first_room = next(room)
    canvases = media.read_indices(timelapse, indices, fps=info.fps, size=canvas_size)
    gain = None
    with media.FrameWriter(out, size, fps) as writer:
        for k, canvas in enumerate(canvases):
            frame = first_room if k == 0 else next(room)
            if gain is None:
                gain = _white_balance(frame, region, canvas)
            projected = project.image(canvas).astype(np.float32) * gain
            writer.write(np.clip(frame * (1 - alpha) + projected * alpha, 0, 255))
    return out
