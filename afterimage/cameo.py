"""Cameo mode: you, sitting next to your canvas, while it paints itself.

Film a short clip of yourself beside the easel, mark the canvas corners, and the clean timelapse
is projected onto that canvas: real pixels in a real room, graded from a white canvas in that room's
light to the finished painting as it looks there. Where you pass in front of the canvas,
SAM 3.1's mask of you keeps you in front of the painting. At the end the projection can fade out
onto the real canvas in the clip, so the last thing on screen is the painting itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from . import media
from .geometry import Warper, homography, order_corners, polygon_mask
from .masks import dilate, resize_nearest, soften
from .render import ease

MATCH_CORRELATION = 0.5
"""How alike the clip's canvas and the finished timelapse frame must look (luminance correlation)
for the colours to be matched painting to painting. Below it, the clip shows a different canvas
(blank, or another painting) and the brightest whites are matched instead."""


def _looped(clip: Path, fps: float, size: tuple[int, int], start: float, length: float | None) -> Iterator[tuple[int, np.ndarray]]:
    """(frame index in the clip from `start`, frame), from `start` for `length` seconds, over and over."""
    while True:
        produced = False
        for index, frame in enumerate(media.read_frames(clip, fps=fps, size=size, start=start, seconds=length)):
            produced = True
            yield index, frame
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


def colour_curve(
    clip_frame: np.ndarray, region: np.ndarray, projected_final: np.ndarray, first_canvas: np.ndarray,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    """A per-channel tone curve (3 x 256 lookup) that puts the timelapse in the clip's light.

    When the clip shows the finished painting (the usual cameo: filmed after the session), the
    tones of the projected last frame are matched to the tones of the real canvas in the clip,
    channel by channel, quantile to quantile, inside the canvas and away from its edges and the
    person. A single gain can't do it: filmed by a window, the painting is far darker than under
    the session's lamp, and a gain that fits the paint turns the white canvas teal. The curve is
    one global grade for every frame; no pixel is moved or made. When the clip shows a different
    canvas, the whites are matched instead, as a straight gain.
    """
    inner = region & ~dilate(~region, 8)
    if hidden is not None:
        inner &= ~dilate(hidden, 6)
    clip_pixels = clip_frame[inner].astype(np.float32)
    projected = projected_final[inner].astype(np.float32)
    levels = np.arange(256, dtype=np.float32)
    if len(clip_pixels) >= 1000:
        a, b = clip_pixels.mean(axis=1), projected.mean(axis=1)
        if a.std() > 1 and b.std() > 1 and np.corrcoef(a, b)[0, 1] >= MATCH_CORRELATION:
            quantiles = np.linspace(0.5, 99.5, 100)
            curve = np.empty((3, 256), dtype=np.float32)
            for channel in range(3):
                source = np.percentile(projected[:, channel], quantiles)
                target = np.percentile(clip_pixels[:, channel], quantiles)
                source, keep = np.unique(source, return_index=True)
                target = np.maximum.accumulate(target[keep])
                top = target[-1] * min(255.0 / max(source[-1], 1.0), 1.5)
                curve[channel] = np.interp(levels, np.r_[0.0, source, 255.0], np.r_[0.0, target, top])
            return np.clip(np.round(curve), 0, 255).astype(np.uint8)
    gain = _white_balance(clip_frame, region, first_canvas)
    return np.clip(np.round(levels[None, :] * gain[:, None]), 0, 255).astype(np.uint8)


def room_white_curve(
    clip_frame: np.ndarray, region: np.ndarray, first_canvas: np.ndarray, hidden: np.ndarray | None = None,
) -> np.ndarray | None:
    """A straight gain that makes the bare canvas as white as the room right around it (the wall,
    the canvas's own edge), or None when nothing around the canvas is light enough to tell."""
    ring = dilate(region, 60) & ~dilate(region, 6)
    if hidden is not None:
        ring &= ~dilate(hidden, 6)
    pixels = clip_frame[ring].astype(np.float32)
    if len(pixels) < 1000:
        return None
    luminance = pixels.mean(axis=1)
    white = pixels[luminance >= np.percentile(luminance, 90)].mean(axis=0)
    if white.mean() < 120:
        return None
    bare = first_canvas.reshape(-1, 3).astype(np.float32)
    bare_luminance = bare.mean(axis=1)
    gain = np.clip(white / np.maximum(bare[bare_luminance >= np.percentile(bare_luminance, 50)].mean(axis=0), 1.0), 0.5, 2.0)
    levels = np.arange(256, dtype=np.float32)
    return np.clip(np.round(levels[None, :] * gain[:, None]), 0, 255).astype(np.uint8)


def graded(frame: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """`frame` through a 3 x 256 per-channel lookup."""
    return np.stack([curve[channel][frame[..., channel]] for channel in range(3)], axis=-1)


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
    start: float = 0.0,
    occluder: Callable[[int], np.ndarray | None] | None = None,
    reveal: bool = False,
) -> Path:
    """`corners` are the canvas corners in the clip, as fractions of its width and height.

    The clip is used from `start` for `seconds` (looped if shorter). `occluder(i)` gives the mask
    of whoever is in front of the canvas in the clip's i-th frame from `start` (any size, or None);
    those pixels stay the clip's. With `reveal`, the projection fades out over the `hold`, onto the
    real canvas in the clip.
    """
    size = media.output_size(media.probe(clip), width)
    info = media.probe(timelapse)
    canvas_size = (info.width, info.height)
    total = max(1, round(info.duration * info.fps))

    quad = order_corners(np.asarray(corners, dtype=np.float64) * size)
    rect = np.array([[0, 0], [canvas_size[0], 0], [canvas_size[0], canvas_size[1]], [0, canvas_size[1]]], float)
    project = Warper(homography(quad, rect), size, (canvas_size[1], canvas_size[0]))
    region = polygon_mask(quad, (size[1], size[0]))
    region_alpha = soften(region, 2)[..., None]

    count = max(1, round(seconds * fps))
    hold_frames = round(hold * fps)
    painting = max(1, count - hold_frames)
    indices = [round(ease(min(k / max(painting - 1, 1), 1.0), "inout") * (total - 1)) for k in range(count)]

    def hidden_at(index: int) -> np.ndarray | None:
        mask = occluder(index) if occluder is not None else None
        if mask is None or not mask.any():
            return None
        return resize_nearest(mask.astype(bool), (size[1], size[0]))

    room = _looped(clip, fps, size, start, seconds)
    first_index, first_room = next(room)
    final = next(media.read_indices(timelapse, [total - 1], fps=info.fps, size=canvas_size))
    first_canvas = next(media.read_indices(timelapse, [0], fps=info.fps, size=canvas_size))
    first_hidden = hidden_at(first_index)
    end_curve = colour_curve(first_room, region, project.image(final), first_canvas, first_hidden)
    start_curve = room_white_curve(first_room, region, first_canvas, first_hidden)
    if start_curve is None:
        start_curve = end_curve
    # The grade eases from "a white canvas in this room" to "this painting in this room" as the
    # paint goes on, so the blank canvas starts white and the last frame meets the real canvas.
    start_curve, end_curve = start_curve.astype(np.float32), end_curve.astype(np.float32)

    canvases = media.read_indices(timelapse, indices, fps=info.fps, size=canvas_size)
    with media.FrameWriter(out, size, fps) as writer:
        for k, canvas in enumerate(canvases):
            index, frame = (first_index, first_room) if k == 0 else next(room)
            alpha = region_alpha
            hidden = hidden_at(index)
            if hidden is not None:
                alpha = alpha * (1.0 - soften(hidden, 2)[..., None])
            if reveal and hold_frames and k >= painting:
                alpha = alpha * (1.0 - ease((k - painting + 1) / hold_frames, "inout"))
            share = indices[k] / max(total - 1, 1)
            curve = np.round(start_curve + (end_curve - start_curve) * share).astype(np.uint8)
            projected = project.image(graded(canvas, curve)).astype(np.float32)
            writer.write(np.clip(frame * (1 - alpha) + projected * alpha, 0, 255))
    return out


def mask_lookup(
    objects: dict, size: tuple[int, int], *, mask_fps: float, fps: float, start: float, min_share: float = 0.0,
) -> Callable[[int], np.ndarray | None]:
    """occluder(i) for render_cameo from SAM objects tracked over the whole clip at `mask_fps`.

    `objects` maps a frame index (from the clip's first frame) to {key: Blob} in a copy of the clip
    at `size` (width, height); frame i of the cameo (from `start`, at `fps`) takes the mask nearest
    in time. Objects smaller than `min_share` of the frame are left out: a "person" prompt also
    finds the figures in posters and wall stickers, which never stand in front of the canvas.
    """
    width, height = size
    smallest = min_share * width * height

    def occluder(index: int) -> np.ndarray | None:
        blobs = objects.get(round((start + index / fps) * mask_fps))
        if not blobs:
            return None
        mask = np.zeros((height, width), dtype=bool)
        for blob in blobs.values():
            if blob.area >= smallest:
                blob.paste_into(mask)
        return mask if mask.any() else None

    return occluder
