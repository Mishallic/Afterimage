"""Measure reconstructions against ground truth, next to the baselines they have to beat.

Metrics: CIEDE2000 colour difference (how different two colours look to a person), PSNR, and
how often the painter or a shadow survives into the output. Baselines: a plain temporal median
(no masks) and a masked median per fixed-length segment, as in Loomis Painter (2025).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from itertools import islice

import numpy as np

from .masks import dilate

_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    colour = rgb.astype(np.float64) / 255.0
    linear = np.where(colour <= 0.04045, colour / 12.92, ((colour + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T / _D65
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    delta_l = l2 - l1
    delta_c = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    delta_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(
        c1p * c2p == 0, h_sum,
        np.where(np.abs(h1p - h2p) <= 180, h_sum / 2, np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)),
    )
    t = (1 - 0.17 * np.cos(np.radians(h_bar - 30)) + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6)) - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = (-2 * np.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25 ** 7))
           * np.sin(np.radians(60 * np.exp(-(((h_bar - 275) / 25) ** 2)))))
    return np.sqrt(
        (delta_l / s_l) ** 2 + (delta_c / s_c) ** 2 + (delta_h / s_h) ** 2
        + r_t * (delta_c / s_c) * (delta_h / s_h)
    )


def colour_error(image: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return delta_e2000(srgb_to_lab(image), srgb_to_lab(truth))


def psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = np.mean((image.astype(np.float64) - truth.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def temporal_median(frames: Iterable[np.ndarray], window: int) -> Iterator[np.ndarray]:
    """The classic clean plate: the median of the last `window` frames, no masks."""
    recent: deque[np.ndarray] = deque(maxlen=window)
    for frame in frames:
        recent.append(frame)
        yield np.median(np.stack(recent), axis=0).astype(np.uint8)


def masked_median(
    frames: Iterable[np.ndarray], masks: Iterable[np.ndarray | None], segment: int, dilate_px: int = 0
) -> Iterator[np.ndarray]:
    """One clean frame per segment: the median of each pixel's uncovered samples in it.

    Pixels covered for the whole segment keep the previous segment's value. This is the
    Loomis Painter approach (10-second segments), given the same masks NoHands gets.
    Streams: only one segment of frames is held at a time.
    """
    previous: np.ndarray | None = None
    pairs = zip(frames, masks)
    while batch := list(islice(pairs, segment)):
        chunk = np.stack([frame for frame, _ in batch]).astype(np.float32)
        covered = np.stack([
            np.zeros(chunk.shape[1:3], dtype=bool) if mask is None else dilate(mask, dilate_px)
            for _, mask in batch
        ])
        samples = np.where(covered[..., None], np.nan, chunk)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                clean = np.nanmedian(samples, axis=0)
        unseen = np.isnan(clean[..., 0])
        fallback = previous if previous is not None else chunk[0]
        clean[unseen] = fallback[unseen]
        previous = clean
        for _ in range(len(chunk)):
            yield clean.astype(np.uint8)


ARTIFACT_DIFFERENCE = 25
"""A pixel this far (0-255, any channel) from every recent real state is an artifact."""
ARTIFACT_SHARE = 0.002
"""A frame with more than this share of artifact pixels counts as showing an artifact."""


def frame_report(
    outputs: list[np.ndarray], truths: list[np.ndarray], *, skin: tuple[int, int, int],
    painted: np.ndarray | None = None, lag: int = 20,
) -> dict:
    """Colour error against the true canvas, separating delay from invention.

    Showing paint late is a delay: the output matches what the canvas looked like a moment
    ago. An artifact matches NO real state from the last `lag` frames: a shadow, part of the
    painter, a smear. Reported separately so neither hides the other.
    """
    skin_colour = np.array(skin, dtype=np.int16)
    recent: deque[np.ndarray] = deque(maxlen=lag + 1)
    errors, artifact_frames, painter_frames, artifact_share = [], 0, 0, []
    for output, truth in zip(outputs, truths):
        errors.append(float(colour_error(output, truth).mean()))
        out16 = output.astype(np.int16)
        recent.append(truth.astype(np.int16))
        history = np.stack(recent)
        closest = np.abs(history - out16[None]).max(axis=3).min(axis=0)
        artifact = closest > ARTIFACT_DIFFERENCE
        artifact_share.append(float(artifact.mean()))
        if artifact.mean() > ARTIFACT_SHARE:
            artifact_frames += 1
        looks_like_skin = np.abs(out16 - skin_colour).max(axis=2) < 12
        if painted is not None:
            looks_like_skin &= ~painted
        if (looks_like_skin & artifact).mean() > ARTIFACT_SHARE:
            painter_frames += 1
    count = max(len(errors), 1)
    return {
        "mean_delta_e": round(float(np.mean(errors)), 2),
        "final_delta_e": round(errors[-1], 2) if errors else None,
        "artifact_frames_pct": round(100 * artifact_frames / count, 1),
        "artifact_pixels_pct": round(100 * float(np.mean(artifact_share)), 3),
        "painter_visible_frames_pct": round(100 * painter_frames / count, 1),
    }
