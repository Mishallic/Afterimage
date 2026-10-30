"""A fake painting session with known ground truth.

A hand and arm sweep over the canvas and leave strokes underneath themselves. They cast a
shadow that is NOT in the mask, and the "phone" jitters its exposure every frame. The
painter steps out of frame at the end, the way we ask people to film. Colours are RGB.
"""

from __future__ import annotations

import numpy as np

from nohands.masks import Blob

SKIN = (200, 160, 130)
WALL = (90, 92, 104)
PALETTE = [(30, 90, 160), (40, 120, 40), (150, 30, 30), (20, 20, 20), (60, 140, 180), (200, 160, 60)]


def synthetic_session(
    frames: int = 90, *, size: tuple[int, int] = (320, 240), seed: int = 7
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns (frame, occluder mask, ground-truth canvas) per frame, RGB uint8."""
    rng = np.random.default_rng(seed)
    width, height = size
    radius = max(12, height // 10)
    yy, xx = np.mgrid[0:height, 0:width]
    truth = np.full((height, width, 3), 228.0, np.float32)
    session = []
    for t in range(frames):
        stepped_out = t >= frames - 8
        hx = width / 2 + width * 0.32 * np.sin(t * 0.37)
        hy = height / 2 + height * 0.3 * np.sin(t * 0.23 + 1.0)
        if not stepped_out and t % 3 == 0:
            half_w, half_h = int(radius * 0.6), int(radius * 0.3)
            x0, y0 = int(hx) - half_w, int(hy) - half_h
            stroke = PALETTE[rng.integers(len(PALETTE))]
            truth[max(y0, 0):y0 + 2 * half_h, max(x0, 0):x0 + 2 * half_w] = stroke

        frame = truth.copy()
        occluder = np.zeros((height, width), dtype=bool)
        if not stepped_out:
            hand = (xx - hx) ** 2 + (yy - hy) ** 2 < radius ** 2
            arm = (np.abs((yy - hy) - 0.35 * (xx - hx)) < radius * 0.45) & (xx > hx)
            occluder = hand | arm
            offset = int(radius * 0.75)
            shadow = np.roll(occluder, (offset, offset), axis=(0, 1)) & ~occluder
            frame[shadow] *= 0.7
            frame[occluder] = SKIN

        gain = 1.0 if t == 0 else rng.uniform(0.93, 1.07)
        frame = np.clip(frame * gain, 0, 255).astype(np.uint8)
        session.append((frame, occluder, truth.astype(np.uint8)))
    return session


def resting_hand_session(
    frames: int = 80, *, size: tuple[int, int] = (320, 240), rest: tuple[int, int] = (20, 50)
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """The painter rests their hand for a while; its shadow (not in the mask) holds still too.

    Returns (frame, occluder mask, ground-truth canvas, shadow area) per frame.
    """
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    truth = np.full((height, width, 3), 228.0, np.float32)
    truth[40:70, 40:120] = PALETTE[0]
    truth[150:180, 180:280] = PALETTE[2]
    session = []
    for t in range(frames):
        resting = rest[0] <= t < rest[1]
        hx = 150.0 if resting else width / 2 + width * 0.3 * np.sin(t * 0.37)
        hy = 110.0 if resting else height / 2 + height * 0.28 * np.sin(t * 0.23 + 1.0)
        frame = truth.copy()
        hand = (xx - hx) ** 2 + (yy - hy) ** 2 < 22 ** 2
        arm = (np.abs((yy - hy) - 0.35 * (xx - hx)) < 10) & (xx > hx)
        occluder = hand | arm
        shadow = np.roll(occluder, (20, 20), axis=(0, 1)) & ~occluder
        frame[shadow] *= 0.6
        frame[occluder] = SKIN
        session.append((frame.astype(np.uint8), occluder, truth.astype(np.uint8), shadow))
    return session


def shadow_session(
    frames: int = 90, *, size: tuple[int, int] = (320, 240), shadow: tuple[int, int] = (10, 50),
    share: float = 0.6, darkness: float = 0.6, seed: int = 3,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """The painter's shadow (not in the mask) lies over most of the canvas for a while.

    The hand rests at the right edge throughout and the "phone" jitters its exposure every
    frame, so the exposure has to be measured on a canvas that is mostly in shadow. Returns
    (frame, occluder mask, ground-truth canvas, shadow area) per frame.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    truth = np.full((height, width, 3), 228.0, np.float32)
    truth[40:70, 40:120] = PALETTE[0]
    truth[150:180, 180:280] = PALETTE[2]
    hand = (xx - width + 10) ** 2 + (yy - height / 2) ** 2 < 20 ** 2
    session = []
    for t in range(frames):
        frame = truth.copy()
        shadowed = np.zeros((height, width), dtype=bool)
        if shadow[0] <= t < shadow[1]:
            shadowed[:, :int(width * share)] = True
            frame[shadowed] *= darkness
        frame[hand] = SKIN
        frame = np.clip(frame * rng.uniform(0.93, 1.07), 0, 255).astype(np.uint8)
        session.append((frame, hand, truth.astype(np.uint8), shadowed))
    return session


def painted_hand_session(
    frames: int = 90, *, size: tuple[int, int] = (320, 240)
) -> tuple[list[tuple[np.ndarray, dict[str, Blob], np.ndarray]], np.ndarray]:
    """The hard case: someone paints a skin-coloured hand while their own hand moves about.

    Returns ([(frame, SAM-like objects, ground-truth canvas)], canvas region). Like SAM, the
    objects include both hands: "hand:1" is the real hand and arm, reaching in past the canvas
    edge; "hand:2" is the painted hand, as far as it's painted and not covered.
    """
    width, height = size
    margin = 24
    canvas = np.zeros((height, width), dtype=bool)
    canvas[margin:height - margin, margin:width - margin] = True
    yy, xx = np.mgrid[0:height, 0:width]
    truth = np.full((height, width, 3), 228.0, np.float32)
    truth[~canvas] = WALL
    palm = (xx - 110) ** 2 + (yy - 120) ** 2 < 26 ** 2
    finger = (np.abs(xx - 104) < 7) & (yy > 66) & (yy < 120)
    painted_hand = palm | finger
    radius = 22

    session = []
    for t in range(frames):
        stepped_out = t >= frames - 8
        if 8 <= t <= 38:
            truth[painted_hand & (xx <= 80 + (t - 8) * 2)] = SKIN
        if t > 40 and t % 4 == 0 and not stepped_out:
            truth[150:164, 170 + (t - 40) * 2:186 + (t - 40) * 2] = PALETTE[t % len(PALETTE)]

        hx = width / 2 + width * 0.3 * np.sin(t * 0.37)
        hy = height / 2 + height * 0.28 * np.sin(t * 0.23 + 1.0)
        frame = truth.copy()
        objects: dict[str, Blob] = {}
        real = np.zeros((height, width), dtype=bool)
        if not stepped_out:
            hand = (xx - hx) ** 2 + (yy - hy) ** 2 < radius ** 2
            arm = (np.abs((yy - hy) - 0.35 * (xx - hx)) < radius * 0.45) & (xx > hx)
            real = hand | arm
            frame[np.roll(real, (16, 16), axis=(0, 1)) & ~real] *= 0.7
            frame[real] = SKIN
            objects["hand:1"] = Blob.of(real)
        visible_painting = painted_hand & (truth == SKIN).all(axis=2) & ~real
        if visible_painting.sum() > 30:
            objects["hand:2"] = Blob.of(visible_painting)
        session.append((frame.astype(np.uint8), objects, truth.astype(np.uint8)))
    return session, canvas
