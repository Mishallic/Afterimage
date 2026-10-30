"""Small image operations in numpy: resizing, fitting into a box, blurred backdrops."""

from __future__ import annotations

import numpy as np


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an RGB image to (width, height)."""
    width, height = size
    source_height, source_width = image.shape[:2]
    if (source_width, source_height) == (width, height):
        return image
    x = np.clip((np.arange(width) + 0.5) * source_width / width - 0.5, 0, source_width - 1)
    y = np.clip((np.arange(height) + 0.5) * source_height / height - 0.5, 0, source_height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    fx = (x - x0).astype(np.float32)[None, :, None]
    fy = (y - y0).astype(np.float32)[:, None, None]
    source = image.astype(np.float32)
    rows = source[y0] * (1 - fy) + source[y1] * fy
    return np.clip(rows[:, x0] * (1 - fx) + rows[:, x1] * fx + 0.5, 0, 255).astype(np.uint8)


def shrink(image: np.ndarray, factor: int) -> np.ndarray:
    """Downscale by an integer factor, averaging blocks (no aliasing)."""
    height, width = image.shape[0] // factor, image.shape[1] // factor
    blocks = image[:height * factor, :width * factor].astype(np.float32)
    return blocks.reshape(height, factor, width, factor, -1).mean(axis=(1, 3)).astype(np.uint8)


def fit_size(size: tuple[int, int], box: tuple[int, int]) -> tuple[int, int]:
    """Largest even size with the aspect of `size` that fits inside `box`."""
    scale = min(box[0] / size[0], box[1] / size[1])
    return max(2, int(size[0] * scale) // 2 * 2), max(2, int(size[1] * scale) // 2 * 2)


def crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    """Center crop to width / height == aspect."""
    height, width = image.shape[:2]
    if width / height > aspect:
        new_width = max(1, round(height * aspect))
        left = (width - new_width) // 2
        return image[:, left:left + new_width]
    new_height = max(1, round(width / aspect))
    top = (height - new_height) // 2
    return image[top:top + new_height]


def backdrop(image: np.ndarray, size: tuple[int, int], darken: float = 0.4) -> np.ndarray:
    """A heavily blurred, darkened fill of `size` made from `image`."""
    factor = max(1, min(image.shape[0], image.shape[1]) // 24)
    small = crop_to_aspect(shrink(image, factor), size[0] / size[1])
    blurred = resize(small, size).astype(np.float32) * darken
    return blurred.astype(np.uint8)


def paste(target: np.ndarray, image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top = center[0] - width // 2, center[1] - height // 2
    target[top:top + height, left:left + width] = image
    return target
