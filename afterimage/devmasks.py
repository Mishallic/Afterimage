"""Offline stand-in for SAM masks while developing without API access. Not used in the product.

It flags pixels that changed since the previous frame: a moving hand gets masked, paint
that stays put doesn't. A hand resting still for a few frames slips through, which is
exactly the case real SAM masks are for.
"""

from __future__ import annotations

import numpy as np


class MotionMasker:
    def __init__(self, threshold: float = 25.0) -> None:
        self.threshold = threshold
        self._previous: np.ndarray | None = None

    def mask(self, frame: np.ndarray) -> np.ndarray:
        current = frame.astype(np.float32)
        previous = current if self._previous is None else self._previous
        self._previous = current
        # Compensate for exposure jumps between frames so they don't flag the whole canvas.
        lit = previous.min(axis=2) > 20
        if lit.any():
            previous = previous * np.median(current[lit] / previous[lit], axis=0)
        return np.abs(current - previous).max(axis=2) > self.threshold
