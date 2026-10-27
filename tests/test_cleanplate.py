"""The clean plate should end up showing all the paint and none of the painter."""

from __future__ import annotations

import unittest

import numpy as np

from nohands.cleanplate import CleanPlate, CleanPlateConfig, first_visible_canvas, visible_mask
from tests.synthetic import SKIN, synthetic_session


def _error(canvas: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(canvas.astype(np.int16) - truth.astype(np.int16)).max(axis=2)


class CleanPlateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = synthetic_session()
        self.config = CleanPlateConfig(dilate_px=6)

    def run_plate(self) -> list[np.ndarray]:
        start = first_visible_canvas(
            (frame for frame, _, _ in self.session),
            (mask for _, mask, _ in self.session),
            dilate_px=self.config.dilate_px,
        )
        plate = CleanPlate(start, self.config)
        return [plate.update(frame, mask) for frame, mask, _ in self.session]

    def start_canvas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The start canvas, the true canvas at frame 0, and what frame 0 shows uncovered."""
        start = first_visible_canvas(
            (frame for frame, _, _ in self.session),
            (mask for _, mask, _ in self.session),
            dilate_px=6, steady_frames=4,
        )
        first_mask = self.session[0][1]
        return start, self.session[0][2], visible_mask(first_mask, first_mask.shape, 6)

    def test_start_skips_the_shadow_in_the_first_frame(self) -> None:
        start, truth, seen = self.start_canvas()
        shadowed = seen & (_error(self.session[0][0], truth) > 25)
        self.assertGreater(shadowed.sum(), 1000)
        self.assertLess((_error(start, truth)[shadowed] > 25).mean(), 0.02)
        # Matched to frame 0's exposure, though the frames after it jitter by up to 7%.
        self.assertLess(_error(start, truth)[seen].mean(), 0.5)

    def test_paint_from_the_first_moments_still_arrives_as_new_paint(self) -> None:
        start, truth, seen = self.start_canvas()
        early = seen & (_error(self.session[6][2], truth) > 25)
        self.assertGreater(early.sum(), 500)
        self.assertLess((_error(start, truth)[early] > 25).mean(), 0.02)

    def test_final_canvas_matches_the_painting(self) -> None:
        final = self.run_plate()[-1]
        error = _error(final, self.session[-1][2])
        self.assertLess(error.mean(), 3.0)
        self.assertLess((error > 25).mean(), 0.01)

    def test_painter_never_shows_up(self) -> None:
        skin = np.array(SKIN, dtype=np.int16)
        for index, canvas in enumerate(self.run_plate()[5:], start=5):
            close_to_skin = np.abs(canvas.astype(np.int16) - skin).max(axis=2) < 12
            self.assertLess(close_to_skin.mean(), 0.002, f"skin visible in frame {index}")

    def test_beats_the_raw_footage(self) -> None:
        outputs = self.run_plate()
        raw_errors, clean_errors = [], []
        for (frame, _, truth), canvas in zip(self.session[10:-8], outputs[10:-8]):
            raw_errors.append((_error(frame, truth) > 25).mean())
            clean_errors.append((_error(canvas, truth) > 25).mean())
        self.assertLess(np.mean(clean_errors), np.mean(raw_errors) / 3)


if __name__ == "__main__":
    unittest.main()
