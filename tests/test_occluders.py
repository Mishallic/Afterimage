"""Paint a hand: the real one must vanish, the painted one must appear."""

from __future__ import annotations

import unittest

import numpy as np

from afterimage.cleanplate import CleanPlate, CleanPlateConfig, first_visible_canvas
from afterimage.occluders import occluders, painted
from tests.synthetic import SKIN, painted_hand_session

FPS = 2.0


def _error(canvas: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(canvas.astype(np.int16) - truth.astype(np.int16)).max(axis=2)


def run(session, masks) -> np.ndarray:
    config = CleanPlateConfig(dilate_px=6)
    start = first_visible_canvas(
        (frame for frame, _, _ in session), (masks.get(i) for i in range(len(session))), dilate_px=6
    )
    plate = CleanPlate(start, config)
    for index, (frame, _, _) in enumerate(session):
        result = plate.update(frame, masks.get(index))
    return result


class PaintedHandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session, self.canvas = painted_hand_session()
        self.objects = {index: objects for index, (_, objects, _) in enumerate(self.session)}
        self.truth = self.session[-1][2]
        self.painted_area = (self.truth == SKIN).all(axis=2)

    def test_tells_the_painted_hand_from_the_real_one(self) -> None:
        self.assertEqual(painted(self.objects, self.canvas, FPS), {"hand:2"})

    def test_real_hand_erased_and_painted_hand_kept(self) -> None:
        keys = painted(self.objects, self.canvas, FPS)
        final = run(self.session, occluders(self.objects, keys, self.canvas.shape))
        error = _error(final, self.truth)
        self.assertLess(error.mean(), 3.0)
        self.assertLess(error[self.painted_area].mean(), 8.0, "painted hand missing")

    def test_without_the_check_the_painted_hand_never_appears(self) -> None:
        final = run(self.session, occluders(self.objects, set(), self.canvas.shape))
        self.assertGreater(_error(final, self.truth)[self.painted_area].mean(), 40.0)



class CanvasItselfTest(unittest.TestCase):
    """An object that covers most of the canvas and stays on it is the canvas, not an occluder."""

    def test_the_canvas_found_as_a_palette_is_kept_and_the_painter_is_not(self) -> None:
        from afterimage.masks import Blob
        from afterimage.occluders import painted

        shape = (120, 160)
        canvas = np.zeros(shape, dtype=bool)
        canvas[20:100, 40:120] = True
        objects = {}
        for frame in range(60):
            # SAM's outline of the canvas: a little past its edge, its shape shifting as paint goes on.
            whole = np.zeros(shape, dtype=bool)
            whole[18:102, 38 + frame % 5:122] = True
            # The painter in front of the left half, reaching in from below the canvas.
            person = np.zeros(shape, dtype=bool)
            person[50:120, 30:90] = True
            objects[frame] = {"palette with paint:c1:0": Blob.of(whole), "person:c1:1": Blob.of(person)}
        keys = painted(objects, canvas, fps=1.0)
        self.assertIn("palette with paint:c1:0", keys)
        self.assertNotIn("person:c1:1", keys)


if __name__ == "__main__":
    unittest.main()
