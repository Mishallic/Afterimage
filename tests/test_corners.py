"""Rough corners from Spark should land on the canvas's real edges, clamp or no clamp."""

from __future__ import annotations

import unittest

import numpy as np

from afterimage.corners import snap_corners
from afterimage.geometry import polygon_mask

VIEW = (640, 480)
WALL = (150, 140, 125)
CANVAS = (236, 238, 240)
CLAMP = (205, 185, 150)
TRUE = np.array([[100, 60], [540, 70], [560, 430], [80, 420]], dtype=np.float64)


def scene(corners: np.ndarray, *, clamp: bool = True, noise: float = 3.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.empty((VIEW[1], VIEW[0], 3), dtype=np.float32)
    frame[:] = WALL
    frame[polygon_mask(corners, (VIEW[1], VIEW[0]))] = CANVAS
    if clamp:
        frame[20:95, 290:350] = CLAMP  # light wood across the top edge, as on an easel
    frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


class SnapCornersTest(unittest.TestCase):
    def test_rough_corners_land_on_the_edges(self) -> None:
        rng = np.random.default_rng(5)
        for trial in range(5):
            rough = TRUE + rng.uniform(-28, 28, size=(4, 2))
            snapped = snap_corners(scene(TRUE, seed=trial), rough)
            self.assertIsNotNone(snapped, f"trial {trial}")
            self.assertLess(np.abs(snapped - TRUE).max(), 1.5, f"trial {trial}: {snapped}")

    def test_corners_a_tenth_of_the_frame_off_still_land(self) -> None:
        # Spark 1.3 put session 1's corners about 13 % of the frame too high.
        rough = TRUE + np.array([[10, -65], [-15, -60], [20, -70], [-10, -62]])
        snapped = snap_corners(scene(TRUE), rough)
        self.assertIsNotNone(snapped)
        self.assertLess(np.abs(snapped - TRUE).max(), 1.5, str(snapped))

    def test_the_clamp_does_not_pull_the_top_edge_up(self) -> None:
        rough = TRUE + np.array([[0, -24], [0, -24], [0, 0], [0, 0]])
        with_clamp = snap_corners(scene(TRUE, clamp=True), rough)
        without = snap_corners(scene(TRUE, clamp=False), rough)
        self.assertLess(np.abs(with_clamp - without).max(), 1.0)

    def test_no_canvas_means_no_answer(self) -> None:
        wall = np.full((VIEW[1], VIEW[0], 3), WALL, dtype=np.uint8)
        self.assertIsNone(snap_corners(wall, TRUE))

    def test_corners_far_off_are_recovered_from_the_bright_region(self) -> None:
        # Even a third of the frame away, the quad's centre is still on the canvas, so the
        # coarse fix finds the bright region and the edges are searched from there.
        snapped = snap_corners(scene(TRUE), TRUE + 150)
        self.assertIsNotNone(snapped)
        self.assertLess(np.abs(snapped - TRUE).max(), 1.5, str(snapped))

    def test_a_quad_on_the_wall_is_refused(self) -> None:
        on_the_wall = np.array([[575, 60], [635, 70], [638, 430], [572, 420]], dtype=np.float64)
        self.assertIsNone(snap_corners(scene(TRUE), on_the_wall))


if __name__ == "__main__":
    unittest.main()
