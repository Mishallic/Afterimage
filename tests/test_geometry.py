"""Straightening a tilted view of the canvas should give the canvas back."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nohands import media
from nohands.geometry import Warper, corners_from_mask, homography, order_corners

CANVAS = (400, 300)
VIEW = (640, 480)
TILTED = np.array([[120, 90], [520, 70], [560, 400], [90, 420]], dtype=np.float64)


def canvas_image() -> np.ndarray:
    width, height = CANVAS
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack([xx * 255 / width, yy * 255 / height, np.full_like(xx, 128.0)], axis=2)
    image[60:140, 80:200] = (20, 200, 40)
    return image.astype(np.uint8)


def camera_view(canvas: np.ndarray) -> np.ndarray:
    width, height = CANVAS
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    return Warper(homography(TILTED, rect), VIEW, canvas.shape[:2]).image(canvas)


class GeometryTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        canvas = canvas_image()
        back = Warper.facing(TILTED, CANVAS, (VIEW[1], VIEW[0])).image(camera_view(canvas))
        error = np.abs(back.astype(np.int16) - canvas.astype(np.int16))[10:-10, 10:-10]
        self.assertLess(error.mean(), 4.0)

    def test_corners_from_mask(self) -> None:
        canvas = np.full((CANVAS[1], CANVAS[0], 3), 255, dtype=np.uint8)
        mask = camera_view(canvas)[..., 0] > 128
        mask[:2, :] = mask[-2:, :] = False
        found = corners_from_mask(mask)
        self.assertIsNotNone(found)
        self.assertLess(np.abs(found - TILTED).max(), 3.0)

    def test_order_corners(self) -> None:
        shuffled = TILTED[[2, 0, 3, 1]]
        np.testing.assert_allclose(order_corners(shuffled), TILTED)

    def test_ffmpeg_straightens_the_footage_like_the_warper(self) -> None:
        canvas = canvas_image()
        view = camera_view(canvas)
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "view.mp4"
            with media.FrameWriter(video, VIEW, fps=2) as writer:
                for _ in range(4):
                    writer.write(view)
            fractions = TILTED / np.array(VIEW, dtype=np.float64)
            frames = list(media.read_frames(video, fps=2, size=CANVAS, corners=fractions, source=VIEW))
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0].shape, (CANVAS[1], CANVAS[0], 3))
        by_warper = Warper.facing(TILTED, CANVAS, (VIEW[1], VIEW[0])).image(view)
        inner = (slice(10, -10), slice(10, -10))
        self.assertLess(np.abs(frames[0].astype(np.int16) - by_warper.astype(np.int16))[inner].mean(), 4.0)
        self.assertLess(np.abs(frames[0].astype(np.int16) - canvas.astype(np.int16))[inner].mean(), 5.0)
        # Both put the green patch where the canvas has it: no disagreement about where the corners are.
        for image in (frames[0], by_warper):
            green = (image[..., 1] > 150) & (image[..., 2] < 90)
            green[:5] = green[-5:] = green[:, :5] = green[:, -5:] = False  # resampling smears the border
            rows, cols = np.nonzero(green)
            self.assertLessEqual(abs(rows.min() - 60) + abs(rows.max() + 1 - 140), 4)
            self.assertLessEqual(abs(cols.min() - 80) + abs(cols.max() + 1 - 200), 4)


if __name__ == "__main__":
    unittest.main()
