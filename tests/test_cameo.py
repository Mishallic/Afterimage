"""Cameo mode: the finished timelapse lands inside the canvas in the room shot, and nowhere else."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nohands import media
from nohands.cameo import render_cameo
from nohands.geometry import Warper, homography, polygon_mask
from nohands.masks import dilate
from nohands.receipt import receipt, time_map
from tests.synthetic import synthetic_session

ROOM = (640, 360)
CORNERS = np.array([[0.55, 0.18], [0.82, 0.2], [0.8, 0.78], [0.56, 0.76]])


class CameoTest(unittest.TestCase):
    def test_painting_lands_in_the_canvas(self) -> None:
        session = synthetic_session(frames=40)
        canvas_size = (session[0][2].shape[1], session[0][2].shape[0])
        quad = CORNERS * ROOM
        rect = np.array([[0, 0], [canvas_size[0], 0], [canvas_size[0], canvas_size[1]], [0, canvas_size[1]]], float)
        blank = np.full((canvas_size[1], canvas_size[0], 3), 228, np.uint8)
        region = polygon_mask(quad, (ROOM[1], ROOM[0]))
        room = np.full((ROOM[1], ROOM[0], 3), (70, 60, 50), np.uint8)
        room[region] = Warper(homography(quad, rect), ROOM, blank.shape[:2]).image(blank)[region]

        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            with media.FrameWriter(work / "room.mp4", ROOM, 30) as clip:
                for _ in range(30):
                    clip.write(room)
            with media.FrameWriter(work / "timelapse.mp4", canvas_size, 30) as timelapse:
                for _, _, truth in session:
                    timelapse.write(truth)
            out = render_cameo(work / "room.mp4", CORNERS, work / "timelapse.mp4", work / "cameo.mp4",
                               seconds=2.0, hold=0.5, width=ROOM[0])
            last = list(media.read_frames(out, fps=30, size=ROOM))[-1].astype(np.int16)

        expected = Warper(homography(quad, rect), ROOM, session[-1][2].shape[:2]).image(session[-1][2])
        inner = region & ~dilate(~region, 4)
        outer = ~dilate(region, 4)
        inside_error = np.abs(last - expected.astype(np.int16)).max(axis=2)[inner]
        outside_error = np.abs(last - room.astype(np.int16)).max(axis=2)[outer]
        self.assertLess(np.median(inside_error), 12)
        self.assertLess(np.median(outside_error), 6)


class ReceiptTest(unittest.TestCase):
    def test_time_map_and_grid(self) -> None:
        painted_at = np.full((60, 90), -1, np.int32)
        painted_at[10:20, 10:30] = 5
        painted_at[30:50, 40:80] = 95
        colours = time_map(painted_at, 100)
        self.assertEqual(colours.shape, (60, 90, 3))
        self.assertTrue((colours[0, 0] == (46, 46, 52)).all())
        self.assertGreater(int(colours[40, 60, 0]), int(colours[15, 20, 0]))
        lookup = receipt(painted_at, sample_fps=1.0, width=45)
        self.assertEqual(len(lookup["grid"]), lookup["width"] * lookup["height"])
        self.assertIn(95, lookup["grid"])
        self.assertEqual(lookup["to_camera"], np.eye(3).ravel().tolist())

    def test_tapped_spot_maps_back_into_the_camera_frame(self) -> None:
        corners = ((0.2, 0.1), (0.8, 0.15), (0.75, 0.9), (0.2, 0.85))
        matrix = np.array(receipt(np.zeros((4, 4), np.int32), sample_fps=1.0, corners=corners)["to_camera"]).reshape(3, 3)

        def mapped(u: float, v: float) -> tuple[float, float]:
            x, y, w = matrix @ (u, v, 1.0)
            return round(x / w, 6), round(y / w, 6)

        self.assertEqual(mapped(0, 0), (0.2, 0.1))
        self.assertEqual(mapped(1, 1), (0.75, 0.9))
        centre = mapped(0.5, 0.5)
        self.assertTrue(0.4 < centre[0] < 0.6 and 0.4 < centre[1] < 0.6)


if __name__ == "__main__":
    unittest.main()
