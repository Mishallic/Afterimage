"""A knocked easel: the canvas's corners should follow it, and stay put when it doesn't move."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from afterimage import media
from afterimage.geometry import Warper, homography
from afterimage.pipeline import Params, run, verify
from afterimage.track import follow

VIEW = (640, 360)
WALL = (172, 166, 158)
SKIN = (200, 160, 130)
CANVAS = (300, 400)
START = np.array([[220, 40], [420, 46], [418, 312], [214, 316]], dtype=np.float64)


def painting(seed: int = 3, strokes: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = CANVAS
    image = np.full((height, width, 3), 238.0)
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(strokes):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        rx, ry = rng.uniform(6, 40), rng.uniform(3, 14)
        angle = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        image[(u / rx) ** 2 + (v / ry) ** 2 < 1] = rng.uniform(20, 220, size=3)
    return image.astype(np.uint8)


def moved(corners: np.ndarray, shift=(0.0, 0.0), turn_degrees: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """The canvas knocked: turned and scaled about its centre, then shifted."""
    centre = corners.mean(axis=0)
    angle = np.radians(turn_degrees)
    rotation = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (corners - centre) @ rotation.T + centre + np.asarray(shift)


def session(places: list[np.ndarray], canvas: np.ndarray, seed: int = 1) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Camera frames of `canvas` at each frame's corners, with a hand (in the mask) and its shadow (not)."""
    rng = np.random.default_rng(seed)
    width, height = CANVAS
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    yy, xx = np.mgrid[0:VIEW[1], 0:VIEW[0]]
    frames, masks = [], []
    for index, corners in enumerate(places):
        frame = np.empty((VIEW[1], VIEW[0], 3), dtype=np.float32)
        frame[:] = WALL
        frame[120:260, 60:150] = (24, 24, 28)  # a monitor beside the easel: it never moves
        inside = Warper(homography(corners, rect), VIEW, canvas.shape[:2])
        view = inside.image(canvas).astype(np.float32)
        on_canvas = view.max(axis=2) > 0
        frame[on_canvas] = view[on_canvas]
        mask = np.zeros((VIEW[1], VIEW[0]), dtype=bool)
        if index % 4 != 0:
            hx, hy = 300 + 80 * np.sin(index * 0.7), 180 + 90 * np.cos(index * 0.5)
            mask = (xx - hx) ** 2 + (yy - hy) ** 2 < 32**2
            shadow = np.roll(mask, (14, 18), axis=(0, 1)) & ~mask
            frame[shadow] *= 0.75
            frame[mask] = SKIN
        frame = frame * rng.uniform(0.95, 1.05) + rng.normal(0, 2.0, frame.shape)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        masks.append(mask)
    return frames, masks


class FollowTest(unittest.TestCase):
    def test_a_knocked_easel_is_followed(self) -> None:
        first = moved(START, shift=(9.0, -3.0), turn_degrees=0.4, scale=0.985)
        second = moved(first, shift=(4.0, 1.5), turn_degrees=-0.2)
        places = [START] * 12 + [first] * 12 + [second] * 10
        frames, masks = session(places, painting())
        moves = follow(frames, START, lambda index: masks[index])
        self.assertEqual(len(moves), 2, [frame for frame, _ in moves])
        for (frame, corners), (when, truth) in zip(moves, [(12, first), (24, second)]):
            self.assertIn(frame, range(when, when + 3))
            self.assertLess(np.abs(corners - truth).max(), 0.3, f"frame {frame}: {corners - truth}")

    def test_a_still_canvas_stays_put(self) -> None:
        frames, masks = session([START] * 30, painting())
        self.assertEqual(follow(frames, START, lambda index: masks[index]), [])

    def test_a_blank_canvas_holds_its_corners(self) -> None:
        # Nothing painted yet: nothing to place it by, and nothing on it to misplace.
        blank = painting(strokes=0)
        frames, masks = session([START] * 10 + [moved(START, shift=(8.0, 0.0))] * 10, blank)
        self.assertEqual(follow(frames, START, lambda index: masks[index]), [])


class FollowedRunTest(unittest.TestCase):
    def test_a_run_follows_the_canvas_and_its_receipt_verifies(self) -> None:
        knocked = moved(START, shift=(10.0, -2.0), turn_degrees=0.3)
        frames, _ = session([START] * 12 + [knocked] * 24, painting(strokes=25))
        corners = tuple(tuple(point) for point in (START / np.array(VIEW)).round(4))
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "knocked.mp4"
            with media.FrameWriter(video, VIEW, fps=2) as writer:
                for frame in frames:
                    writer.write(frame)
            out = Path(folder) / "out"
            receipt = run(video, out, Params(fps=2, width=VIEW[0], masks="none", corners=corners), reels=False)
            moves = Params.from_dict(receipt["params"]).moves
            self.assertEqual(len(moves), 1, moves)
            frame, where = moves[0]
            self.assertIn(frame, range(12, 15))
            self.assertLess(np.abs(np.array(where) * VIEW - knocked).max(), 0.5)
            # Verify straightens each frame where the receipt says the canvas stood, and gets the same frames.
            self.assertTrue(verify(out / "receipt.json", video).ok)
            # The page puts a tapped spot painted after the knock on the canvas where it then stood.
            self.assertIn('"moves": [[', (out / "index.html").read_text(encoding="utf-8"))
            # Held at the first corners, every edge of the painting reads as repainted where it
            # slid; followed, little is left of the knock.
            fixed = run(video, Path(folder) / "fixed", Params(fps=2, width=VIEW[0], masks="none", corners=corners),
                        reels=False, follow_canvas=False)
            counts = (receipt["timeline"]["kept"], fixed["timeline"]["kept"])
            self.assertLess(10 * counts[0], counts[1], counts)


if __name__ == "__main__":
    unittest.main()
