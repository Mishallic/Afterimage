"""The held-out protocol: hide the frames where the painter stepped back, rebuild, compare."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nohands import heldout, media, pipeline
from nohands.heldout import METHODS, clear_windows, false_commit_rate, heldout_report, markdown
from nohands.pipeline import Params
from nohands.timeline import Timeline
from tests.synthetic import synthetic_session

SIZE = (320, 240)
AWAY = (36, 46)
"""The painter steps back mid-session; synthetic_session also has them step out for its last 8 frames."""


def write_step_back_session(path: Path, frames: int = 80, *, plant: bool = False) -> dict[int, np.ndarray]:
    """Write the session to `path` and return its masks, as SAM would give them.

    `plant` adds a stroke painted under the arm just before the step-back: it's first
    uncovered at the window's first frame, so only a method that sees the window knows it.
    """
    session = synthetic_session(frames=frames, size=SIZE)
    rng = np.random.default_rng(3)
    stroke = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    stroke[20:60, 20:120] = True
    masks = {}
    with media.FrameWriter(path, SIZE, fps=2) as writer:
        for index, (frame, mask, _) in enumerate(session):
            if AWAY[0] <= index < AWAY[1]:
                frame = np.clip(session[AWAY[0] - 1][2] * rng.uniform(0.95, 1.05), 0, 255).astype(np.uint8)
                mask = np.zeros_like(mask)
            if plant and index >= AWAY[0] - 6:
                frame = frame.copy()
                frame[stroke] = (200, 30, 30)
                if index <= AWAY[0]:
                    mask = mask | stroke
            if mask.any():
                masks[index] = mask
            writer.write(frame)
    return masks


class ClearWindowsTest(unittest.TestCase):
    def test_long_enough_clear_stretches_trimmed_at_both_ends(self) -> None:
        occlusion = [0.1] * 5 + [0.0] * 8 + [0.2] * 3 + [0.0] * 4 + [0.3] + [0.0] * 7
        self.assertEqual(clear_windows(occlusion, min_frames=5), [(6, 12), (22, 27)])

    def test_false_commits_per_minute(self) -> None:
        zeros = np.zeros((6, 3), np.uint8)
        timeline = Timeline((10, 10), np.arange(6, dtype=np.int32), np.array([1, 5, 6, 7, 20, 21], np.int32), zeros, zeros)
        # 3 of 100 pixels committed during 3 frames at 1 fps: 3% in 0.05 minutes.
        self.assertAlmostEqual(false_commit_rate(timeline, [(5, 8)], fps=1.0), 60.0)


class HeldOutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory()
        cls.video = Path(cls.folder.name) / "session.mp4"
        cls.masks = write_step_back_session(cls.video)
        params = Params(fps=2, width=SIZE[0], dilate=6, persist_seconds=2.0)
        cls.report = heldout_report(cls.video, params, cls.masks, static=[(10, 20)])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def test_both_step_backs_are_found_and_scored(self) -> None:
        windows = [(entry["start_s"], entry["stop_s"]) for entry in self.report["windows"]]
        self.assertEqual(windows, [(18.5, 22.5), (36.5, 39.5)])
        self.assertEqual(self.report["rejected"], [])

    def test_nohands_rebuilds_the_hidden_frames_better_than_the_baselines(self) -> None:
        summary = self.report["summary"]
        ours = summary["NoHands"]
        self.assertLess(ours["clearly_off_pct"], 0.5)
        for baseline in ("masked median (Loomis-style)", "temporal median (no masks)"):
            self.assertLess(ours["delta_e"], summary[baseline]["delta_e"], baseline)

    def test_static_stretches_are_scored(self) -> None:
        self.assertEqual(self.report["static"]["minutes"], round(10 / 2 / 60, 2))
        self.assertGreaterEqual(self.report["static"]["false_commit_pct_per_min"], 0)

    def test_markdown_lists_every_method_and_stretch(self) -> None:
        text = markdown(self.report, "session.mp4")
        for method in METHODS:
            self.assertIn(f"| {method} | {self.report['summary'][method]['delta_e']} |", text)
        self.assertIn("| 18.5–22.5 | 8 |", text)
        self.assertIn("False commits while not painting", text)


class HiddenMeansUnseenTest(unittest.TestCase):
    def test_paint_first_uncovered_in_a_hidden_window_is_missed_unless_peeked_at(self) -> None:
        params = Params(fps=2, width=SIZE[0], dilate=6, persist_seconds=2.0)
        windows = [(AWAY[0] + 1, AWAY[1] - 1)]
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "planted.mp4"
            masks = write_step_back_session(video, plant=True)
            hidden = heldout_report(video, params, masks, windows=windows)

            def peeking(*args, **kwargs):
                return pipeline.clean(*args, **{**kwargs, "hidden": frozenset()})

            with mock.patch.object(heldout, "clean", peeking):
                peeked = heldout_report(video, params, masks, windows=windows)
        # The stroke covers 5.2% of the frame.
        self.assertGreater(hidden["summary"]["NoHands"]["clearly_off_pct"], 4.0)
        self.assertLess(peeked["summary"]["NoHands"]["clearly_off_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
