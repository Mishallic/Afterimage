"""Stats from a synthetic session, and a full render of the auto-cut reel from them."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from afterimage import media
from afterimage.autocut import auto_edl
from afterimage.cleanplate import CleanPlate, CleanPlateConfig, first_visible_canvas
from afterimage.render import Sources, render
from afterimage.stats import PIGMENTS, StatsTracker
from tests.synthetic import synthetic_session

FPS = 2.0


def session_stats():
    session = synthetic_session(frames=60)
    config = CleanPlateConfig(dilate_px=6)
    start = first_visible_canvas((f for f, _, _ in session), (m for _, m, _ in session), dilate_px=6)
    plate = CleanPlate(start, config)
    tracker = StatsTracker(start, FPS)
    canvases = []
    for frame, mask, _ in session:
        canvas = plate.update(frame, mask)
        tracker.update(canvas, plate.last_commit, float(mask.mean()))
        canvases.append(canvas)
    return session, canvases, tracker.finish(canvases[-1]).to_dict()


class StatsTest(unittest.TestCase):
    def test_stats_make_sense(self) -> None:
        _, _, stats = session_stats()
        self.assertEqual(stats["frames"], 60)
        self.assertIsNotNone(stats["first_paint_frame"])
        self.assertGreater(stats["apm"], 0)
        self.assertIn(stats["mvp_color"], PIGMENTS)
        self.assertNotEqual(stats["mvp_color"], "titanium white", "bare canvas counted as paint")
        self.assertGreater(stats["coverage"][-1], stats["coverage"][0])
        self.assertTrue(0 <= stats["hook_frame"] < 60)


class RenderTest(unittest.TestCase):
    def test_auto_reel_renders(self) -> None:
        session, canvases, stats = session_stats()
        size = (session[0][0].shape[1], session[0][0].shape[0])
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            with media.FrameWriter(work / "clean.mp4", size, 30) as clean, \
                    media.FrameWriter(work / "raw.mp4", size, 30) as raw:
                for (frame, _, _), canvas in zip(session, canvases):
                    clean.write(canvas)
                    raw.write(frame)
            edl = auto_edl(stats, layout="landscape")
            edl.validate(stats["frames"])
            sources = Sources(clean=work / "clean.mp4", raw=work / "raw.mp4", size=size, fps=30)
            reel = render(edl, sources, work / "reel.mp4")
            info = media.probe(reel)
            self.assertEqual((info.width, info.height), (1920, 1080))
            self.assertAlmostEqual(info.duration, edl.duration, delta=0.5)


if __name__ == "__main__":
    unittest.main()
