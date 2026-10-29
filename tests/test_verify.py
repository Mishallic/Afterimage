"""The receipt holds up: the real footage verifies, and anything tampered with doesn't."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nohands import media
from nohands.pipeline import Params, run, verify
from tests.synthetic import synthetic_session

SIZE = (320, 240)


def write_session(path: Path, *, nudge: bool = False) -> None:
    with media.FrameWriter(path, SIZE, fps=2) as writer:
        for index, (frame, _, _) in enumerate(synthetic_session(frames=40, size=SIZE)):
            if nudge and index == 20:
                frame = frame.copy()
                frame[100:120, 100:140] = (250, 30, 30)
            writer.write(frame)


class VerifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory()
        work = Path(cls.folder.name)
        cls.video = work / "session.mp4"
        cls.tampered = work / "tampered.mp4"
        write_session(cls.video)
        write_session(cls.tampered, nudge=True)
        cls.out = work / "out"
        cls.receipt = run(cls.video, cls.out, Params(fps=2, width=SIZE[0], masks="motion"), title="t", reels=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def test_receipt_contents(self) -> None:
        self.assertEqual(self.receipt["output"]["frames"], 40)
        self.assertEqual(self.receipt["generative_steps"], [])
        self.assertEqual(len(self.receipt["output"]["canvas_sha256"]), 64)

    def test_a_start_offset_skips_the_first_seconds_everywhere(self) -> None:
        whole = list(media.read_frames(self.video, fps=2, size=SIZE))
        later = list(media.read_frames(self.video, fps=2, size=SIZE, start=2.0))
        self.assertEqual(len(later), len(whole) - 4)
        difference = np.abs(later[0].astype(np.int16) - whole[4].astype(np.int16)).mean()
        self.assertLess(difference, 3.0)
        proxy = self.out / "from2.mp4"
        media.make_proxy(self.video, proxy, fps=2, size=SIZE, start=2.0)
        self.assertAlmostEqual(media.probe(proxy).duration, 18.0, delta=0.6)
        picked = next(iter(media.read_indices(self.video, [0], fps=2, size=SIZE, start=2.0)))
        self.assertLess(np.abs(picked.astype(np.int16) - whole[4].astype(np.int16)).mean(), 3.0)

    def test_the_moments_copy_runs_at_natural_speed(self) -> None:
        info = media.probe(self.out / "moments.mp4")
        self.assertAlmostEqual(info.duration, 20.0, delta=0.6)
        self.assertEqual((info.width, info.height), SIZE)

    def test_the_painting_is_peeled_down_to_the_bare_canvas(self) -> None:
        page = (self.out / "index.html").read_text(encoding="utf-8")
        stills = sorted(self.out.glob("underneath-*.jpg"))
        self.assertTrue(stills)
        self.assertTrue((self.out / "underneath.mp4").exists())
        self.assertIn(stills[0].name, page)

    def test_real_footage_verifies(self) -> None:
        verdict = verify(self.out / "receipt.json", self.video)
        self.assertTrue(verdict.ok, verdict.checks)

    def test_different_footage_fails(self) -> None:
        self.assertFalse(verify(self.out / "receipt.json", self.tampered).ok)

    def test_edited_receipt_fails(self) -> None:
        edited = json.loads((self.out / "receipt.json").read_text(encoding="utf-8"))
        edited["output"]["canvas_sha256"] = "0" * 64
        path = self.out / "edited.json"
        path.write_text(json.dumps(edited), encoding="utf-8")
        verdict = verify(path, self.video)
        self.assertFalse(verdict.ok)
        self.assertTrue(verdict.checks[0][1], "footage itself should still match")


if __name__ == "__main__":
    unittest.main()
