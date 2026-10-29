"""The critic loop: Spark points at a leftover, SAM tracks it (or the area is held), and it's gone.

SAM is played by masks that miss the outer 10 px of the resting hand, the way a soft mask
edge does. The hand rests until the end, so the leaked ring can't be vetoed as transient:
only the critic can remove it.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nohands import media, pipeline, spark
from nohands.masks import Blob, dilate
from nohands.pipeline import Params
from tests.synthetic import resting_hand_session

SIZE = (320, 240)
FRAMES = 60
REST = (20, FRAMES)
ERODE = 10
HAND_BOX = [380, 340, 560, 580]
"""The resting hand (centre 150,110, radius 22) on Spark's 0-1000 grid."""
SHADOW_BOX = [440, 430, 620, 660]
"""Its shadow, 20 px down and right."""


def to_pixels(box: list[int]) -> tuple[slice, slice]:
    left, top, right, bottom = spark.to_pixels(box, SIZE)
    return slice(top, bottom), slice(left, right)


class CriticLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory()
        cls.work = Path(cls.folder.name)
        cls.video = cls.work / "session.mp4"
        session = resting_hand_session(frames=FRAMES, size=SIZE, rest=REST)
        cls.truth = session[-1][2]
        cls.full = [mask for _, mask, _, _ in session]
        cls.eroded = [~dilate(~mask, ERODE) for mask in cls.full]
        # What the plate leaves uncovered of the resting hand once the eroded mask is dilated back by 4.
        cls.ring = cls.full[-1] & ~dilate(cls.eroded[-1], 4)
        cls.shadow = session[-1][3]
        with media.FrameWriter(cls.video, SIZE, fps=2) as writer:
            for frame, _, _, _ in session:
                writer.write(frame)
        cls.params = Params(fps=2, width=SIZE[0], sam_width=SIZE[0], dilate=4, persist_seconds=2.0,
                            masks="sam", scan=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def mocks(self, answers: list[dict]) -> tuple[mock.Mock, mock.Mock]:
        async def sam_objects(video, params, proxy_size, events):
            for path in events.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("cached", encoding="utf-8")
            return {index: {"hand:1": Blob.of(mask)} for index, mask in enumerate(self.eroded)}

        async def tracked(proxy, proxy_size, boxes, events_file, frames, work, prefix):
            events_file.write_text("tracked", encoding="utf-8")
            return {index: {f"{prefix}1": Blob.of(self.full[index])} for index in range(REST[0], REST[1])}

        asked = mock.Mock(side_effect=lambda api, pairs: answers.pop(0))
        tracker = mock.AsyncMock(side_effect=tracked)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(pipeline, "_sam_objects", sam_objects).start()
        mock.patch.object(pipeline, "_tracked_objects", tracker).start()
        mock.patch.object(spark, "client").start()
        mock.patch.object(spark, "critic_answer", asked).start()
        return asked, tracker

    def run_session(self, name: str, params: Params) -> tuple[dict, Path, np.ndarray]:
        out = self.work / name
        if out.exists():
            shutil.rmtree(out)
        receipt = pipeline.run(self.video, out, params, title="t", reels=False)
        final = None
        for final in media.read_frames(out / "timelapse.mp4", fps=params.output_fps, size=SIZE):
            pass
        return receipt, out, final

    def error(self, final: np.ndarray, region: np.ndarray) -> float:
        return float(np.abs(final.astype(np.int16) - self.truth.astype(np.int16)).max(axis=2)[region].mean())

    def test_without_the_critic_the_leaked_ring_stays(self) -> None:
        self.mocks([])
        receipt, _, final = self.run_session("plain", pipeline.Params(**{**self.params.to_dict(), "critic_rounds": 0}))
        self.assertEqual(receipt["critic"]["rounds"], [])
        self.assertGreater(self.error(final, self.ring), 50)

    def test_a_hand_leftover_goes_back_to_sam_and_disappears(self) -> None:
        def answer(api, pairs):
            if asked.call_count > 1:
                return {"defects": []}
            return {"defects": [{"image": i, "box": HAND_BOX, "kind": "hand", "confidence": 0.9} for i in range(len(pairs))]}

        asked, tracker = self.mocks([])
        asked.side_effect = answer
        receipt, out, final = self.run_session("hand", self.params)

        rounds = receipt["critic"]["rounds"]
        self.assertEqual([entry["round"] for entry in rounds], [1, 2])
        self.assertIn(FRAMES - 1, rounds[0]["keyframes"])
        self.assertEqual(rounds[0]["tracked"], len(rounds[0]["keyframes"]))
        self.assertEqual(rounds[0]["held"], 0)
        self.assertEqual(rounds[1]["defects"], [])
        tracker.assert_called_once()
        self.assertEqual([stream["concept"] for stream in receipt["sam_streams"]], ["person", "hand", "paintbrush", "critic1"])
        self.assertLess(self.error(final, self.ring), 20)
        self.assertEqual(asked.call_count, 2)

        verdict = pipeline.verify(out / "receipt.json", self.video)
        self.assertTrue(verdict.ok, verdict.checks)
        self.assertEqual(asked.call_count, 2, "verify must not ask Spark again")

    def test_a_shadow_is_held_rather_than_tracked(self) -> None:
        answers = [{"defects": [{"image": 0, "box": SHADOW_BOX, "kind": "shadow", "confidence": 0.8}]}, {"defects": []}]
        asked, tracker = self.mocks(answers)
        receipt, _, final = self.run_session("shadow", self.params)

        first = receipt["critic"]["rounds"][0]
        self.assertEqual(first["tracked"], 0)
        self.assertGreater(first["held"], 0)
        tracker.assert_not_called()
        rows, cols = to_pixels(SHADOW_BOX)
        boxed = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
        boxed[rows, cols] = True
        self.assertLess(self.error(final, self.shadow & boxed), 20)
        self.assertGreater(self.error(final, self.ring), 50, "the hand ring was not reported, so it stays")

    def test_low_confidence_findings_are_ignored(self) -> None:
        answers = [{"defects": [{"image": 0, "box": HAND_BOX, "kind": "hand", "confidence": 0.2}]}]
        asked, tracker = self.mocks(answers)
        receipt, _, _ = self.run_session("doubtful", self.params)
        self.assertEqual(len(receipt["critic"]["rounds"]), 1)
        self.assertEqual(receipt["critic"]["rounds"][0]["defects"], [])
        tracker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
