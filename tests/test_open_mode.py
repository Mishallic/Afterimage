"""Open-weight masks: SAM 3.1 on Modal stands in for the API, and the receipt still checks out.

The Modal calls are played by functions that cut masks from the synthetic session and write
the same mask files the deployed tracker returns.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nohands import maskfile, media, pipeline, sam_open, spark
from nohands.masks import Blob, dilate
from nohands.pipeline import Params
from tests.synthetic import resting_hand_session

SIZE = (320, 240)
FRAMES = 60
REST = (20, FRAMES)
ERODE = 10
HAND_BOX = [380, 340, 560, 580]


class OpenModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory()
        cls.work = Path(cls.folder.name)
        cls.video = cls.work / "session.mp4"
        session = resting_hand_session(frames=FRAMES, size=SIZE, rest=REST)
        cls.truth = session[-1][2]
        cls.full = [mask for _, mask, _, _ in session]
        cls.eroded = [~dilate(~mask, ERODE) for mask in cls.full]
        cls.ring = cls.full[-1] & ~dilate(cls.eroded[-1], 4)
        with media.FrameWriter(cls.video, SIZE, fps=2) as writer:
            for frame, _, _, _ in session:
                writer.write(frame)
        cls.params = Params(fps=2, width=SIZE[0], sam_width=SIZE[0], dilate=4, persist_seconds=2.0,
                            masks="open", scan=False, critic_rounds=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def mocks(self) -> tuple[mock.Mock, mock.Mock]:
        def concepts(proxy, names, out, *, frames=None):
            if not out.exists():
                maskfile.save(out, {i: {"hand:c0:1": Blob.of(m)} for i, m in enumerate(self.eroded)}, {"model": "test"})
            return maskfile.load(out)[0]

        def boxes(proxy, prompts, out, *, frames=None):
            if not out.exists():
                maskfile.save(out, {i: {"box:1:1": Blob.of(self.full[i])} for i in range(REST[0], REST[1])})
            return maskfile.load(out)[0]

        self.addCleanup(mock.patch.stopall)
        tracked = mock.patch.object(sam_open, "track_concepts", mock.Mock(side_effect=concepts)).start()
        boxed = mock.patch.object(sam_open, "track_boxes", mock.Mock(side_effect=boxes)).start()
        return tracked, boxed

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

    def test_masks_come_from_the_file_and_verify_checks_its_hash(self) -> None:
        tracked, _ = self.mocks()
        receipt, out, _ = self.run_session("open", self.params)
        tracked.assert_called_once()
        stream = receipt["sam_streams"][0]
        self.assertEqual(stream["concept"], "person, hand, paintbrush")
        self.assertTrue(stream["file"].endswith("concepts.npz"))
        self.assertIn("open weights", receipt["models"]["masks"])
        self.assertIsNone(receipt["models"]["spark"])
        self.assertTrue(pipeline.verify(out / "receipt.json", self.video).ok)

        with (out / stream["file"]).open("ab") as handle:
            handle.write(b"\0")
        verdict = pipeline.verify(out / "receipt.json", self.video)
        self.assertFalse(verdict.ok)
        self.assertFalse(verdict.checks[1][1], "the mask file's hash must be what fails")

    def test_the_critic_sends_its_boxes_to_modal_and_credits_spark_open(self) -> None:
        _, boxed = self.mocks()
        asked = mock.patch.object(spark, "critic_answer", mock.Mock(side_effect=[
            {"defects": [{"image": i, "box": HAND_BOX, "kind": "hand", "confidence": 0.9} for i in range(6)]},
        ])).start()
        mock.patch.object(spark, "client").start()
        mock.patch.dict(os.environ, {"SPARK_PROVIDER": "baseten", "BASETEN_API_KEY": "bt-test"}).start()
        params = Params(**{**self.params.to_dict(), "critic_rounds": 1})
        receipt, out, final = self.run_session("open-critic", params)

        asked.assert_called_once()
        boxed.assert_called_once()
        self.assertTrue(receipt["sam_streams"][-1]["file"].endswith("critic-1.npz"))
        self.assertEqual(receipt["critic"]["rounds"][0]["tracked"], len(receipt["critic"]["rounds"][0]["keyframes"]))
        self.assertLess(self.error(final, self.ring), 20)
        self.assertEqual(receipt["models"]["spark"], f"{spark.SPARK_OPEN_MODEL} via Baseten")
        self.assertIn("Built with Muse Spark Open.", (out / "index.html").read_text(encoding="utf-8"))
        self.assertTrue(pipeline.verify(out / "receipt.json", self.video).ok)
        asked.assert_called_once()


class ResumeTest(unittest.TestCase):
    """A tracking call survives a dropped connection: its id is kept on disk and waited on again."""

    def fake_modal(self, data: bytes, failures: int = 0) -> tuple[mock.Mock, mock.Mock]:
        class ConnectionError(Exception):
            pass

        call = mock.Mock(object_id="fc-123")
        call.get.side_effect = [ConnectionError("dns")] * failures + [data]
        tracker = mock.Mock()
        tracker.track.spawn.return_value = call
        modal = mock.Mock()
        modal.Cls.from_name.return_value.return_value = tracker
        modal.FunctionCall.from_id.return_value = call
        modal.exception.ConnectionError = ConnectionError
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(sam_open, "_modal", return_value=modal).start()
        mock.patch.object(sam_open.time, "sleep").start()
        return modal, call

    def test_first_run_spawns_and_forgets_the_call_once_done(self) -> None:
        modal, call = self.fake_modal(b"masks")
        with tempfile.TemporaryDirectory() as folder:
            proxy = Path(folder) / "proxy.mp4"
            proxy.write_bytes(b"video")
            call_file = Path(folder) / "concepts.call"
            self.assertEqual(sam_open._result(proxy, [{"text": "hand"}], call_file), b"masks")
            modal.Cls.from_name.return_value.return_value.track.spawn.assert_called_once_with(b"video", [{"text": "hand"}])
            self.assertFalse(call_file.exists())

    def test_a_dropped_connection_is_retried_and_a_saved_call_is_resumed(self) -> None:
        modal, call = self.fake_modal(b"masks", failures=2)
        with tempfile.TemporaryDirectory() as folder:
            proxy = Path(folder) / "proxy.mp4"
            proxy.write_bytes(b"video")
            call_file = Path(folder) / "concepts.call"
            call_file.write_text("fc-123", encoding="utf-8")
            self.assertEqual(sam_open._result(proxy, [{"text": "hand"}], call_file), b"masks")
            modal.FunctionCall.from_id.assert_called_once_with("fc-123")
            modal.Cls.from_name.assert_not_called()
            self.assertEqual(call.get.call_count, 3)
            self.assertFalse(call_file.exists())


if __name__ == "__main__":
    unittest.main()
