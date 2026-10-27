"""Every Meta call lands in the ledger, successful or not, without media bytes or the API key."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nohands import ledger, sam, spark, voice
from nohands.config import SAM_MODEL, SPARK_MODEL


class _Event:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, mode: str) -> dict:
        return self.data


async def _stream(events: list[dict]):
    for event in events:
        yield _Event(event)


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.path = self.folder / "ledger.jsonl"

    def test_calls_are_appended_with_their_outcome(self) -> None:
        with ledger.recording(self.path):
            with ledger.call("responses.create", "scan", model="m", prompt="find the painter") as logged:
                logged["id"] = "resp_1"
            with self.assertRaises(ValueError), ledger.call("responses.create", "critic", model="m"):
                raise ValueError("bad schema")
        ok, failed = ledger.entries(self.path)
        self.assertEqual((ok["status"], ok["response"]["id"], ok["request"]["prompt"]), ("ok", "resp_1", "find the painter"))
        self.assertIsInstance(ok["ms"], int)
        self.assertEqual(failed["status"], "error")
        self.assertIn("bad schema", failed["error"])

    def test_recording_only_applies_inside_its_block(self) -> None:
        inner = self.folder / "inner.jsonl"
        with ledger.recording(self.path):
            with ledger.recording(inner), ledger.call("files.create", "first"):
                pass
            with ledger.call("files.create", "second"):
                pass
        self.assertEqual([entry["purpose"] for entry in ledger.entries(inner)], ["first"])
        self.assertEqual([entry["purpose"] for entry in ledger.entries(self.path)], ["second"])

    def test_the_api_key_never_reaches_the_ledger(self) -> None:
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "sk-test-1234567890"}), ledger.recording(self.path):
            with self.assertRaises(RuntimeError), ledger.call("files.create", "upload", note="sk-test-1234567890"):
                raise RuntimeError("401: invalid key sk-test-1234567890")
        self.assertNotIn("sk-test-1234567890", self.path.read_text(encoding="utf-8"))

    def test_summary_counts_calls_errors_and_cost_per_model(self) -> None:
        with ledger.recording(self.path):
            for cost in (0.25, 0.5):
                with ledger.call("responses.create", "track", model=SAM_MODEL, est_cost_usd=cost):
                    pass
            with self.assertRaises(TimeoutError), ledger.call("responses.create", "track", model=SAM_MODEL, est_cost_usd=9.0):
                raise TimeoutError
            with ledger.call("files.create", "upload"):
                pass
        summary = ledger.summary(self.path)
        self.assertEqual(summary["calls"], 4)
        self.assertEqual(summary["sha256"], ledger.file_sha256(self.path))
        self.assertEqual(summary["by_model"][SAM_MODEL], {"calls": 3, "errors": 1, "est_cost_usd": 0.75})
        self.assertEqual(summary["by_model"]["files.create"]["calls"], 1)
        self.assertEqual(ledger.summary(self.folder / "none.jsonl")["calls"], 0)


class WiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.path = self.folder / "ledger.jsonl"

    def test_spark_requests_keep_the_prompt_and_answer_but_only_hash_images(self) -> None:
        reply = SimpleNamespace(id="resp_9", usage={"input_tokens": 812, "output_tokens": 40}, output_text='{"defects": []}')
        api = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: reply))
        content = [{"type": "input_text", "text": spark.CRITIC_PROMPT}, spark.image_block(np.zeros((16, 16, 3), np.uint8))]
        with ledger.recording(self.path):
            answer = spark.ask_json(api, content, spark.CRITIC_SCHEMA, "reconstruction_critic", effort="high")
        (entry,) = ledger.entries(self.path)
        self.assertEqual(answer, {"defects": []})
        self.assertEqual((entry["model"], entry["purpose"], entry["response"]["id"]), (SPARK_MODEL, "reconstruction_critic", "resp_9"))
        self.assertEqual(entry["request"]["prompt"], spark.CRITIC_PROMPT)
        self.assertEqual(entry["response"]["answer"], {"defects": []})
        self.assertEqual(entry["response"]["usage"]["input_tokens"], 812)
        self.assertEqual(len(entry["request"]["images"]), 1)
        self.assertNotIn("base64", self.path.read_text(encoding="utf-8"))

    def test_sam_calls_are_logged_once_and_match_the_cached_stream(self) -> None:
        events = [
            {"type": "response.created", "response": {"id": "resp_s", "usage": None}},
            {"type": "response.output_text.delta", "delta": "..."},
            {"type": "response.completed", "response": {"id": "resp_s", "usage": {"frames": 60}}},
        ]

        async def upload(**_):
            return SimpleNamespace(id="file_1")

        async def respond(**_):
            return _stream(events)

        api = SimpleNamespace(files=SimpleNamespace(create=upload), responses=SimpleNamespace(create=respond))
        proxy = self.folder / "proxy.mp4"
        proxy.write_bytes(b"not really a video")
        events_file = self.folder / "person.events.ndjson"

        async def track() -> None:
            file_id = await sam.upload_cached(api, proxy, self.folder / "file_id.txt")
            await sam.segment_concept(api, file_id, "person", events_file, frames=60)

        with ledger.recording(self.path), mock.patch.object(sam, "parse_cached", mock.AsyncMock()):
            asyncio.run(track())
            asyncio.run(track())  # all cached now: no new calls
        upload_entry, track_entry = ledger.entries(self.path)
        self.assertEqual(upload_entry["response"]["file_id"], "file_1")
        self.assertEqual(upload_entry["request"]["sha256"], ledger.file_sha256(proxy))
        self.assertEqual((track_entry["model"], track_entry["purpose"]), (SAM_MODEL, "track 'person'"))
        self.assertEqual(track_entry["response"]["id"], "resp_s")
        self.assertEqual(track_entry["response"]["usage"], {"frames": 60})
        self.assertEqual(track_entry["response"]["events"], 3)
        self.assertEqual(track_entry["response"]["events_sha256"], ledger.file_sha256(events_file))
        self.assertEqual(track_entry["request"]["est_cost_usd"], 0.012)

    def test_voice_logs_every_attempt_but_not_what_was_said(self) -> None:
        wav = self.folder / "chunk_000.wav"
        with wave.open(str(wav), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\0\0" * 16_000)
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = (
            b'{"transcript": "this blue is too loud", "turns": [{"transcript": "this blue is too loud", "startMs": 0, "endMs": 900}]}'
        )
        busy = urllib.error.HTTPError("https://api.meta.ai/v1/asr/transcribe", 429, "Too Many Requests", None, None)
        with (
            mock.patch.dict(os.environ, {"MODEL_API_KEY": "sk-test-1234567890"}),
            mock.patch("urllib.request.urlopen", side_effect=[busy, reply]),
            mock.patch.object(voice.time, "sleep"),
            ledger.recording(self.path),
        ):
            result = voice.transcribe(wav)
        self.assertEqual(result["transcript"], "this blue is too loud")
        busy_entry, ok_entry = ledger.entries(self.path)
        self.assertEqual((busy_entry["status"], busy_entry["request"]["attempt"]), ("error", 1))
        self.assertEqual((ok_entry["status"], ok_entry["request"]["attempt"]), ("ok", 2))
        self.assertEqual((ok_entry["request"]["seconds"], ok_entry["response"]["turns"]), (1.0, 1))
        self.assertNotIn("too loud", self.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
