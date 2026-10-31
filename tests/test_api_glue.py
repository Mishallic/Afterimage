"""The parts of the Spark, SAM and Voice glue that can be checked without calling the API."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from afterimage import ledger, media, pipeline, sam, spark, voice

SCAN_ANSWER = {
    "occluders": [
        {"phrase": "Person", "kind": "body", "sightings": [{"second": 1.0, "box": [100, 100, 400, 900]}]},
        {"phrase": "paint  rag", "kind": "tool", "sightings": [{"second": 20.0, "box": [700, 800, 800, 900]}]},
        {"phrase": "paintbrush", "kind": "brush", "sightings": []},
        {"phrase": "cat", "kind": "animal", "sightings": [{"second": 40.0, "box": [0, 0, 100, 100]}, {"second": 41.0, "box": [1, 2]}]},
        {"phrase": "palette", "kind": "tool", "sightings": []},
        {"phrase": "cup", "kind": "tool", "sightings": []},
        {"phrase": " ", "kind": "other", "sightings": []},
    ],
    "canvas_corners": [[880, 70], [120, 80], [100, 960], [900, 950]],
}


def strict_problems(schema: dict, path: str = "$") -> list[str]:
    """Rules the API enforces for strict structured output."""
    problems = []
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        if sorted(schema.get("required", [])) != sorted(schema.get("properties", {})):
            problems.append(f"{path}: every property must be required")
        for key, child in schema.get("properties", {}).items():
            problems += strict_problems(child, f"{path}.{key}")
    if schema.get("type") == "array":
        problems += strict_problems(schema.get("items", {}), f"{path}[]")
    return problems


class SparkTest(unittest.TestCase):
    def test_schemas_are_strict(self) -> None:
        for schema in (spark.SCAN_SCHEMA, spark.CRITIC_SCHEMA, spark.CHAPTERS_SCHEMA):
            self.assertEqual(strict_problems(schema), [])

    def test_grid_boxes_become_clamped_pixel_boxes(self) -> None:
        self.assertEqual(spark.to_pixels([100, 200, 500, 600], (640, 360)), (64, 72, 320, 216))
        self.assertEqual(spark.to_pixels([-20, 0, 1003, 1000], (640, 360)), (0, 0, 640, 360))
        self.assertEqual(spark.to_pixels([500, 500, 400, 400], (640, 360)), (256, 144, 320, 180))
        left, top, right, bottom = spark.to_pixels([999, 999, 1000, 1000], (640, 360))
        self.assertTrue(right > left and bottom > top)

    def test_scan_frames_cover_the_whole_session(self) -> None:
        picks = spark.scan_picks(10_800)
        self.assertLessEqual(len(picks), spark.SCAN_FRAMES)
        self.assertEqual((picks[0], picks[-1]), (0, 10_799))
        self.assertEqual(spark.scan_picks(40), list(range(40)))
        self.assertEqual(spark._scan_frame(0.0, picks), 0)
        self.assertEqual(spark._scan_frame(999.0, picks), 10_799)

    def test_scan_answers_map_back_to_session_frames_and_pixels(self) -> None:
        scan = spark.parse_scan(SCAN_ANSWER, spark.scan_picks(960), (640, 360))
        self.assertEqual([o.phrase for o in scan.occluders], ["person", "paint rag", "paintbrush", "cat", "palette", "cup"])
        cat = scan.occluders[3]
        self.assertEqual(cat.sightings, [spark.Sighting(spark.scan_picks(960)[80], (0, 0, 64, 36))])
        self.assertAlmostEqual(scan.canvas_corners[0][0], 0.88)

    def test_points_and_boxes_are_read_in_whatever_shape_the_model_chose(self) -> None:
        # Spark Open on session 1 answered the corners as {"x", "y"} objects, and once as four numbers each.
        answer = {
            "occluders": [{"phrase": "cup", "kind": "tool", "sightings": [
                {"second": "3", "box": {"x_min": 100, "y_min": 100, "x_max": 400, "y_max": 900}},
                {"second": 4, "box": [1, 2]},
            ]}],
            "canvas_corners": [{"x": 880, "y": 70}, {"x": 120, "y": 80}, {"x": 100, "y": 960}, {"x": 900, "y": 950}],
        }
        scan = spark.parse_scan(answer, spark.scan_picks(960), (640, 360))
        self.assertEqual([s.box for s in scan.occluders[0].sightings], [spark.to_pixels([100, 100, 400, 900], (640, 360))])
        self.assertAlmostEqual(scan.canvas_corners[0][0], 0.88)
        defects = spark.parse_critic(
            {"defects": [{"image": "1", "box": {"left": 0, "top": 0, "right": 500, "bottom": 500}, "kind": "glare", "confidence": "0.9"},
                         {"image": 7, "box": [0, 0, 10, 10], "kind": "glare", "confidence": 0.9}]},
            2, (640, 360),
        )
        self.assertEqual([(d.keyframe, d.box, d.confidence) for d in defects], [(1, (0, 0, 320, 180), 0.9)])

    def test_only_things_beyond_painter_hands_and_brushes_get_a_sam_pass(self) -> None:
        scan = spark.parse_scan(SCAN_ANSWER, spark.scan_picks(960), (640, 360))
        self.assertEqual(spark.extra_concepts(scan, ("person", "hand", "paintbrush")), ["paint rag", "cat", "palette"])
        self.assertEqual(spark.extra_concepts(scan, ("person", "paintrag")), ["cat", "palette", "cup"])


class SparkScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)
        self.info = media.VideoInfo(1280, 720, 480.0, 30.0, False)

    def scan(self, params: pipeline.Params, asked: mock.Mock) -> tuple[pipeline.Params, dict]:
        with (
            mock.patch.object(pipeline, "_proxy", return_value=(self.out / "proxy.mp4", 960)),
            mock.patch.object(spark, "make_scan_video"),
            mock.patch.object(spark, "client"),
            mock.patch.object(spark, "scan_answer", asked),
        ):
            return pipeline.spark_scan(Path("session.mp4"), params, self.info, self.out)

    def test_spark_adds_concepts_and_corners_once_then_the_answer_is_reused(self) -> None:
        asked = mock.Mock(return_value=SCAN_ANSWER)
        params, found = self.scan(pipeline.Params(), asked)
        again, _ = self.scan(pipeline.Params(), asked)
        asked.assert_called_once()
        self.assertEqual(params, again)
        self.assertEqual(params.concepts, ("person", "hand", "paintbrush", "paint rag", "cat", "palette"))
        self.assertEqual(params.corners, ((0.12, 0.08), (0.88, 0.07), (0.9, 0.95), (0.1, 0.96)))
        self.assertEqual(found["canvas_corners"], params.corners)
        self.assertEqual(found["sha256"], ledger.file_sha256(self.out / "spark" / "scan.json"))
        self.assertEqual(pipeline.Params.from_dict(params.to_dict()), params)

    def test_corners_given_by_hand_win(self) -> None:
        mine = replace(pipeline.Params(), corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        params, found = self.scan(mine, mock.Mock(return_value=SCAN_ANSWER))
        self.assertEqual(params.corners, mine.corners)
        self.assertIsNone(found["canvas_corners"])


class BasetenSparkTest(unittest.TestCase):
    """Spark Open through Baseten: chat completions, a forced tool call, frames instead of video."""

    def setUp(self) -> None:
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, {"SPARK_PROVIDER": "baseten", "BASETEN_API_KEY": "bt-secret-123"}).start()
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.ledger = self.folder / "ledger.jsonl"

    @staticmethod
    def response(name: str, arguments: str | None = None, content: str | None = None) -> SimpleNamespace:
        calls = [SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))] if arguments else None
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(id="chat_1", usage=None, choices=[SimpleNamespace(message=message)])

    def test_the_schema_goes_in_as_a_forced_tool_and_images_as_chat_parts(self) -> None:
        api = mock.Mock()
        api.chat.completions.create.return_value = self.response("occluder_scan", json.dumps(SCAN_ANSWER))
        frame = np.zeros((36, 64, 3), np.uint8)
        content = [{"type": "input_text", "text": "look"}, spark.image_block(frame, detail="low")]
        with ledger.recording(self.ledger):
            answer = spark.ask_json(api, content, spark.SCAN_SCHEMA, "occluder_scan")
        self.assertEqual(answer, SCAN_ANSWER)
        kwargs = api.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], spark.SPARK_OPEN_MODEL)
        self.assertEqual(kwargs["tools"][0]["function"]["parameters"], spark.SCAN_SCHEMA)
        self.assertEqual(kwargs["tool_choice"], {"type": "function", "function": {"name": "occluder_scan"}})
        parts = kwargs["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "look"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["detail"], "low")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        entry = ledger.entries(self.ledger)[0]
        self.assertEqual((entry["endpoint"], entry["model"]), ("chat.completions", spark.SPARK_OPEN_MODEL))
        self.assertNotIn("base64", json.dumps(entry))
        self.assertNotIn("bt-secret-123", self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(spark.model_name(), f"{spark.SPARK_OPEN_MODEL} via Baseten")

    def test_json_in_the_text_is_accepted_when_the_tool_call_is_skipped(self) -> None:
        api = mock.Mock()
        api.chat.completions.create.return_value = self.response(
            "reconstruction_critic", content="```json\n" + json.dumps({"defects": []}) + "\n```"
        )
        with ledger.recording(self.ledger):
            answer = spark.ask_json(api, [{"type": "input_text", "text": "x"}], spark.CRITIC_SCHEMA, "reconstruction_critic")
        self.assertEqual(answer, {"defects": []})

    def test_the_scan_video_is_sent_as_frames(self) -> None:
        api = mock.Mock()
        api.chat.completions.create.return_value = self.response("occluder_scan", json.dumps(SCAN_ANSWER))
        frames = [np.zeros((36, 64, 3), np.uint8)] * 20
        with (
            mock.patch.object(media, "probe", return_value=media.VideoInfo(64, 36, 10.0, 2.0, False)),
            mock.patch.object(media, "read_frames", return_value=iter(frames)),
            ledger.recording(self.ledger),
        ):
            spark.scan_answer(api, Path("scan.mp4"))
        parts = api.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual([part["type"] for part in parts], ["text"] + ["image_url"] * spark.BASETEN_MAX_IMAGES)
        self.assertEqual(parts[1]["image_url"]["detail"], "high")
        self.assertIn("10 seconds long", parts[0]["text"])
        self.assertIn("image 0: 0 s", parts[0]["text"])
        self.assertIn(f"image {spark.BASETEN_MAX_IMAGES - 1}: 9.5 s", parts[0]["text"])


class SamPromptTest(unittest.TestCase):
    def test_box_prompt_text(self) -> None:
        text = sam.box_prompt((640, 360), [(12, (10, 20, 110, 220)), (40, (300, 50, 340, 90))])
        self.assertIn("<12f>1<|box;", text)
        self.assertIn("<40f>2<|box;", text)
        self.assertIn("w=640;h=360", text)


class VoiceTest(unittest.TestCase):
    def test_multipart_body(self) -> None:
        body, content_type = voice.multipart([
            ("request", None, "application/json", b'{"model": "m"}'),
            ("audio", "chunk_000.wav", "audio/wav", b"RIFF...."),
        ])
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(body.startswith(f"--{boundary}\r\n".encode()))
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))
        self.assertIn(b'name="request"', body)
        self.assertIn(b'name="audio"; filename="chunk_000.wav"', body)
        self.assertIn(b"Content-Type: audio/wav", body)



class EmptyAnswerTest(unittest.TestCase):
    """Spark spending its whole budget on reasoning is asked again with more room and less effort."""

    def test_an_empty_answer_is_asked_again_with_more_room(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from afterimage import spark

        calls = []

        def create(**kwargs):
            calls.append((kwargs["reasoning"]["effort"], kwargs["max_output_tokens"]))
            text = "" if len(calls) == 1 else '{"defects": []}'
            return SimpleNamespace(id=f"resp_{len(calls)}", usage=None, output_text=text)

        api = SimpleNamespace(responses=SimpleNamespace(create=create))
        with mock.patch.object(spark, "spark_provider", return_value="meta"):
            answer = spark.ask_json(api, [{"type": "input_text", "text": "?"}], spark.CRITIC_SCHEMA, "reconstruction_critic", effort="high")
        self.assertEqual(answer, {"defects": []})
        self.assertEqual(calls, [("high", 8000), ("medium", 16000)])


if __name__ == "__main__":
    unittest.main()
