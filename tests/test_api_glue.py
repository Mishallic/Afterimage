"""The parts of the Spark, SAM and Voice glue that can be checked without calling the API."""

from __future__ import annotations

import unittest

from nohands import sam, spark, voice


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


if __name__ == "__main__":
    unittest.main()
