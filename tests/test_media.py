"""Videos written from RGB frames should show the same colours in any player."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from afterimage import media
from afterimage.config import find_tool

COLOURS = [(220, 40, 40), (40, 180, 60), (40, 70, 210), (235, 150, 60), (238, 236, 230), (30, 30, 34)]


def swatches() -> np.ndarray:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    for index, colour in enumerate(COLOURS):
        frame[:, index * 16:(index + 1) * 16] = colour
    return frame


class ColourTest(unittest.TestCase):
    def test_written_colours_are_tagged_and_read_back_by_a_player(self) -> None:
        frame = swatches()
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "swatches.mp4"
            with media.FrameWriter(video, (96, 64), fps=2) as writer:
                writer.write(frame)
                writer.write(frame)
            tags = subprocess.run(
                [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=color_space,color_primaries,color_transfer,color_range", "-of", "csv=p=0", str(video)],
                capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(tags.split(","), ["tv", "bt709", "bt709", "bt709"])
            ours = next(iter(media.read_frames(video, fps=2, size=(96, 64))))
            # What a player does with HD video: decode as BT.709, whatever ffmpeg would assume.
            player = np.frombuffer(subprocess.run(
                [find_tool("ffmpeg"), "-v", "error", "-i", str(video), "-frames:v", "1", "-vf",
                 "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24", "-f", "rawvideo", "-"],
                capture_output=True, check=True).stdout, np.uint8).reshape(64, 96, 3)
        for index, colour in enumerate(COLOURS):
            middle = (slice(16, 48), slice(index * 16 + 4, index * 16 + 12))
            for decoded in (ours, player):
                error = np.abs(decoded[middle].astype(int) - np.array(colour)).max()
                self.assertLessEqual(error, 4, f"{colour} came back as {decoded[middle].reshape(-1, 3)[0]}")


if __name__ == "__main__":
    unittest.main()
