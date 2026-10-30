"""Save a frame with a 10% grid over it, to read off the canvas corners for --corners.

    python scripts/first_frame.py data/session.mp4 --at 5

Writes out/<video name>/corners.png. Each grid line is 0.1 of the frame, so a corner
two and a half lines across and one line down is 0.25,0.1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage import media
from afterimage.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=float, default=2.0, help="seconds into the video")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    info = media.probe(args.video)
    size = media.output_size(info, args.width)
    frames = media.read_frames(args.video, fps=1, size=size, seconds=args.at + 1)
    frame = list(frames)[-1].copy()
    height, width = frame.shape[:2]
    for step in range(1, 10):
        x, y = round(width * step / 10), round(height * step / 10)
        colour = (255, 220, 0) if step == 5 else (0, 255, 255)
        frame[:, max(x - 1, 0):x + 1] = colour
        frame[max(y - 1, 0):y + 1, :] = colour
    out = media.write_image(frame, ROOT / "out" / args.video.stem / "corners.png")
    print(out)


if __name__ == "__main__":
    main()
