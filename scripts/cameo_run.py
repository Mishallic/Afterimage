"""Cameo mode: put the self-painting canvas into a clip of you sitting next to it.

    python scripts/cameo_run.py data/sitting.mp4 out/session/timelapse.mp4 --corners 0.55,0.2 0.8,0.22 0.79,0.7 0.56,0.68

Film the clip with the same easel and light, and sit beside the canvas rather than in front
of it. Read the corners off with scripts/first_frame.py. Writes out/<timelapse folder>/cameo.mp4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afterimage.cameo import render_cameo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path, help="you sitting next to the canvas")
    parser.add_argument("timelapse", type=Path, help="timelapse.mp4 from afterimage_run.py")
    parser.add_argument("--corners", nargs=4, metavar="X,Y", required=True,
                        help="canvas corners in the clip, as fractions of the frame")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--hold", type=float, default=1.5, help="seconds to hold the finished painting")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    corners = np.array([[float(v) for v in point.split(",")] for point in args.corners])
    out = render_cameo(args.clip, corners, args.timelapse, args.timelapse.parent / "cameo.mp4",
                       seconds=args.seconds, hold=args.hold, width=args.width)
    print(out)


if __name__ == "__main__":
    main()
