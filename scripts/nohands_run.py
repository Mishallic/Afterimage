"""Run the whole NoHands pipeline on one session video.

    python scripts/nohands_run.py data/session.mp4 --corners 0.12,0.08 0.88,0.07 0.9,0.95 0.1,0.96 --aspect 1.5
    python scripts/nohands_run.py data/synthetic.mp4 --masks motion --fps 2 --width 640

Corners are fractions of the camera frame (top-left, top-right, bottom-right, bottom-left).
Without --corners, Spark's scan finds them; scripts/first_frame.py helps read them off by hand. Outputs land in out/<video name>/: timelapse.mp4,
raw.mp4, compare.mp4, reel.mp4, reel-landscape.mp4, stats.json, receipt.json and index.html.
Check any output against its footage with scripts/nohands_verify.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nohands.config import ROOT
from nohands.pipeline import Params, run


def parse_args() -> tuple[Path, Params, str, bool]:
    defaults = Params()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--masks", choices=["sam", "open", "motion", "none"], default=defaults.masks,
                        help="sam: the Meta Model API; open: SAM 3.1 weights on Modal (see scripts/sam_modal.py)")
    parser.add_argument("--concept", action="append", dest="concepts",
                        help="SAM noun phrase, repeatable (default: person, hand, paintbrush)")
    parser.add_argument("--no-scan", action="store_true",
                        help="skip Spark's scan (it adds things to track and finds the canvas corners)")
    parser.add_argument("--fps", type=float, default=defaults.fps, help="frames sampled per second of footage")
    parser.add_argument("--width", type=int, default=defaults.width, help="working width")
    parser.add_argument("--sam-width", type=int, default=defaults.sam_width, help="width of the copy sent to SAM")
    parser.add_argument("--seconds", type=float, help="only process the first N seconds")
    parser.add_argument("--start", type=float, default=0.0, help="skip the first N seconds (camera handling)")
    parser.add_argument("--corners", nargs=4, metavar="X,Y", help="canvas corners as fractions of the frame")
    parser.add_argument("--aspect", type=float, help="canvas height / width, e.g. 1.5 for 40x60 portrait")
    parser.add_argument("--dilate", type=int, default=defaults.dilate, help="pixels to grow masks by")
    parser.add_argument("--persist-seconds", type=float, default=defaults.persist_seconds,
                        help="how long new paint must hold before it counts")
    parser.add_argument("--revert-window", type=float, default=defaults.revert_window,
                        help="seconds within which an undone change counts as transient")
    parser.add_argument("--output-fps", type=float, default=defaults.output_fps)
    parser.add_argument("--title", default="", help="name of the painting, for the match page")
    parser.add_argument("--no-reels", action="store_true", help="skip rendering the social reels")
    args = parser.parse_args()
    params = Params(
        fps=args.fps,
        width=args.width,
        sam_width=args.sam_width,
        seconds=args.seconds,
        start=args.start,
        corners=tuple(tuple(float(v) for v in point.split(",")) for point in args.corners) if args.corners else None,
        aspect=args.aspect,
        dilate=args.dilate,
        persist_seconds=args.persist_seconds,
        revert_window=args.revert_window,
        concepts=tuple(args.concepts) if args.concepts else defaults.concepts,
        masks=args.masks,
        scan=not args.no_scan,
        output_fps=args.output_fps,
    )
    return args.video, params, args.title, not args.no_reels


def main() -> None:
    video, params, title, reels = parse_args()
    started = time.perf_counter()
    out = ROOT / "out" / video.stem
    receipt = run(video, out, params, title=title, reels=reels)
    print(f"{receipt['output']['frames']} frames, receipt {receipt['output']['canvas_sha256'][:16]}…")
    print(f"done in {time.perf_counter() - started:.0f}s -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
