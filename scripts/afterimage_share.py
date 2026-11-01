"""Copy a run's results page into a folder small enough to host as a static site.

    python scripts/afterimage_share.py out/session2-api share/session2

The page runs entirely in the browser, so any static host works (a Hugging Face static Space,
GitHub Pages, any web server). What changes is size: the camera footage behind the pixel
receipt is re-encoded at the width the page shows it, with a keyframe every second so a tap
seeks fast, and the two videos behind the ghost slider are re-encoded with the same settings so
they stay in step. Nothing in a picture is changed beyond the re-encode.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage.config import find_tool


def encode(source: Path, out: Path, *, width: int, crf: int, audio: bool, keyframe_s: float = 1.0) -> None:
    fps = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
         "-of", "csv=p=0", str(source)], capture_output=True, text=True, check=True,
    ).stdout.strip()
    numerator, _, denominator = fps.partition("/")
    rate = float(numerator) / float(denominator or 1)
    sound = ["-c:a", "aac", "-b:a", "64k"] if audio else ["-an"]
    subprocess.run([
        find_tool("ffmpeg"), "-v", "error", "-y", "-i", str(source),
        "-vf", f"scale={width}:-2:flags=area", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-g", str(max(1, round(rate * keyframe_s))), "-pix_fmt", "yuv420p", *sound, "-movflags", "+faststart", str(out),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a finished run's folder (with index.html)")
    parser.add_argument("out", type=Path, help="the folder to write")
    parser.add_argument("--moments-width", type=int, default=540, help="width of the camera footage behind the receipt")
    parser.add_argument("--video-width", type=int, default=720, help="width of the timelapse, ghost and compare videos")
    parser.add_argument("--crf", type=int, default=30, help="x264 quality: higher is smaller")
    args = parser.parse_args()

    page = (args.run / "index.html").read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    under = re.search(r"const under = (\[.*?\]);", page, re.S)
    for item in json.loads(under.group(1)) if under else []:
        shutil.copy2(args.run / item["file"], args.out / item["file"])
    missing = []
    if (args.run / "reel.mp4").exists():
        shutil.copy2(args.run / "reel.mp4", args.out / "reel.mp4")
    else:
        missing.append("reel.mp4")
    widths = {"moments.mp4": args.moments_width, "timelapse.mp4": args.video_width, "raw.mp4": args.video_width,
              "compare.mp4": 2 * args.video_width}
    for name, width in widths.items():
        if not (args.run / name).exists():
            missing.append(name)
            continue
        encode(args.run / name, args.out / name, width=width, crf=args.crf, audio=name == "moments.mp4")
    if missing:
        print(f"not in the run, so not copied: {', '.join(missing)}")
    files = sorted(args.out.iterdir(), key=lambda path: -path.stat().st_size)
    total = sum(path.stat().st_size for path in files)
    print(f"wrote {args.out}: {len(files)} files, {total / 1e6:.0f} MB; largest {files[0].name} {files[0].stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
