"""Cameo mode: put the self-painting canvas into a clip of you sitting next to it.

    python scripts/cameo_run.py data/sitting.mp4 out/session/timelapse.mp4 --corners 0.55,0.2 0.8,0.22 0.79,0.7 0.56,0.68
    python scripts/cameo_run.py data/sitting.mp4 out/session/timelapse.mp4 --corners ... --start 2 --end 44 --track open --reveal

Film the clip with the canvas in view and sit beside it. Read the corners off with
scripts/first_frame.py. With --track, SAM 3.1 (open: the weights on Modal; sam: the Meta API)
masks the person in the clip, so where you pass in front of the canvas you stay in front of the
painting. --soften fades each stroke in over that many session seconds (a running average of real
frames). Writes out/<timelapse folder>/cameo.mp4, with the clip's own sound.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afterimage import maskfile, media
from afterimage.cameo import mask_lookup, render_cameo
from afterimage.config import ROOT, find_tool

MASK_FPS = 15.0
"""People beside a canvas move slowly; each cameo frame takes the nearest mask in time."""
MASK_WIDTH = 640
"""A 960-px, 30 fps copy of a 49 s clip ran an A100-40GB out of memory (wall stickers count as people too)."""


def person_masks(clip: Path, mode: str) -> tuple[dict, tuple[int, int]]:
    """SAM 3.1's person masks for the whole clip at MASK_FPS, cached under out/cameo/<clip>/."""
    info = media.probe(clip)
    size = media.output_size(info, MASK_WIDTH)
    work = ROOT / "out" / "cameo" / clip.stem
    proxy = work / f"proxy-{MASK_FPS:g}fps-{size[0]}.mp4"
    if not proxy.exists():
        media.make_proxy(clip, proxy, fps=MASK_FPS, size=size)
    frames = round(media.probe(proxy).duration * MASK_FPS)
    if mode == "open":
        from afterimage import sam_open

        return sam_open.track_concepts(proxy, ["person"], work / f"person-{proxy.stem}.npz", frames=frames), size
    from afterimage import sam

    async def tracked() -> dict:
        api = sam.client()
        file_id = await sam.upload_cached(api, proxy, work / "file_id.txt")
        result = await sam.segment_concept(api, file_id, "person", work / f"person-{proxy.stem}.events.ndjson", frames=frames)
        return sam.frame_objects(result, prefix="person:")

    cache = work / f"person-{proxy.stem}-api.npz"
    if cache.exists():
        return maskfile.load(cache)[0], size
    objects = asyncio.run(tracked())
    maskfile.save(cache, objects, {"model": "sam-3.1"})
    return objects, size


def softened(timelapse: Path, frames: int) -> Path:
    """The timelapse with each frame the average of the last `frames`: strokes fade in instead of popping."""
    out = timelapse.with_name(f"{timelapse.stem}-soft{frames}.mp4")
    if not out.exists():
        weights = " ".join(["1"] * frames)
        subprocess.run([
            find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(timelapse),
            "-vf", f"tmix=frames={frames}:weights='{weights}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", str(out),
        ], check=True)
    return out


def with_sound(video: Path, clip: Path, start: float, seconds: float, out: Path) -> Path:
    """The cameo with the clip's own sound for the same stretch (the room, not music)."""
    subprocess.run([
        find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video),
        "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(clip),
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
        "-movflags", "+faststart", str(out),
    ], check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path, help="you sitting next to the canvas")
    parser.add_argument("timelapse", type=Path, help="timelapse.mp4 from afterimage_run.py")
    parser.add_argument("--corners", nargs=4, metavar="X,Y", required=True,
                        help="canvas corners in the clip, as fractions of the frame")
    parser.add_argument("--start", type=float, default=0.0, help="second of the clip to start from")
    parser.add_argument("--end", type=float, help="second of the clip to stop at (default: --seconds after --start)")
    parser.add_argument("--seconds", type=float, default=8.0, help="length when --end isn't given")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds to hold the finished painting")
    parser.add_argument("--reveal", action="store_true", help="fade onto the real canvas in the clip over the hold")
    parser.add_argument("--track", choices=["none", "open", "sam"], default="none",
                        help="mask the person with SAM 3.1 so they stay in front of the painting")
    parser.add_argument("--soften", type=int, default=0, help="session seconds each stroke fades in over")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--out", type=Path, help="output file (default: cameo.mp4 next to the timelapse)")
    args = parser.parse_args()

    corners = np.array([[float(v) for v in point.split(",")] for point in args.corners])
    seconds = (args.end - args.start) if args.end is not None else args.seconds
    occluder = None
    if args.track != "none":
        objects, mask_size = person_masks(args.clip, args.track)
        occluder = mask_lookup(objects, mask_size, mask_fps=MASK_FPS, fps=30.0, start=args.start, min_share=0.015)
    timelapse = softened(args.timelapse, args.soften) if args.soften > 1 else args.timelapse
    out = args.out or args.timelapse.parent / "cameo.mp4"
    silent = out.with_name(f"{out.stem}-silent.mp4")
    render_cameo(args.clip, corners, timelapse, silent, seconds=seconds, hold=args.hold, width=args.width,
                 start=args.start, occluder=occluder, reveal=args.reveal)
    if media.probe(args.clip).has_audio:
        with_sound(silent, args.clip, args.start, seconds, out)
        silent.unlink()
    else:
        silent.replace(out)
    print(out)


if __name__ == "__main__":
    main()
