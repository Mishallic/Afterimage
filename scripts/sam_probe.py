"""See what SAM 3.1 finds in the first seconds of a video, how long it takes, and what it costs.

    python scripts/sam_probe.py data/test.mp4 --concept hand --concept paintbrush --seconds 60

Writes out/probe/<video>-.../overlay.mp4 with every mask tinted red. Results are cached, so
re-running with the same settings costs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afterimage import media, sam
from afterimage.config import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--concept", action="append", dest="concepts", required=True,
                        help="noun phrase to segment, repeatable")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    info = media.probe(args.video)
    size = media.output_size(info, args.width)
    work = ROOT / "out" / "probe" / f"{args.video.stem}-{args.fps:g}fps-{size[0]}px-{args.seconds:g}s"
    proxy = work / "proxy.mp4"
    if not proxy.exists():
        media.make_proxy(args.video, proxy, fps=args.fps, size=size, seconds=args.seconds)
    frames = list(media.read_frames(proxy, fps=args.fps, size=size))
    print(f"{len(frames)} frames at {size[0]}x{size[1]}")

    api = sam.client()
    file_id = await sam.upload_cached(api, proxy, work / "file_id.txt")
    union = [np.zeros((size[1], size[0]), dtype=bool) for _ in frames]
    for concept in args.concepts:
        started = time.perf_counter()
        result = await sam.segment_concept(
            api, file_id, concept, work / f"{sam.slug(concept)}.events.ndjson", frames=len(frames)
        )
        segmented = time.perf_counter()
        masks = sam.frame_masks(result, size)
        decoded = time.perf_counter()
        objects = {r.object_id for r in result.records if isinstance(r, sam.SegmentationMaskRecord)}
        print(
            f"{concept!r}: {len(objects)} object(s), masks on {len(masks)}/{len(frames)} frames, "
            f"SAM {segmented - started:.1f}s, decode {decoded - segmented:.1f}s, "
            f"about ${sam.estimated_cost(len(frames)):.3f}"
        )
        for diagnostic in result.diagnostics:
            print(f"  {diagnostic.severity}: {diagnostic.code} {diagnostic.message}")
        for index, mask in masks.items():
            if index < len(union):
                union[index] |= mask

    tint = np.array([255, 60, 60], dtype=np.float32)
    with media.FrameWriter(work / "overlay.mp4", size, fps=10) as writer:
        for frame, mask in zip(frames, union):
            shown = frame.copy()
            shown[mask] = (0.45 * frame[mask] + 0.55 * tint).astype(np.uint8)
            writer.write(shown)
    print(f"overlay: {writer.out}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
