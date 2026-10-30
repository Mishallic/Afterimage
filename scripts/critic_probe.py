"""Measure the critic on planted leftovers, using a finished SAM run.

    python scripts/critic_probe.py data/session.mp4 [--plants 20] [--seed 1]

Takes (camera | reconstruction) pairs the way the critic sees them, pastes hand crops cut from
the raw footage with the SAM masks into half of the reconstructions at random places, asks
Spark where the leftovers are, and scores it: a plant counts as found when a reported box
overlaps it (IoU >= 0.3); a reported box that overlaps no plant is a false flag. Clean images
measure false flags on their own. Writes eval/critic.md. Costs one Spark call per 6 images,
logged to out/<video name>/ledger-probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage import critic, ledger, media, spark
from afterimage.config import ROOT
from afterimage.pipeline import Geometry, Params, _proxy, _sam_work, clean, occluder_masks, sam_objects

BATCH = 6
MIN_COVER, MAX_COVER = 0.02, 0.3
MATCH_IOU = 0.3


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    shared = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union else 0.0


def plant(target: np.ndarray, donor: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    """Paste the masked part of `donor` into `target` somewhere random; the box it landed in."""
    rows, cols = np.nonzero(mask)
    if len(rows) < 200:
        return None
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    height, width = bottom - top, right - left
    if width > target.shape[1] // 3 or height > target.shape[0] // 3:
        return None
    y = int(rng.integers(0, target.shape[0] - height + 1))
    x = int(rng.integers(0, target.shape[1] - width + 1))
    part = mask[top:bottom, left:right]
    target[y:y + height, x:x + width][part] = donor[top:bottom, left:right][part]
    return x, y, x + width, y + height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--plants", type=int, default=20, help="planted images; as many clean ones are added")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    folder = ROOT / "out" / args.video.stem
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    params = Params.from_dict(receipt["params"])
    if params.masks != "sam":
        raise SystemExit("The probe needs a SAM run: it cuts the plants out with its masks.")
    info = media.probe(args.video)
    objects, _, proxy_size = sam_objects(args.video, params, info, folder)
    masks, _ = occluder_masks(objects, params, proxy_size)
    geometry = Geometry.of(info, params)
    cleaned = clean(args.video, params, geometry, masks, raw_out=None)
    proxy, frames = _proxy(args.video, params, proxy_size, _sam_work(folder, params, proxy_size))

    rng = np.random.default_rng(args.seed)
    covered = [index for index, share in enumerate(cleaned.occlusion) if MIN_COVER <= share <= MAX_COVER]
    if len(covered) < 2:
        raise SystemExit("Too few frames with the painter partly over the canvas to cut plants from.")
    shown = sorted(rng.choice(frames, size=min(2 * args.plants, frames), replace=False).tolist())
    donors = sorted(rng.choice(covered, size=min(args.plants, len(covered)), replace=False).tolist())
    wanted = sorted(set(shown + donors))
    pairs = dict(zip(wanted, critic.pairs(
        proxy, wanted, cleaned.start, cleaned.kept, len(cleaned.occlusion),
        fps=params.fps, corners=params.corners, size=geometry.size, proxy_size=proxy_size,
    )))

    images: list[tuple[int, np.ndarray, np.ndarray, tuple[int, int, int, int] | None]] = []
    for position, index in enumerate(shown):
        raw, rebuilt = pairs[index]
        rebuilt = rebuilt.copy()
        box = None
        if position % 2 == 0:
            donor = donors[(position // 2) % len(donors)]
            mask = geometry.view_mask(masks.get(donor))
            if mask is not None:
                box = plant(rebuilt, pairs[donor][0], mask, rng)
        images.append((index, raw, rebuilt, box))

    found, planted, flags, misses = 0, 0, 0, []
    api = spark.client()
    with ledger.recording(folder / "ledger-probe.jsonl"):
        for start in range(0, len(images), BATCH):
            batch = images[start:start + BATCH]
            answer = spark.critic_answer(api, [(raw, rebuilt) for _, raw, rebuilt, _ in batch])
            defects = [d for d in spark.parse_critic(answer, len(batch), geometry.size) if d.confidence >= critic.MIN_CONFIDENCE]
            for number, (index, _, _, box) in enumerate(batch):
                reported = [d.box for d in defects if d.keyframe == number]
                if box is not None:
                    planted += 1
                    if any(iou(box, other) >= MATCH_IOU for other in reported):
                        found += 1
                    else:
                        misses.append(index)
                    flags += sum(iou(box, other) < MATCH_IOU for other in reported)
                else:
                    flags += len(reported)

    calls = ledger.summary(folder / "ledger-probe.jsonl")
    cost = sum(model["est_cost_usd"] for model in calls["by_model"].values())
    lines = [
        f"# Critic probe: {args.video.name}", "",
        f"{planted} planted leftovers across {len(images)} images ({len(images) - planted} clean), seed {args.seed}.", "",
        "| Found | Recall | False flags | Flags per image |", "|---|---|---|---|",
        f"| {found} / {planted} | {found / planted:.0%} | {flags} | {flags / len(images):.2f} |" if planted else "| 0 / 0 | — | — | — |",
        "",
        f"Missed at frames: {', '.join(map(str, misses)) or 'none'}.",
        f"Spark calls: {calls['calls']}, about ${cost:.2f}.",
    ]
    out = ROOT / "eval" / "critic.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
