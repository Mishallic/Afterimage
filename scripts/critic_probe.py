"""Measure the critic on planted leftovers, using a finished SAM run.

    python scripts/critic_probe.py data/session.mp4 [--run out/session] [--plants 20] [--seed 1]

Takes (camera | reconstruction) pairs the way the critic sees them, pastes hand crops cut from
the raw footage with the SAM masks into half of the reconstructions at random places, asks
Spark where the leftovers are, and scores it: a plant counts as found when a reported box
overlaps it (IoU >= 0.3); a reported box that overlaps no plant is a false flag. Clean images
measure false flags on their own. Every reported object then gets the pipeline's second look
(a crop of the reconstruction alone), and both are scored: what Spark reported, and what the
pipeline would act on. Uses the run's cached clean pass. Writes eval/critic-<run>.md. Costs
one Spark call per 2 images (--batch) plus one per reported object, logged to <run>/ledger-probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage import critic, ledger, media, pipeline, spark
from afterimage.config import ROOT
from afterimage.pipeline import Geometry, Params, _proxy, _sam_work, clean, occluder_masks, sam_objects

BATCH = 2
"""Images per Spark call: the critic asks about all of a round's keyframes at once, and both
sessions' rounds had two."""
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
    parser.add_argument("--batch", type=int, default=BATCH, help="images per Spark call (default: %(default)s)")
    parser.add_argument("--run", type=Path, default=None, help="the finished run's folder (default: out/<video name>)")
    args = parser.parse_args()

    folder = args.run if args.run is not None else ROOT / "out" / args.video.stem
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    params = Params.from_dict(receipt["params"])
    if params.masks != "sam":
        raise SystemExit("The probe needs a SAM run: it cuts the plants out with its masks.")
    info = media.probe(args.video)
    objects, _, proxy_size = sam_objects(args.video, params, info, folder)
    masks, _ = occluder_masks(objects, params, proxy_size)
    geometry = Geometry.of(info, params)
    caches = sorted(folder.glob("clean-*.npz"), key=lambda path: path.stat().st_mtime)
    if caches:
        cleaned = pipeline.vetoed_again(pipeline._load_cleaned(caches[-1]), params)
    else:
        cleaned = clean(args.video, params, geometry, masks, raw_out=None)
    proxy, frames = _proxy(args.video, params, proxy_size, _sam_work(folder, params, proxy_size))

    rng = np.random.default_rng(args.seed)
    tools = ("hand", "paintbrush")
    covered = [index for index, share in enumerate(cleaned.occlusion) if MIN_COVER <= share <= MAX_COVER
               and any(key.split(":")[0] in tools for key in objects.get(index, {}))]
    if len(covered) < 2:
        raise SystemExit("Too few frames with a hand or brush partly over the canvas to cut plants from.")
    shown = sorted(rng.choice(frames, size=min(2 * args.plants, frames), replace=False).tolist())
    # A leftover is a fragment: a hand or a brush, cut out with its own SAM masks (the whole
    # painter is too big to pass for one). Spare donors, since some land too big or too small.
    donors = rng.choice(covered, size=min(3 * args.plants, len(covered)), replace=False).tolist()
    wanted = sorted(set(shown + donors))
    pairs = dict(zip(wanted, critic.pairs(
        proxy, wanted, cleaned.start, cleaned.kept, len(cleaned.occlusion),
        fps=params.fps, corners=params.corners, size=geometry.size, proxy_size=proxy_size, moves=params.moves,
    )))

    def tool_masks(frame: int):
        """Each hand or brush SAM tracked in `frame`, one at a time, in the canvas view."""
        for key, blob in objects.get(frame, {}).items():
            if key.split(":")[0] in tools:
                found = np.zeros((proxy_size[1], proxy_size[0]), dtype=bool)
                blob.paste_into(found)
                view = geometry.view_mask(found, frame)
                if view is not None and view.any():
                    yield view

    cut_outs = ((donor, mask) for donor in donors for mask in tool_masks(donor))
    images: list[tuple[int, np.ndarray, np.ndarray, tuple[int, int, int, int] | None]] = []
    for position, index in enumerate(shown):
        raw, rebuilt = pairs[index]
        rebuilt = rebuilt.copy()
        box = None
        if position % 2 == 0:
            for donor, mask in cut_outs:
                box = plant(rebuilt, pairs[donor][0], mask, rng)
                if box is not None:
                    break
        images.append((index, raw, rebuilt, box))

    scores = {stage: {"found": 0, "flags": 0, "misses": []} for stage in ("reported", "acted on")}
    planted = 0
    api = spark.client()
    with ledger.recording(folder / "ledger-probe.jsonl"):
        for start in range(0, len(images), args.batch):
            batch = images[start:start + args.batch]
            answer = spark.critic_answer(api, [(raw, rebuilt) for _, raw, rebuilt, _ in batch])
            defects = [d for d in spark.parse_critic(answer, len(batch), geometry.size) if d.confidence >= critic.MIN_CONFIDENCE]
            for number, (index, raw, rebuilt, box) in enumerate(batch):
                reported = [d for d in defects if d.keyframe == number]
                # What the pipeline acts on: an object only if a crop of the reconstruction alone
                # confirms it; glare and shadows are held; smears and the rest are only recorded.
                acted = [d for d in reported if (d.kind in critic.BRIGHTENING) or (d.is_object and spark.parse_check(
                    spark.check_answer(api, critic.check_crop(d, (raw, rebuilt)), d.kind)))]
                planted += box is not None
                for stage, boxes in (("reported", [d.box for d in reported]), ("acted on", [d.box for d in acted])):
                    score = scores[stage]
                    if box is not None:
                        if any(iou(box, other) >= MATCH_IOU for other in boxes):
                            score["found"] += 1
                        else:
                            score["misses"].append(index)
                        score["flags"] += sum(iou(box, other) < MATCH_IOU for other in boxes)
                    else:
                        score["flags"] += len(boxes)

    calls = ledger.summary(folder / "ledger-probe.jsonl")
    cost = sum(model["est_cost_usd"] for model in calls["by_model"].values())
    lines = [
        f"# Critic probe: {args.video.name} ({folder.name})", "",
        f"{planted} planted leftovers across {len(images)} images ({len(images) - planted} clean), seed {args.seed}, "
        f"{args.batch} images per Spark call.",
        "Reported: what Spark boxed. Acted on: what the pipeline would change, i.e. objects a second look at a crop "
        "of the reconstruction alone confirms, plus glare and shadows (smears are only recorded). Plants are hand and "
        "brush cut-outs from the footage, pasted into half the reconstructions.", "",
        "| Stage | Found | Recall | False flags | Flags per image |", "|---|---|---|---|---|",
    ]
    for stage, score in scores.items():
        recall = f"{score['found'] / planted:.0%}" if planted else "—"
        lines.append(f"| {stage} | {score['found']} / {planted} | {recall} | {score['flags']} | {score['flags'] / len(images):.2f} |")
    lines += [
        "",
        f"Missed after the second look, at frames: {', '.join(map(str, scores['acted on']['misses'])) or 'none'}.",
        f"Spark calls: {calls['calls']}, about ${cost:.2f}.",
    ]
    out = ROOT / "eval" / f"critic-{folder.name}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
