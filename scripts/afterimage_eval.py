"""Score a processed session on the frames where the painter stepped back, next to the baselines.

    python scripts/afterimage_eval.py data/session.mp4
    python scripts/afterimage_eval.py data/session.mp4 --holdout 600-612 --holdout 1210-1225 --static 300-360

Needs a finished SAM run in out/<video name>/ (its receipt.json and cached SAM streams), so it
makes no API calls. Held-out stretches have nobody over the canvas: found in the masks unless
given in seconds. Static stretches, given in seconds, are when the painter is in view but not
painting (mixing on the palette, say), so anything committed then is a false commit.
Writes eval/<video name>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage import media
from afterimage.config import ROOT
from afterimage.heldout import heldout_report, markdown
from afterimage.pipeline import Params, sam_occluders


def seconds(text: str) -> tuple[float, float]:
    start, stop = (float(value) for value in text.split("-"))
    return start, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--holdout", action="append", type=seconds, default=[], metavar="START-STOP",
                        help="a stretch with nobody over the canvas, in seconds (repeatable; default: from the masks)")
    parser.add_argument("--static", action="append", type=seconds, default=[], metavar="START-STOP",
                        help="a stretch with no painting, in seconds (repeatable)")
    parser.add_argument("--min-seconds", type=float, default=3.0, help="shortest clear stretch to use")
    parser.add_argument("--run", type=Path, default=None,
                        help="the finished run's folder (default: out/<video name>; a run made with --out lives elsewhere)")
    args = parser.parse_args()

    folder = args.run if args.run is not None else ROOT / "out" / args.video.stem
    name = folder.name
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    params = Params.from_dict(receipt["params"])
    if not params.tracked:
        raise SystemExit("The evaluation needs a SAM run (--masks sam or open): it reuses that run's masks.")
    missing = [stream["file"] for stream in receipt["sam_streams"] if not (folder / stream["file"]).exists()]
    if missing:
        raise SystemExit(f"Cached SAM streams are missing ({', '.join(missing)}); re-run the session first.")

    info = media.probe(args.video)
    boxes = [stream for stream in receipt["sam_streams"] if stream["concept"].startswith("critic")]
    masks = sam_occluders(args.video, params, info, folder, box_streams=boxes)[0]

    def frames(stretch: tuple[float, float]) -> tuple[int, int]:
        return round(stretch[0] * params.fps), round(stretch[1] * params.fps)

    report = heldout_report(
        args.video, params, masks,
        windows=[frames(stretch) for stretch in args.holdout] or None,
        static=[frames(stretch) for stretch in args.static],
        min_seconds=args.min_seconds,
    )
    report["session"] = {"video": args.video.name, "sha256": receipt["source"]["sha256"],
                         "minutes": round(info.duration / 60, 1)}
    report["meta_calls"] = receipt.get("meta_calls")

    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = markdown(report, f"{args.video.name} ({name})" if name != args.video.stem else args.video.name)
    calls = report["meta_calls"]
    if calls:
        cost = sum(model["est_cost_usd"] for model in calls["by_model"].values())
        text += f"\nMeta calls for this run: {calls['calls']}, about ${cost:.2f} (see the run's ledger.jsonl).\n"
    (out / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
