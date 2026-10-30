"""Ask Spark one small structured question about one frame, to check the provider path end to end.

    python scripts/spark_probe.py data/session.mp4 [--second 5]

Uses whichever provider .env selects (SPARK_PROVIDER=meta or baseten), prints the answer and
what the ledger recorded (time, usage), and costs about a cent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterimage import ledger, media, spark
from afterimage.config import ROOT, spark_provider

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["things", "painting_surface_visible", "one_line"],
    "properties": {
        "things": {"type": "array", "items": {"type": "string"}},
        "painting_surface_visible": {"type": "boolean"},
        "one_line": {"type": "string"},
    },
}
PROMPT = ("List the physical things you can see in this frame as short noun phrases, say whether a "
          "painting surface (canvas, paper, board) is visible, and describe the frame in one line.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--second", type=float, default=0.0)
    args = parser.parse_args()

    info = media.probe(args.video)
    size = media.output_size(info, 640)
    index = int(args.second * 2)
    frame = next(iter(media.read_indices(args.video, [index], fps=2.0, size=size))) if index else next(
        iter(media.read_frames(args.video, fps=2.0, size=size))
    )
    content = [{"type": "input_text", "text": PROMPT}, spark.image_block(frame, detail="low")]
    path = ROOT / "out" / "ledger.jsonl"
    with ledger.recording(path):
        answer = spark.ask_json(spark.client(), content, SCHEMA, "frame_probe")
    entry = ledger.entries(path)[-1]
    print(f"provider: {spark_provider()} · model: {entry['model']} · {entry['ms']} ms · usage: {entry['response'].get('usage')}")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
