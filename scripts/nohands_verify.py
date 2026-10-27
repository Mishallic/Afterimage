"""Check a NoHands output against its footage: re-derive every frame and compare with the receipt.

    python scripts/nohands_verify.py out/session/receipt.json --video data/session.mp4

Needs the raw footage and the SAM 3.1 mask streams cached next to the receipt; no API key.
Exit code 0 means every output frame was re-derived from that footage, bit for bit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nohands.pipeline import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--video", type=Path, required=True, help="the raw session footage")
    args = parser.parse_args()
    verdict = verify(args.receipt, args.video)
    for name, ok, detail in verdict.checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    print("VERIFIED: re-derived from the footage, no generative step." if verdict.ok else "NOT VERIFIED")
    sys.exit(0 if verdict.ok else 1)


if __name__ == "__main__":
    main()
