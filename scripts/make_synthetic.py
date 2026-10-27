"""Write the synthetic test session to data/synthetic.mp4, so the pipeline can run offline.

    python scripts/make_synthetic.py
    python scripts/cleanplate_run.py data/synthetic.mp4 --masks motion --fps 2 --width 640
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nohands import media
from nohands.config import ROOT
from tests.synthetic import synthetic_session

SIZE = (640, 360)


def main() -> None:
    out = ROOT / "data" / "synthetic.mp4"
    with media.FrameWriter(out, SIZE, fps=2) as writer:
        for frame, _, _ in synthetic_session(frames=120, size=SIZE):
            writer.write(frame)
    print(out)


if __name__ == "__main__":
    main()
