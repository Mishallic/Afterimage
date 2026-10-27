"""Edit decision lists: the plan a director (rule-based, or Muse Spark) hands to the renderer.

Frame numbers refer to the sampled session frames (frame i of timelapse.mp4 and raw.mp4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

ClipKind = Literal["timelapse", "raw", "wipe", "split", "freeze"]
Ease = Literal["linear", "in", "out", "inout"]
CaptionStyle = Literal["caption", "title", "card"]


@dataclass
class Caption:
    text: str
    start: float = 0.0
    """Seconds after the clip starts."""
    end: float | None = None
    """Seconds after the clip starts; None means until the clip ends."""
    style: CaptionStyle = "caption"


@dataclass
class Clip:
    kind: ClipKind
    """timelapse: the clean canvas; raw: the footage; wipe: raw erased into clean;
    split: raw and clean together; freeze: one clean frame held."""
    start: int
    """First frame (for freeze: the frame that's held)."""
    end: int
    """One past the last frame; ignored for freeze."""
    duration: float
    """Seconds this clip lasts in the reel."""
    ease: Ease = "linear"
    zoom: tuple[float, float, float, float] | None = None
    """Box (x, y, w, h) as fractions of the canvas to push into by the end of the clip."""
    dim: float = 0.0
    """0 to 1: how much to darken the picture, e.g. under an end card."""
    captions: list[Caption] = field(default_factory=list)


@dataclass
class EditDecisionList:
    clips: list[Clip]
    layout: Literal["vertical", "landscape"] = "vertical"
    fps: float = 30.0

    @property
    def duration(self) -> float:
        return sum(clip.duration for clip in self.clips)

    def validate(self, frames: int) -> None:
        for number, clip in enumerate(self.clips, start=1):
            if not 0 <= clip.start < frames:
                raise ValueError(f"clip {number}: start {clip.start} is outside 0..{frames - 1}")
            if clip.kind != "freeze" and not clip.start < clip.end <= frames:
                raise ValueError(f"clip {number}: end {clip.end} must be in {clip.start + 1}..{frames}")
            if clip.duration <= 0:
                raise ValueError(f"clip {number}: duration must be positive")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> EditDecisionList:
        clips = []
        for raw in data["clips"]:
            zoom = raw.get("zoom")
            clips.append(Clip(
                **{key: value for key, value in raw.items() if key not in ("zoom", "captions")},
                zoom=tuple(zoom) if zoom else None,
                captions=[Caption(**caption) for caption in raw.get("captions", [])],
            ))
        return cls(clips=clips, layout=data.get("layout", "vertical"), fps=data.get("fps", 30.0))
