"""Tell the painter apart from what's painted.

SAM finds every hand and person, painted ones included. Real ones reach in from outside the
canvas, move, and belong to a body. Painted ones sit entirely on the canvas, stay put (they
only grow as they're painted) and outlast any pose. Only the real ones are occluders: treat
a painted hand as one and it can never appear in the timelapse.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .masks import Blob

Objects = dict[int, dict[str, Blob]]
"""Frame index -> {object key -> blob}, in camera-view pixels."""

MAX_OUTSIDE = 0.02
"""A painted object never reaches past the canvas edge."""
MIN_STILLNESS = 0.85
"""Median overlap between an object's consecutive appearances."""
MIN_SECONDS = 20.0
"""Painted things stay for good; a resting hand rarely holds perfectly still this long."""
MAX_ATTACHED = 0.6
"""A real hand is nearly always joined to an arm or body reaching in from outside."""
BODY_OUTSIDE = 0.2
"""Objects with this much of their area off the canvas count as the painter's body."""


@dataclass
class Track:
    frames: list[int] = field(default_factory=list)
    outside: list[float] = field(default_factory=list)
    stillness: list[float] = field(default_factory=list)
    attached: int = 0


def tracks(objects: Objects, canvas: np.ndarray) -> dict[str, Track]:
    found: dict[str, Track] = defaultdict(Track)
    previous: dict[str, Blob] = {}
    for frame in sorted(objects):
        blobs = objects[frame]
        outside = {key: blob.share_outside(canvas) for key, blob in blobs.items()}
        bodies = [key for key, share in outside.items() if share >= BODY_OUTSIDE]
        for key, blob in blobs.items():
            track = found[key]
            track.frames.append(frame)
            track.outside.append(outside[key])
            if key in previous:
                track.stillness.append(previous[key].iou(blob))
            grown = blob.grown(3)
            if any(body != key and grown.overlap(blobs[body]) for body in bodies):
                track.attached += 1
            previous[key] = blob
    return dict(found)


def painted(objects: Objects, canvas: np.ndarray, fps: float) -> set[str]:
    """Keys of objects that are part of the painting rather than the painter."""
    keys = set()
    for key, track in tracks(objects, canvas).items():
        lasting = (track.frames[-1] - track.frames[0] + 1) / fps >= MIN_SECONDS
        inside = max(track.outside) <= MAX_OUTSIDE
        still = bool(track.stillness) and float(np.median(track.stillness)) >= MIN_STILLNESS
        free = track.attached / len(track.frames) <= MAX_ATTACHED
        if lasting and inside and still and free:
            keys.add(key)
    return keys


class Occluders:
    """Per frame, the union of every object that is the painter or their tools, built when asked for.

    Behaves like {frame: bool mask}. An 84-minute session has five thousand frames, and a
    full-frame mask for each would be gigabytes; a blob is pasted into a fresh mask on demand.
    """

    def __init__(self, objects: Objects, painted_keys: set[str], shape: tuple[int, int]) -> None:
        self._objects = objects
        self._painted = painted_keys
        self.shape = shape

    def get(self, frame: int, default: np.ndarray | None = None) -> np.ndarray | None:
        if frame not in self._objects:
            return default
        return self[frame]

    def __getitem__(self, frame: int) -> np.ndarray:
        target = np.zeros(self.shape, dtype=bool)
        for key, blob in self._objects[frame].items():
            if key not in self._painted:
                blob.paste_into(target)
        return target

    def __contains__(self, frame: object) -> bool:
        return frame in self._objects

    def __iter__(self):
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def items(self):
        for frame in self._objects:
            yield frame, self[frame]


def occluders(objects: Objects, painted_keys: set[str], shape: tuple[int, int]) -> Occluders:
    """Per frame, the union of every object that is the painter or their tools."""
    return Occluders(objects, painted_keys, shape)
