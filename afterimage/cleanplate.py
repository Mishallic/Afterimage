"""Rebuild the canvas behind the painter, frame by frame.

Each frame updates a running "clean" canvas from the pixels nobody is covering.
New paint is committed only once it holds steady for a few visible frames, so
moving shadows, glare and motion blur never make it into the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain

import numpy as np

from .masks import dilate, resize_nearest
from .timeline import Timeline


GAIN_STEP = 4
"""Exposure is measured on every fourth row and column: a global quantity needs no more."""
GAIN_PERCENTILE = 80.0
"""Which per-pixel ratio counts as the exposure.

A shadow only ever darkens, so when the painter's shadow (never in the masks) lies over part of
the sample, the brightest ratios are the ones with nothing in the way. The median would take a
shadow over half the canvas for a darker exposure, and brighten everything else to match.
"""
REFERENCE_AGE = 300
"""Updates a committed value must have stood before exposure is measured against it.

A shadow or a glint that sits still long enough to commit must not become the yardstick for the
frames after it leaves, or its leaving would read as a change of light.
"""
MIN_VISIBLE = 0.3
"""Below this share of the canvas uncovered, exposure isn't measured and the last gain is kept.

A sliver of margin next to the painter is in their shadow more often than not, and says nothing
about the light on the canvas.
"""
MAX_DISAGREEMENT = 0.5
"""When more of the uncovered canvas than this differs from the plate at once, the frame is not a
view of the canvas: someone walked in front of the lens, the camera was bumped, the light
switched. Painting never changes half the canvas in a second."""
BLIND_FRAMES = 30
"""How many such updates in a row are skipped before the plate accepts that the scene has changed."""
REFINE_THRESHOLD = 3.0
"""A settled pixel whose plate value has drifted this far (per channel, 0-255) from the colour on
record has its last recorded change amended to the drifted value.

The plate follows what the camera keeps showing (drift_rate), but every output is replayed from
the record, so a stroke committed from one noisy frame would keep that frame's value for good.
Neighbours committed a second apart under slightly different light then differ by up to the
change threshold, and the replay looks mottled and posterised. The amendment is to the colour of
the same stroke, not a new event: the pixel receipt and the Underneath layers are unchanged.
"""


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(values + 0.5, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class CleanPlateConfig:
    dilate_px: int = 12
    """Grow occluder masks to cover soft edges, motion blur and the brush tip."""
    change_threshold: float = 20.0
    """Per-channel difference (0-255) that counts as a real change on the canvas."""
    persist_frames: int = 3
    """Visible frames a change must hold before it's committed as paint."""
    drift_rate: float = 0.05
    """How quickly unchanged pixels follow slow lighting drift."""
    normalize_exposure: bool = True
    record: bool = False
    """Keep every commit, so a Timeline can later veto the ones that turn out to be transient.

    Commits are dated to the update their value was first seen, not the one that confirmed it,
    so replaying the Timeline shows paint as soon as it appeared, without the persistence delay.
    """


def visible_mask(occluder: np.ndarray | None, shape: tuple[int, int], dilate_px: int) -> np.ndarray:
    if occluder is None:
        return np.ones(shape, dtype=bool)
    return ~dilate(resize_nearest(occluder.astype(bool), shape), dilate_px)


def exposure_gain(
    current: np.ndarray, reference: np.ndarray, usable: np.ndarray, *, step: int = 1, percentile: float = 50.0,
) -> np.ndarray | None:
    """Per-channel gain from `reference` to `current`: phone auto-exposure and white-balance drift.

    Measured on every `step`-th row and column of the `usable` pixels; None when too few of them
    are well lit to tell. The `percentile` of the per-pixel ratios is taken as the gain: the
    median ignores the small share of pixels that changed because of new paint, and a higher one
    also ignores a shadow over part of the sample (see GAIN_PERCENTILE).
    """
    if step > 1:
        current, reference, usable = current[::step, ::step], reference[::step, ::step], usable[::step, ::step]
    sample, base = current[usable], reference[usable]
    lit = base.min(axis=1) > 20
    if lit.sum() < 500:
        return None
    return np.clip(np.percentile(sample[lit] / base[lit], percentile, axis=0), 0.6, 1.6)


SURROUND_MARGIN = 0.03
"""Share of the frame width kept clear around the canvas when measuring light on the room: the
canvas's own edge, the easel and the canvas's shadow sit there."""
MIN_SURROUND = 0.08
"""Below this share of the frame showing the room around the canvas, exposure is measured on the
canvas instead (a canvas filling the frame)."""
SURROUND_FLOOR = 80
"""Room pixels darker than this in the reference (shadowed floor, furniture) are left out: their
ratios are mostly noise. With them, and the 80th percentile, the second session's blue gain read
1.3 where the wall above the canvas had moved 1.18; lit pixels and the median read 1.18."""


def surround_gains(
    frames: Iterable[np.ndarray], canvas: np.ndarray, occluder: Callable[[int], np.ndarray | None],
    *, smooth: int = 5,
) -> list[list[float]] | None:
    """Per-frame exposure gain measured on the room around the canvas, in the camera view.

    Light and auto-exposure change the whole frame; paint lands only on the canvas. Measured on the
    canvas, a wash of one colour over the whole bare canvas reads as a change of light and is
    divided out: the second session's pink came out white. Measured on the wall, the floor and the
    easel around it, minus whoever is in the way, the gain only follows the light. The reference is
    the first frame with most of the surround clear; each gain is the median of the per-pixel ratios
    to it over lit room pixels (SURROUND_FLOOR), carried over when too little is visible, and
    smoothed over `smooth` frames. None when the canvas leaves too little of the room in view.
    """
    width = canvas.shape[1]
    surround = ~dilate(canvas, max(2, round(SURROUND_MARGIN * width)))
    if surround.mean() < MIN_SURROUND:
        return None
    reference: np.ndarray | None = None
    reference_usable: np.ndarray | None = None
    last = np.ones(3, dtype=np.float32)
    gains: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        usable = surround.copy()
        mask = occluder(index)
        if mask is not None and mask.any():
            usable &= ~dilate(resize_nearest(mask.astype(bool), canvas.shape), 4)
        current = frame.astype(np.float32)
        usable &= current.max(axis=2) < 250
        if reference is None:
            if usable.mean() >= 0.6 * surround.mean():
                reference, reference_usable = current, usable & (current.min(axis=2) > SURROUND_FLOOR)
            gains.append(last.copy())
            continue
        gain = exposure_gain(current, reference, usable & reference_usable, step=2, percentile=50.0)
        if gain is not None:
            last = gain.astype(np.float32)
        gains.append(last.copy())
    if not gains or reference is None:
        return None
    stacked = np.stack(gains)
    if smooth > 1 and len(stacked) >= smooth:
        half = smooth // 2
        padded = np.pad(stacked, ((half, half), (0, 0)), mode="edge")
        stacked = np.median(np.stack([padded[k:k + len(stacked)] for k in range(smooth)]), axis=0)
    return [[float(value) for value in row] for row in stacked]


def first_visible_canvas(
    frames: Iterable[np.ndarray], occluders: Iterable[np.ndarray | None], *, dilate_px: int,
    steady_frames: int = 3, change_threshold: float = 20.0,
) -> np.ndarray:
    """The canvas before anything happened: what each pixel showed once the painter was off it.

    The masks keep the painter out but miss their shadow. A pixel's first uncovered value is
    settled once it holds for `steady_frames` uncovered frames. If it changes sooner, it gives
    way to the next value only if it could have been a shadow over it, i.e. was nowhere
    brighter. So the hand's shadow in the first frame is skipped, while paint added in the first
    moments still arrives as new paint. Frames are matched to the first one's exposure; pixels
    never uncovered fall back to the first frame.
    """
    pairs = zip(frames, occluders)
    try:
        frame, occluder = next(pairs)
    except StopIteration:
        raise ValueError("no frames") from None
    shape = frame.shape[:2]
    anchor = frame.astype(np.float32)
    anchor_visible = visible_mask(occluder, shape, dilate_px)
    start, candidate = anchor.copy(), anchor.copy()
    streak = np.zeros(shape, dtype=np.int32)
    following = np.ones(shape, dtype=bool)
    """Start still follows the latest run of values: everything before it was shadow-like."""
    done = np.zeros(shape, dtype=bool)
    for frame, occluder in chain([(frame, occluder)], pairs):
        current = frame.astype(np.float32)
        visible = visible_mask(occluder, shape, dilate_px)
        gain = exposure_gain(current, anchor, visible & anchor_visible, step=GAIN_STEP, percentile=GAIN_PERCENTILE)
        if gain is not None:
            current /= gain

        looking = visible & ~done
        steady = np.abs(current - candidate).max(axis=2) <= change_threshold
        holding = looking & steady & (streak > 0)
        fresh = looking & ~holding
        ended = fresh & (streak > 0) & following
        following[ended] = (start[ended] <= current[ended] + change_threshold / 4).all(axis=1)
        streak[holding] += 1
        streak[fresh] = 1
        candidate[fresh] = current[fresh]
        candidate[holding] += (current[holding] - candidate[holding]) / streak[holding][:, None]
        tracking = looking & following
        start[tracking] = candidate[tracking]

        done |= looking & (streak >= steady_frames)
        if done.all():
            break
    return _to_uint8(start)


class CleanPlate:
    """Feed frames in order; each update returns the canvas with occluders removed.

    Start it from first_visible_canvas() when the masks are known up front, or from the
    first frame when they aren't.
    """

    def __init__(self, initial_canvas: np.ndarray, config: CleanPlateConfig | None = None) -> None:
        self.config = config or CleanPlateConfig()
        self._clean = initial_canvas.astype(np.float32)
        self._candidate = self._clean.copy()
        self._reference = self._clean.copy()
        """What each pixel looked like when it was last committed (the start canvas until then).

        Exposure is measured against this, never against the drifting plate: measured against
        the plate, the painter's shadow pulls the gain down, the plate brightens to match, and
        the next estimate takes the brighter plate as the truth. Over an hour that runs away.
        """
        self._committed_at = np.full(initial_canvas.shape[:2], -REFERENCE_AGE, dtype=np.int32)
        self.gain = np.ones(3, dtype=np.float32)
        """The exposure gain last measured (or carried over when it couldn't be)."""
        self._streak = np.zeros(initial_canvas.shape[:2], dtype=np.int32)
        self._since = np.zeros(initial_canvas.shape[:2], dtype=np.int32)
        """Update at which each pixel's current run of a new value started."""
        self.last_commit = np.zeros(initial_canvas.shape[:2], dtype=bool)
        """Pixels that became new paint in the most recent update."""
        self.painted_at = np.full(initial_canvas.shape[:2], -1, dtype=np.int32)
        """Update at which each pixel's latest paint was first seen; -1 if it never changed."""
        self.disagreement = 0.0
        """Share of the uncovered canvas that differed from the plate in the most recent update.

        A stroke in progress is a percent or two; a bumped, blocked or blurred camera is most of it.
        """
        self._blind = 0
        """Updates skipped in a row because the frame was no view of the canvas (see MAX_DISAGREEMENT)."""
        self._updates = 0
        self._commits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._recorded = self._clean.copy()
        """The colour each pixel's latest recorded change says it has (see REFINE_THRESHOLD)."""
        self._entry_chunk = np.full(initial_canvas.shape[:2], -1, dtype=np.int32)
        self._entry_pos = np.zeros(initial_canvas.shape[:2], dtype=np.int32)
        """Where in _commits each pixel's latest change is recorded; chunk -1 means never."""

    def timeline(self) -> Timeline:
        return Timeline.concat(self.painted_at.shape, self._commits)

    @property
    def canvas(self) -> np.ndarray:
        return np.clip(self._clean, 0, 255).astype(np.uint8)

    def update(self, frame: np.ndarray, occluder: np.ndarray | None = None, gain: Iterable[float] | None = None) -> np.ndarray:
        """`gain`, when given, is the exposure measured elsewhere (see surround_gains); otherwise
        it is measured on the canvas."""
        cfg = self.config
        current = frame.astype(np.float32)
        visible = visible_mask(occluder, frame.shape[:2], cfg.dilate_px)
        measured = None if gain is None else np.asarray(list(gain), dtype=np.float32)
        gain = self.gain
        if cfg.normalize_exposure:
            if measured is not None:
                gain = measured
                current = current / gain
            else:
                current, gain = self._match_exposure(current, visible)

        differs = np.abs(current - self._clean).max(axis=2) > cfg.change_threshold
        steady = np.abs(current - self._candidate).max(axis=2) <= cfg.change_threshold
        changing = visible & differs
        self.disagreement = float(changing.sum() / max(int(visible.sum()), 1))
        if self.disagreement > MAX_DISAGREEMENT:
            if self._blind < BLIND_FRAMES:
                # Not a view of the canvas: leave every pixel, streak and the gain exactly as they were.
                self._blind += 1
                self.last_commit = np.zeros(frame.shape[:2], dtype=bool)
                self._updates += 1
                return self.canvas
            # Past the limit the scene has changed for good; keep taking frames until the plate catches up.
        else:
            self._blind = 0
        self.gain = gain

        # Count how long each changed pixel has shown the same new value. Covered
        # pixels keep their streak: we just didn't get to see them this frame.
        continuing = changing & steady & (self._streak > 0)
        restarting = changing & ~continuing
        self._streak[continuing] += 1
        self._streak[restarting] = 1
        self._since[restarting] = self._updates
        self._candidate[restarting] = current[restarting]
        self._candidate[continuing] = 0.5 * (self._candidate[continuing] + current[continuing])

        settled = visible & ~differs
        self._streak[settled] = 0
        self._clean[settled] += cfg.drift_rate * (current[settled] - self._clean[settled])
        if cfg.record:
            self._refine(settled)

        commit = self._streak >= cfg.persist_frames
        if cfg.record and commit.any():
            pixels = np.flatnonzero(commit)
            after = _to_uint8(self._candidate.reshape(-1, 3)[pixels])
            self._commits.append((
                pixels.astype(np.int32),
                self._since.reshape(-1)[pixels],
                _to_uint8(self._clean.reshape(-1, 3)[pixels]),
                after,
            ))
            self._recorded.reshape(-1, 3)[pixels] = after
            self._entry_chunk.reshape(-1)[pixels] = len(self._commits) - 1
            self._entry_pos.reshape(-1)[pixels] = np.arange(len(pixels), dtype=np.int32)
        self._clean[commit] = self._candidate[commit]
        self._reference[commit] = self._candidate[commit]
        self._committed_at[commit] = self._updates
        self._streak[commit] = 0
        self.last_commit = commit
        self.painted_at[commit] = self._since[commit]
        self._updates += 1
        return self.canvas

    def _refine(self, settled: np.ndarray) -> None:
        """Amend the recorded colour of settled pixels that have drifted from it (see REFINE_THRESHOLD)."""
        off = settled & (self._entry_chunk >= 0)
        off[off] = np.abs(self._clean[off] - self._recorded[off]).max(axis=1) > REFINE_THRESHOLD
        if not off.any():
            return
        flat = np.flatnonzero(off)
        chunks = self._entry_chunk.reshape(-1)[flat]
        positions = self._entry_pos.reshape(-1)[flat]
        values = _to_uint8(self._clean.reshape(-1, 3)[flat])
        for chunk in np.unique(chunks):
            inside = chunks == chunk
            self._commits[chunk][3][positions[inside]] = values[inside]
        self._recorded.reshape(-1, 3)[flat] = values

    def _match_exposure(self, current: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undo phone auto-exposure and white-balance drift, using the uncovered canvas.

        The bare canvas is the yardstick while enough of it shows; after that, paint that has
        stood for REFERENCE_AGE updates. When neither shows, or too little of the canvas does,
        the last gain is kept. Returns the matched frame and the gain used.
        """
        gain = None
        if visible.mean() >= MIN_VISIBLE:
            bare = self.painted_at < 0
            settled = self._updates - self._committed_at >= REFERENCE_AGE
            for usable in (visible & bare, visible & settled):
                gain = exposure_gain(current, self._reference, usable, step=GAIN_STEP, percentile=GAIN_PERCENTILE)
                if gain is not None:
                    break
        if gain is None:
            gain = self.gain
        return current / gain, gain
