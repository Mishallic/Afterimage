"""Post-match stats for a painting session, in the spirit of an end-of-game scoreboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Rough RGB values of common tube colours, for naming the MVP colour.
PIGMENTS: dict[str, tuple[int, int, int]] = {
    "titanium white": (242, 242, 236),
    "ivory black": (32, 30, 30),
    "payne's grey": (62, 72, 84),
    "cadmium red": (214, 45, 35),
    "alizarin crimson": (150, 25, 45),
    "cadmium orange": (236, 120, 35),
    "cadmium yellow": (248, 205, 40),
    "yellow ochre": (200, 150, 65),
    "naples yellow": (240, 214, 150),
    "raw sienna": (190, 120, 60),
    "burnt sienna": (138, 68, 40),
    "burnt umber": (88, 60, 42),
    "sap green": (80, 110, 45),
    "phthalo green": (20, 100, 80),
    "cerulean blue": (60, 140, 200),
    "ultramarine blue": (45, 60, 165),
    "phthalo blue": (20, 45, 105),
    "dioxazine purple": (80, 40, 110),
}

COMMIT_SHARE = 0.0005
"""Frames where less than this share of the canvas changed count as idle."""
TOUCH_SHARE = 0.02
"""A grid cell counts as touched when this share of it changed in one frame."""
EPISODE_GAP_S = 60.0
"""Touches of an area further apart than this count as separate visits."""
PAINT_DIFFERENCE = 30
"""How far (0-255) a pixel must be from the canvas's base colour to count as paint,
rather than bare canvas reappearing from behind a hand or its shadow."""
COVERAGE_POINTS = 60


def pigment_name(rgb: tuple[int, int, int]) -> str:
    colour = np.array(rgb, dtype=np.float64)
    return min(PIGMENTS, key=lambda name: float(np.sum((np.array(PIGMENTS[name]) - colour) ** 2)))


def region_name(row: int, col: int, grid: tuple[int, int]) -> str:
    vertical = ["top", "middle", "bottom"][min(row * 3 // grid[0], 2)]
    horizontal = ["left", "center", "right"][min(col * 3 // grid[1], 2)]
    if (vertical, horizontal) == ("middle", "center"):
        return "center"
    return f"{vertical} {horizontal}"


@dataclass
class SessionStats:
    frames: int
    duration_s: float
    active_s: float
    apm: float
    first_paint_frame: int | None
    first_paint_s: float | None
    mvp_color: str
    mvp_rgb: tuple[int, int, int]
    longest_fight: str
    longest_fight_minutes: float
    longest_fight_frame: int
    longest_fight_box: tuple[float, float, float, float]
    rework_areas: int
    hook_frame: int
    coverage: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


class StatsTracker:
    """Fed every frame of the clean plate; summarises the session at the end."""

    def __init__(self, initial_canvas: np.ndarray, fps: float, grid: tuple[int, int] = (9, 9)) -> None:
        height, width = initial_canvas.shape[:2]
        self.fps = fps
        self.grid = grid
        self._row_starts = np.linspace(0, height, grid[0] + 1).astype(int)[:-1]
        self._col_starts = np.linspace(0, width, grid[1] + 1).astype(int)[:-1]
        row_sizes = np.diff(np.append(self._row_starts, height))
        col_sizes = np.diff(np.append(self._col_starts, width))
        self._cell_pixels = np.outer(row_sizes, col_sizes).astype(np.float64)
        # One base colour (gesso white, or a toned ground) is sturdier than a per-pixel snapshot,
        # which can carry a hand's shadow from the first frame.
        self._base = np.median(initial_canvas.reshape(-1, 3), axis=0).astype(np.int16)
        self._frames = 0
        self._active = 0
        self._actions = 0
        self._first: int | None = None
        self._touches = np.zeros(grid, dtype=np.int64)
        self._touch_frame_sum = np.zeros(grid, dtype=np.float64)
        self._last_touch = np.full(grid, -np.inf)
        self._episodes = np.zeros(grid, dtype=np.int64)
        self._colours = np.zeros(16 ** 3, dtype=np.int64)
        self._coverage: list[tuple[int, float]] = []
        self._occlusion: list[float] = []
        self._coverage_every = max(1, round(fps * 20))

    def update(self, canvas: np.ndarray, committed: np.ndarray, occluded_share: float = 0.0) -> None:
        index = self._frames
        self._frames += 1
        self._occlusion.append(occluded_share)
        paint = committed & self._painted(canvas)

        if paint.mean() >= COMMIT_SHARE:
            self._active += 1
            if self._first is None:
                self._first = index
            pixels = canvas[paint]
            quantized = pixels[:: max(1, len(pixels) // 4000)].astype(np.int64) // 16
            bins = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
            self._colours += np.bincount(bins, minlength=16 ** 3)

        per_cell = np.add.reduceat(
            np.add.reduceat(paint.astype(np.int64), self._row_starts, axis=0), self._col_starts, axis=1
        )
        touched = per_cell / self._cell_pixels >= TOUCH_SHARE
        if touched.any():
            self._actions += int(touched.sum())
            self._touches += touched
            self._touch_frame_sum += touched * index
            self._episodes += touched & (index - self._last_touch > EPISODE_GAP_S * self.fps)
            self._last_touch[touched] = index

        if index % self._coverage_every == 0:
            self._coverage.append((index, self._coverage_of(canvas)))

    def finish(self, final_canvas: np.ndarray) -> SessionStats:
        frames = max(self._frames, 1)
        self._coverage.append((frames - 1, self._coverage_of(final_canvas)))
        active_s = self._active / self.fps

        if self._colours.any():
            top = int(np.argmax(self._colours))
            rgb = ((top // 256) * 16 + 8, (top // 16 % 16) * 16 + 8, (top % 16) * 16 + 8)
        else:
            rgb = (0, 0, 0)

        row, col = np.unravel_index(int(np.argmax(self._touches)), self.grid)
        touches = int(self._touches[row, col])
        return SessionStats(
            frames=frames,
            duration_s=frames / self.fps,
            active_s=active_s,
            apm=self._actions / max(active_s / 60, 1 / 60),
            first_paint_frame=self._first,
            first_paint_s=None if self._first is None else self._first / self.fps,
            mvp_color=pigment_name(rgb),
            mvp_rgb=rgb,
            longest_fight=region_name(int(row), int(col), self.grid),
            longest_fight_minutes=touches / self.fps / 60,
            longest_fight_frame=round(self._touch_frame_sum[row, col] / touches) if touches else 0,
            longest_fight_box=(col / self.grid[1], row / self.grid[0], 1 / self.grid[1], 1 / self.grid[0]),
            rework_areas=int((self._episodes >= 3).sum()),
            hook_frame=self._hook_frame(frames),
            coverage=self._coverage_curve(),
        )

    def _painted(self, canvas: np.ndarray) -> np.ndarray:
        return np.abs(canvas.astype(np.int16) - self._base).max(axis=2) > PAINT_DIFFERENCE

    def _coverage_of(self, canvas: np.ndarray) -> float:
        return float(self._painted(canvas[::4, ::4]).mean())

    def _hook_frame(self, frames: int) -> int:
        """The most covered-up moment once the painting is mostly there: the best "before"."""
        occlusion = np.array(self._occlusion[:frames] or [0.0])
        if not occlusion.any():
            return round((frames - 1) * 0.8)
        indices, values = zip(*self._coverage)
        coverage = np.interp(np.arange(len(occlusion)), indices, values)
        late = coverage >= 0.7 * coverage[-1]
        return int(np.argmax(np.where(late, occlusion, -1.0)))

    def _coverage_curve(self) -> list[float]:
        values = [value for _, value in self._coverage]
        if len(values) > COVERAGE_POINTS:
            picks = np.linspace(0, len(values) - 1, COVERAGE_POINTS).round().astype(int)
            values = [values[i] for i in picks]
        return [round(value, 4) for value in values]
