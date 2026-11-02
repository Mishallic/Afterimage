"""Follow the canvas when the easel is knocked: where its corners are, second by second.

The corner snap finds the canvas once, in the first frames. On the third session the painter's
knife pushed the easel twice, 13 px two minutes in and 25 more at half an hour (1080p), with a
slow creep and settling around both, and from then on the fixed corners straightened a strip of
wall into the painting's left edge. The camera hadn't moved, so the room is no guide: the canvas
is matched against itself only.

Each frame the canvas is compared with a recent clear view of it, by inverse-compositional
Lucas-Kanade from coarse to fine: a shift, turn and scale within the canvas's plane (what a
knocked easel does), fitted on the view's edges (strokes, the canvas border) with the painter's
masks left out, a gain and offset for the phone's exposure, and robust weights so that paint
laid since the reference doesn't pull. Blank canvas is left out of the fit: it agrees with any
shift. The corners move only when the canvas has, by more than APPLY_PX, holding there for
STEADY_FRAMES estimates. A canvas with too few edges to place it holds its corners; nothing on it
can be misplaced then either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .geometry import canvas_size, homography, order_corners
from .masks import dilate

TRACK_WIDTH = 384
"""Width the canvas is matched at, about its width in a 1280-wide frame."""
LEVELS = 3
"""Pyramid levels, each half the last: a knock of 20 px is 5 at the coarsest."""
ITERATIONS = (30, 15, 8)
"""Gauss-Newton steps allowed per level, coarsest first."""
EDGE = 4.0
"""Gray levels per pixel (of each level) from which a pixel of the reference counts as an edge."""
MIN_EDGES = 0.02
"""Share of the canvas that must be visible edges for a match: below it, too little is painted."""
INLIER_GRAY = 12.0
"""An edge pixel within this many gray levels of the reference, once aligned, lines up."""
MIN_INLIERS = 0.6
"""Share of the visible edges that must line up for a fit to count."""
MAX_CORNER_SD = 0.1
"""Frame pixels: a fit counts only if the noise left in it places every corner this closely."""
APPLY_PX = 0.5
"""Frame pixels the canvas must have moved before its corners follow: half a pixel at 1280 is
three quarters of one in 1080p footage. On the third session the fits of a still canvas wandered
by a third of a pixel while the painter worked over it."""
STEADY_FRAMES = 5
"""Good fits in a row that must agree (within APPLY_PX of their median) before a move is applied:
mid-knock frames are blurred and half covered. The move is dated to the first of them."""
MAX_SCALE = 0.05
"""A fit that scales the canvas by more than this since the reference is wrong: a knocked easel
moves it a few centimetres, not a twentieth of the way to the camera."""
MAX_TURN = np.radians(3.0)
"""Nor does a knock turn it by more than this."""
MAX_JUMP = 0.1
"""Nor move a corner further than this share of the canvas's width from where it last stood:
on the third session, a fit made over the first strokes on a blank canvas put a corner 87 px
off, and every fit after it started from there."""
MAX_OCCLUSION = 0.6
"""Above this share of the canvas covered, the frame isn't matched at all."""
REFERENCE_OCCLUSION = 0.1
"""A frame becomes the reference only with at most this share of the canvas covered."""
REFRESH_OUTLIERS = 0.15
"""Share of the reference's edges that no longer line up (new paint) at which it is replaced."""
REFRESH_FRAMES = 120
"""A reference is replaced after this many frames anyway, when a clear frame comes."""
MASK_PAD = 2
"""Pixels of the occluder mask's own resolution added around it: a hand's edge blurs."""


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _pyramid(gray: np.ndarray) -> list[np.ndarray]:
    levels = [gray]
    for _ in range(LEVELS - 1):
        image = levels[-1]
        image = image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]
        levels.append(0.25 * (image[0::2, 0::2] + image[1::2, 0::2] + image[0::2, 1::2] + image[1::2, 1::2]))
    return levels


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    return mapped[:, :2] / mapped[:, 2:3]


def _sample(image: np.ndarray, points: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear samples at level-0 coordinates (pixel centres at +0.5), and which fell inside the image."""
    height, width = image.shape
    xs = points[:, 0] / 2**level - 0.5
    ys = points[:, 1] / 2**level - 0.5
    x0, y0 = np.floor(xs), np.floor(ys)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < width - 1) & (y0 < height - 1)
    x0, y0 = np.clip(x0, 0, width - 2), np.clip(y0, 0, height - 2)
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy, inside


def _covered(mask: np.ndarray | None, points: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Whether each level-0 point falls on the occluder mask (any resolution, same framing)."""
    if mask is None:
        return np.zeros(len(points), dtype=bool)
    height, width = mask.shape
    xs = np.clip((points[:, 0] * width / frame_size[0]).astype(np.int64), 0, width - 1)
    ys = np.clip((points[:, 1] * height / frame_size[1]).astype(np.int64), 0, height - 1)
    return mask[ys, xs]


def _similarity(delta: np.ndarray) -> np.ndarray:
    scale, turn, dx, dy = delta
    return np.array([[1 + scale, -turn, dx], [turn, 1 + scale, dy], [0.0, 0.0, 1.0]])


def _robust_error(current: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(residual after the best gain and offset, Tukey weights): exposure drift and new paint taken out."""
    weights = np.ones(len(current))
    error = current - template
    for _ in range(3):
        total = weights.sum()
        mean_c, mean_t = (weights * current).sum() / total, (weights * template).sum() / total
        spread = (weights * (current - mean_c) ** 2).sum()
        gain = (weights * (current - mean_c) * (template - mean_t)).sum() / spread if spread > 0 else 1.0
        error = gain * (current - mean_c) + mean_t - template
        sigma = float(np.clip(1.4826 * np.median(np.abs(error)), 1.5, 5.0))
        cutoff = 4.685 * sigma
        weights = np.where(np.abs(error) < cutoff, (1 - (error / cutoff) ** 2) ** 2, 0.0)
    return error, weights


def _phase_shift(template: np.ndarray, current: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(dx, dy) in pixels such that current(x) ~ template(x - d), from the masked images' phase correlation."""
    a = np.where(valid, template, template[valid].mean() if valid.any() else 0.0)
    b = np.where(valid, current, current[valid].mean() if valid.any() else 0.0)
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.rfft2((a - a.mean()) * window)
    fb = np.fft.rfft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    surface = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(surface)), surface.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dx), float(dy)


@dataclass(frozen=True)
class _Fit:
    motion: np.ndarray
    """Similarity in the reference's normalised canvas coordinates: where its content is now."""
    inliers: float
    """Share of the visible edges that line up."""
    corner_sd: float
    """How closely the fit's own noise places the corners, in frame pixels."""
    edges: float
    """Visible edges, as a share of the canvas."""

    @property
    def good(self) -> bool:
        return self.edges >= MIN_EDGES and self.inliers >= MIN_INLIERS and self.corner_sd <= MAX_CORNER_SD


NO_FIT = _Fit(np.eye(3), 0.0, np.inf, 0.0)


class _Reference:
    """A clear view of the canvas, resampled facing it at each pyramid level, ready to match against."""

    def __init__(self, gray: np.ndarray, mask: np.ndarray | None, to_camera: np.ndarray, size: tuple[int, int]) -> None:
        """`to_camera` maps canvas pixels of a `size` view to the frame's pixels."""
        width, height = size
        self.size = size
        self.scale = width / 2
        self.normalise = np.array([[1 / self.scale, 0, -1.0], [0, 1 / self.scale, -height / width], [0, 0, 1]])
        self.to_camera = to_camera @ np.linalg.inv(self.normalise)
        """Normalised canvas coordinates (x from -1 to 1) to frame pixels."""
        corners = self.corners(np.eye(3))
        self.pixels_per_unit = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 4
        frame_size = (gray.shape[1], gray.shape[0])
        self.levels = []
        for level, image in enumerate(_pyramid(gray)):
            step = 2**level
            columns, rows = width // step, height // step
            u, v = np.meshgrid((np.arange(columns) + 0.5) * step, (np.arange(rows) + 0.5) * step)
            grid = np.stack([u.ravel(), v.ravel()], axis=1) @ self.normalise[:2, :2].T + self.normalise[:2, 2]
            camera = _apply(self.to_camera, grid)
            values, inside = _sample(image, camera, level)
            template = values.reshape(rows, columns)
            gy, gx = np.gradient(template)
            visible = (inside & ~_covered(mask, camera, frame_size)).reshape(rows, columns)
            visible[[0, -1], :] = False
            visible[:, [0, -1]] = False
            edges = (visible & (np.hypot(gx, gy) >= EDGE)).ravel()
            x, y = grid[edges, 0], grid[edges, 1]
            # Gradients per normalised unit, times the similarity's Jacobian at the identity.
            gx, gy = gx.ravel()[edges] * self.scale / step, gy.ravel()[edges] * self.scale / step
            self.levels.append({
                "shape": (rows, columns), "grid": grid, "template": template.ravel(), "visible": visible.ravel(),
                "edges": edges, "steepest": np.stack([gx * x + gy * y, gy * x - gx * y, gx, gy], axis=1),
            })
        finest = self.levels[0]
        self.edges = float(finest["edges"].mean())
        """Visible edges, as a share of the canvas."""
        coarsest = self.levels[-1]
        rows, columns = coarsest["shape"]
        self.occlusion = 1.0 - float(coarsest["visible"].reshape(rows, columns)[1:-1, 1:-1].mean())
        """Share of the canvas the view's mask covered or that fell outside the frame."""

    def corners(self, motion: np.ndarray) -> np.ndarray:
        """The canvas corners in the frame, if the canvas moved by `motion` since this view."""
        width, height = self.size
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        return _apply(self.to_camera @ motion @ self.normalise, rect)

    def visible(self, mask: np.ndarray | None, motion: np.ndarray, frame_size: tuple[int, int]) -> float:
        if mask is None:
            return 1.0
        camera = _apply(self.to_camera @ motion, self.levels[-1]["grid"])
        return 1.0 - float(_covered(mask, camera, frame_size).mean())

    def match(
        self, pyramid: list[np.ndarray], mask: np.ndarray | None, start: np.ndarray, frame_size: tuple[int, int],
    ) -> _Fit:
        """The canvas's motion since this view, fitted from `start`."""
        motion = start.copy()
        for level in reversed(range(LEVELS)):
            for _ in range(ITERATIONS[LEVELS - 1 - level]):
                terms = self._terms(pyramid, mask, motion, frame_size, level)
                if terms is None:
                    return NO_FIT
                error, weights, steepest = terms
                hessian = steepest.T @ (weights[:, None] * steepest)
                try:
                    delta = np.linalg.solve(hessian, steepest.T @ (weights * error))
                except np.linalg.LinAlgError:
                    return NO_FIT
                update = np.linalg.inv(_similarity(delta))
                motion = motion @ update
                if np.abs(self.corners(update) - self.corners(np.eye(3))).max() < 0.01 * 2**level:
                    break
        terms = self._terms(pyramid, mask, motion, frame_size, 0)
        if terms is None:
            return NO_FIT
        error, weights, steepest = terms
        lined_up = np.abs(error) < INLIER_GRAY
        hessian = steepest.T @ (weights[:, None] * steepest)
        try:
            covariance = np.linalg.inv(hessian) * float(np.mean(error[lined_up] ** 2)) if lined_up.any() else None
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is None:
            return NO_FIT
        width, height = self.size
        spread = 0.0
        for x, y in ((-1, -height / width), (1, -height / width), (1, height / width), (-1, height / width)):
            jacobian = np.array([[x, -y, 1, 0], [y, x, 0, 1]], dtype=np.float64)
            spread = max(spread, float(np.trace(jacobian @ covariance @ jacobian.T)))
        return _Fit(motion, float(lined_up.mean()), np.sqrt(spread) * self.pixels_per_unit,
                    len(error) / len(self.levels[0]["grid"]))

    def _terms(self, pyramid, mask, motion, frame_size, level):
        """(residual, weights, steepest descent rows) over the reference's edges visible now at `motion`."""
        data = self.levels[level]
        grid = data["grid"][data["edges"]]
        camera = _apply(self.to_camera @ motion, grid)
        values, inside = _sample(pyramid[level], camera, level)
        use = inside & ~_covered(mask, camera, frame_size)
        if use.sum() < 64:
            return None
        error, weights = _robust_error(values[use].astype(np.float64), data["template"][data["edges"]][use].astype(np.float64))
        return error, weights, data["steepest"][use]

    def coarse_shift(self, pyramid: list[np.ndarray], mask: np.ndarray | None, start: np.ndarray, frame_size) -> np.ndarray:
        """`start` moved by the translation phase correlation finds at the coarsest level."""
        level = LEVELS - 1
        data = self.levels[level]
        camera = _apply(self.to_camera @ start, data["grid"])
        values, inside = _sample(pyramid[level], camera, level)
        valid = (data["visible"] & inside & ~_covered(mask, camera, frame_size)).reshape(data["shape"])
        dx, dy = _phase_shift(data["template"].reshape(data["shape"]), values.reshape(data["shape"]), valid)
        step = 2**level / self.scale
        return start @ np.array([[1, 0, dx * step], [0, 1, dy * step], [0, 0, 1]])


def _facing(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return homography(np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64), corners)


def follow(
    frames: Iterable[np.ndarray], corners: np.ndarray,
    occluders: Callable[[int], np.ndarray | None] | None = None, *, log: Callable[[str], None] | None = None,
    fits: Callable[[int, _Fit, np.ndarray, bool], None] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Where the canvas moved to: (first frame, corners in pixels) for each move, in order.

    `frames` are camera frames, `corners` the canvas's corners in the first of them (pixels, any
    order), `occluders(index)` the painter's mask for a frame at any resolution (or None).
    `fits`, if given, is told every fit made: (frame, fit, corners it puts the canvas at, used).
    """
    current = order_corners(corners)
    size = canvas_size(current, TRACK_WIDTH)
    jump = MAX_JUMP * float(np.linalg.norm(current[1] - current[0]))
    moves: list[tuple[int, np.ndarray]] = []
    reference: _Reference | None = None
    reference_frame = 0
    anchor = np.eye(3)
    """Where the canvas last stood still, as motion since the reference: every fit starts here,
    so a wrong one is forgotten by the next frame."""
    recent: list[tuple[int, np.ndarray, np.ndarray]] = []
    """The last good fits: (frame, corners they put the canvas at, motion)."""

    def plausible(fit: _Fit) -> bool:
        if not fit.good:
            return False
        linear = fit.motion[:2, :2] / fit.motion[2, 2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        turn = float(np.arctan2(linear[1, 0], linear[0, 0]))
        if abs(scale - 1) > MAX_SCALE or abs(turn) > MAX_TURN:
            return False
        return float(np.abs(reference.corners(fit.motion) - reference.corners(anchor)).max()) <= jump

    for index, frame in enumerate(frames):
        frame_size = (frame.shape[1], frame.shape[0])
        mask = None if occluders is None else occluders(index)
        if mask is not None and MASK_PAD:
            mask = dilate(mask, MASK_PAD)
        pyramid = _pyramid(_gray(frame))
        if reference is None or reference.edges < MIN_EDGES:
            # Nothing painted to place the canvas by yet: keep a clear view of it where the corners are.
            view = _Reference(pyramid[0], mask, _facing(current, size), size)
            if view.occlusion <= REFERENCE_OCCLUSION:
                reference, reference_frame, anchor, recent = view, index, np.eye(3), []
            continue
        if reference.visible(mask, anchor, frame_size) < 1 - MAX_OCCLUSION:
            continue
        fit = reference.match(pyramid, mask, anchor, frame_size)
        if not plausible(fit):
            # A knock moves the canvas further than Lucas-Kanade reaches: start again from the phase.
            retry = reference.match(pyramid, mask, reference.coarse_shift(pyramid, mask, anchor, frame_size), frame_size)
            if plausible(retry) or retry.inliers > fit.inliers:
                fit = retry
        used = plausible(fit)
        if fits:
            fits(index, fit, reference.corners(fit.motion), used)
        if not used:
            continue
        recent = (recent + [(index, reference.corners(fit.motion), fit.motion)])[-STEADY_FRAMES:]
        steady = len(recent) == STEADY_FRAMES
        if steady:
            middle = np.median([corners for _, corners, _ in recent], axis=0)
            steady = max(np.abs(corners - middle).max() for _, corners, _ in recent) <= APPLY_PX
        if not steady:
            continue
        anchor = fit.motion
        if np.abs(middle - current).max() > APPLY_PX:
            current = middle
            moves.append((recent[0][0], current.copy()))
            if log:
                log(f"  canvas moved at frame {recent[0][0]}: corners {np.round(current, 1).tolist()}")
        if 1 - fit.inliers >= REFRESH_OUTLIERS or index - reference_frame >= REFRESH_FRAMES:
            view = _Reference(pyramid[0], mask, reference.to_camera @ anchor @ reference.normalise, size)
            if view.occlusion <= REFERENCE_OCCLUSION and view.edges >= MIN_EDGES:
                # The new view's own frame is where the canvas is now; earlier fits were against the old one.
                reference, reference_frame, anchor, recent = view, index, np.eye(3), []
    return moves
