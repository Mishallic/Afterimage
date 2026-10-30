"""Turn an edit decision list into a finished MP4 reel with burned-in captions."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import imaging, media
from .edl import Caption, Clip, EditDecisionList

SIZES = {"vertical": (1080, 1920), "landscape": (1920, 1080)}
FONT = "Arial Black" if sys.platform == "win32" else "DejaVu Sans"
BACKDROP_EVERY = 15
"""Rebuild the blurred backdrop every this many output frames; it barely changes."""


@dataclass(frozen=True)
class Sources:
    clean: Path
    raw: Path
    size: tuple[int, int]
    """(width, height) of the frames in both files."""
    fps: float
    """Frame rate the files were written at: sampled frame i sits at i / fps."""


def ease(progress: float, kind: str) -> float:
    if kind == "in":
        return progress * progress
    if kind == "out":
        return 1 - (1 - progress) ** 2
    if kind == "inout":
        return progress * progress * (3 - 2 * progress)
    return progress


def source_indices(clip: Clip, fps: float) -> list[int]:
    """Which session frame each output frame of the clip shows."""
    count = max(1, round(clip.duration * fps))
    if clip.kind == "freeze" or clip.end - 1 <= clip.start:
        return [clip.start] * count
    span = clip.end - 1 - clip.start
    return [round(clip.start + ease((k + 0.5) / count, clip.ease) * span) for k in range(count)]


def time_of(clip: Clip, fps: float, frame: int) -> float:
    """Seconds into the clip when it first reaches session frame `frame`."""
    for k, index in enumerate(source_indices(clip, fps)):
        if index >= frame:
            return k / fps
    return clip.duration


class Composer:
    """Lays the canvas out on the reel's frame over a blurred backdrop of itself."""

    def __init__(self, layout: str, canvas_size: tuple[int, int]) -> None:
        self.size = SIZES[layout]
        width, height = self.size
        if layout == "vertical":
            box, self.center = (width - 80, round(height * 0.6)), (width // 2, round(height * 0.46))
        else:
            box, self.center = (width - 200, height - 240), (width // 2, height // 2 - 30)
        self.layout = layout
        self.display = imaging.fit_size(canvas_size, box)
        self._backdrop: np.ndarray | None = None
        self._frames = 0

    def _base(self, canvas: np.ndarray) -> np.ndarray:
        if self._backdrop is None or self._frames % BACKDROP_EVERY == 0:
            self._backdrop = imaging.backdrop(canvas, self.size)
        self._frames += 1
        return self._backdrop.copy()

    def single(self, canvas: np.ndarray, zoom: tuple[float, float, float, float] | None, progress: float) -> np.ndarray:
        view = canvas if zoom is None else _zoomed(canvas, zoom, progress, self.display)
        return imaging.paste(self._base(canvas), imaging.resize(view, self.display), self.center)

    def wipe(self, raw: np.ndarray, clean: np.ndarray, progress: float) -> np.ndarray:
        """The painter gets erased as a line sweeps across: clean behind it, raw ahead of it."""
        shown = imaging.resize(raw, self.display).copy()
        clean_shown = imaging.resize(clean, self.display)
        cut = round(progress * shown.shape[1])
        shown[:, :cut] = clean_shown[:, :cut]
        shown[:, max(cut - 3, 0):cut + 3] = 255
        return imaging.paste(self._base(clean), shown, self.center)

    def split(self, raw: np.ndarray, clean: np.ndarray) -> np.ndarray:
        """Raw and clean together: stacked on vertical reels, side by side on landscape."""
        base = self._base(clean)
        cx, cy = self.center
        if self.layout == "vertical":
            half = imaging.fit_size(self.display, (self.display[0], self.display[1] // 2 - 10))
            offsets = [(cx, cy - half[1] // 2 - 10), (cx, cy + half[1] // 2 + 10)]
        else:
            half = imaging.fit_size(self.display, (self.display[0] // 2 - 10, self.display[1]))
            offsets = [(cx - half[0] // 2 - 10, cy), (cx + half[0] // 2 + 10, cy)]
        for image, center in zip((raw, clean), offsets):
            imaging.paste(base, imaging.resize(image, half), center)
        return base


def _zoomed(canvas: np.ndarray, box: tuple[float, float, float, float], progress: float, display: tuple[int, int]) -> np.ndarray:
    """Crop that moves from the full canvas to `box`, kept at the display's aspect ratio."""
    height, width = canvas.shape[:2]
    x, y, w, h = (a + (b - a) * progress for a, b in zip((0.0, 0.0, 1.0, 1.0), box))
    crop_w, crop_h = max(w * width, 8), max(h * height, 8)
    aspect = display[0] / display[1]
    if crop_w / crop_h > aspect:
        crop_h = crop_w / aspect
    else:
        crop_w = crop_h * aspect
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    left = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
    top = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
    return canvas[top:top + int(crop_h), left:left + int(crop_w)]


def _clip_frames(clip: Clip, indices: list[int], sources: Sources) -> Iterator[tuple[np.ndarray, ...]]:
    def read(path: Path) -> Iterator[np.ndarray]:
        return media.read_indices(path, indices, fps=sources.fps, size=sources.size)

    if clip.kind == "freeze":
        with closing(read(sources.clean)) as frames:
            frame = next(frames)
        for _ in indices:
            yield (frame,)
    elif clip.kind in ("timelapse", "raw"):
        with closing(read(sources.clean if clip.kind == "timelapse" else sources.raw)) as frames:
            for frame in frames:
                yield (frame,)
    else:
        with closing(read(sources.raw)) as raw, closing(read(sources.clean)) as clean:
            yield from zip(raw, clean)


def _compose(composer: Composer, clip: Clip, frames: tuple[np.ndarray, ...], progress: float) -> np.ndarray:
    if clip.kind == "wipe":
        picture = composer.wipe(frames[0], frames[1], ease(progress, "inout"))
    elif clip.kind == "split":
        picture = composer.split(frames[0], frames[1])
    else:
        picture = composer.single(frames[0], clip.zoom, ease(progress, "inout"))
    if clip.dim:
        picture = (picture.astype(np.float32) * (1 - clip.dim)).astype(np.uint8)
    return picture


def render(edl: EditDecisionList, sources: Sources, out: Path) -> Path:
    composer = Composer(edl.layout, sources.size)
    work = out.with_suffix(".work")
    work.mkdir(parents=True, exist_ok=True)
    silent = work / "silent.mp4"
    events: list[tuple[float, float, Caption]] = []
    elapsed = 0.0
    with media.FrameWriter(silent, composer.size, edl.fps) as writer:
        for clip in edl.clips:
            indices = source_indices(clip, edl.fps)
            for k, frames in enumerate(_clip_frames(clip, indices, sources)):
                writer.write(_compose(composer, clip, frames, (k + 0.5) / len(indices)))
            length = len(indices) / edl.fps
            for caption in clip.captions:
                end = length if caption.end is None else min(caption.end, length)
                events.append((elapsed + caption.start, elapsed + end, caption))
            elapsed += length

    if not events:
        silent.replace(out)
        return out
    subtitles = work / "captions.ass"
    subtitles.write_text(_ass(events, composer.size), encoding="utf-8")
    try:
        media.burn_subtitles(silent, subtitles, out)
    except RuntimeError as error:
        print(f"captions skipped ({error})", file=sys.stderr)
        silent.replace(out)
    return out


def _ass(events: list[tuple[float, float, Caption]], size: tuple[int, int]) -> str:
    width, height = size
    scale = min(width, height) / 1080
    vertical = height > width
    bottom = round((330 if vertical else 70) * scale)
    top = round((150 if vertical else 50) * scale)

    def style(name: str, font_size: int, alignment: int, margin: int, outline: int) -> str:
        return (
            f"Style: {name},{FONT},{round(font_size * scale)},&H00FFFFFF,&H00FFFFFF,&H00000000,"
            f"&H96000000,-1,0,0,0,100,100,0,0,1,{outline},2,{alignment},60,60,{margin},1"
        )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        style("caption", 76, 2, bottom, 6),
        style("title", 44, 8, top, 4),
        style("card", 58, 5, 0, 5),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, caption in events:
        text = caption.text.replace("{", "(").replace("}", ")").replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},{caption.style},,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def _timestamp(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
