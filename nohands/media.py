"""ffmpeg helpers: probing, SAM proxies, and piping frames in and out as numpy arrays."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

from .config import find_tool


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def _run(args: list[str], **kwargs) -> str:
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} failed:\n{result.stderr[-2000:]}")
    return result.stdout


def probe(video: Path) -> VideoInfo:
    """Read size (after phone rotation), duration, frame rate and whether there's audio."""
    data = json.loads(
        _run([
            find_tool("ffprobe"), "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(video),
        ])
    )
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = abs(int(float(side["rotation"])))
    rotation = rotation or abs(int(stream.get("tags", {}).get("rotate", 0)))
    if rotation % 180 == 90:
        width, height = height, width
    numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0.0)
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return VideoInfo(width, height, duration, fps, has_audio)


def output_size(info: VideoInfo, width: int) -> tuple[int, int]:
    """Even (width, height) at most `width` wide, keeping the aspect ratio."""
    width = min(width, info.width) // 2 * 2
    height = round(info.height * width / info.width / 2) * 2
    return width, height


def _input_args(video: Path, seconds: float | None) -> list[str]:
    args = [find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video)]
    if seconds:
        args += ["-t", str(seconds)]
    return args


def _sampling_filter(fps: float, size: tuple[int, int]) -> str:
    # Every output uses the same fps filter, so frame i sits at time i / fps everywhere
    # and SAM's frame indices line up with the full-resolution frames.
    return f"fps={fps},scale={size[0]}:{size[1]}:flags=area"


def make_proxy(
    video: Path, out: Path, *, fps: float, size: tuple[int, int], seconds: float | None = None
) -> Path:
    """Low-fps, low-res H.264 copy for SAM: it bills per frame, and occluder masks need no detail."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(_input_args(video, seconds) + [
        "-vf", _sampling_filter(fps, size), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(out),
    ])
    return out


def make_moments(
    video: Path, out: Path, *, size: tuple[int, int], seconds: float | None = None
) -> Path:
    """Natural-speed copy with sound for the pixel receipt: the whole camera frame, cheap to seek in."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(_input_args(video, seconds) + [
        "-vf", f"scale={size[0]}:{size[1]}:flags=area", "-g", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        str(out),
    ])
    return out


def _stream(args: list[str], size: tuple[int, int], label: str) -> Iterator[np.ndarray]:
    width, height = size
    frame_bytes = width * height * 3
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    try:
        while len(chunk := process.stdout.read(frame_bytes)) == frame_bytes:
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed reading {label}:\n{process.stderr.read().decode()[-2000:]}")
    finally:
        # A caller that stops early leaves ffmpeg blocked on a full pipe; don't wait on it.
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def read_frames(
    video: Path, *, fps: float, size: tuple[int, int], seconds: float | None = None
) -> Iterator[np.ndarray]:
    """Decode RGB frames, sampled exactly like make_proxy."""
    args = _input_args(video, seconds) + [
        "-vf", _sampling_filter(fps, size), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    return _stream(args, size, str(video))


def read_indices(
    video: Path, indices: Sequence[int], *, fps: float, size: tuple[int, int]
) -> Iterator[np.ndarray]:
    """Frames at non-decreasing `indices` of a constant-frame-rate video, repeats included."""
    if not indices:
        return iter(())
    first = indices[0]
    # Seek a quarter frame early: frame `first` sits exactly on its timestamp.
    args = [
        find_tool("ffmpeg"), "-v", "error", "-ss", f"{max(first - 0.25, 0) / fps:.6f}",
        "-i", str(video), "-vf", f"scale={size[0]}:{size[1]}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    return _pick(_stream(args, size, str(video)), indices, first)


def _pick(frames: Iterator[np.ndarray], indices: Sequence[int], first: int) -> Iterator[np.ndarray]:
    wanted = iter(indices)
    target = next(wanted, None)
    try:
        for position, frame in enumerate(frames, start=first):
            while target == position:
                yield frame
                target = next(wanted, None)
            if target is None:
                return
    finally:
        frames.close()


def write_image(frame: np.ndarray, out: Path) -> Path:
    """Save one RGB frame as PNG or JPEG (picked by the file extension)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    quality = ["-q:v", "3"] if out.suffix.lower() in (".jpg", ".jpeg") else []
    result = subprocess.run(
        [
            find_tool("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1", *quality, str(out),
        ],
        input=np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}:\n{result.stderr.decode()[-2000:]}")
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 3) -> bytes:
    """One RGB frame as JPEG bytes, without touching the disk."""
    height, width = frame.shape[:2]
    result = subprocess.run(
        [
            find_tool("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1",
            "-q:v", str(quality), "-f", "image2", "-c:v", "mjpeg", "pipe:1",
        ],
        input=np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding a JPEG:\n{result.stderr.decode()[-2000:]}")
    return result.stdout


def extract_audio(video: Path, out_dir: Path, *, chunk_seconds: int = 590) -> list[tuple[Path, float]]:
    """Mono 16-bit 24 kHz WAV chunks (Voice Transcribe takes up to 10 minutes per file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("chunk_*.wav"):
        stale.unlink()
    _run([
        find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-vn",
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1",
        "-f", "segment", "-segment_time", str(chunk_seconds), str(out_dir / "chunk_%03d.wav"),
    ])
    chunks = sorted(out_dir.glob("chunk_*.wav"))
    return [(path, number * float(chunk_seconds)) for number, path in enumerate(chunks)]


def burn_subtitles(video: Path, subtitles: Path, out: Path) -> Path:
    """Burn an .ass file into a video."""
    # Running from the subtitle's folder keeps Windows drive colons out of the filter string.
    _run(
        [
            find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video.resolve()),
            "-vf", f"subtitles={subtitles.name}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out.resolve()),
        ],
        cwd=subtitles.parent,
    )
    return out


class FrameWriter:
    """Pipe RGB frames straight into a browser-friendly H.264 MP4."""

    def __init__(self, out: Path, size: tuple[int, int], fps: float = 30.0) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.out = out
        self._process = subprocess.Popen(
            [
                find_tool("ffmpeg"), "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                str(out),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self._process.stdin is not None and self._process.stderr is not None
        self._process.stdin.close()
        code = self._process.wait()
        error = self._process.stderr.read().decode()
        self._process.stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.out}:\n{error[-2000:]}")

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
