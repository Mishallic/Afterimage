"""SAM 3.1 from its open weights on a Modal GPU: the pipeline's mask source when the API isn't.

One-time setup (needs a Hugging Face account with access to facebook/sam3.1 granted):

    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_...
    modal deploy scripts/sam_modal.py

Then `--masks open` in scripts/nohands_run.py uses it. To try it on a clip by hand:

    modal run scripts/sam_modal.py --video data/session.mp4 --concepts hand,person --seconds 30

which writes out/probe-open/<video>/masks.npz and overlay.mp4 (every mask tinted red).

The container loads the model once and keeps it warm for a couple of minutes; a text prompt
tracks every instance of the concept, a box prompt tracks one thing from one frame both ways.
Long videos are tracked in chunks, so object ids restart per chunk: keys are concept:c<chunk>:<id>.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import modal

APP = "nohands-sam"
VERSION = "sam3.1"
CHUNK_FRAMES = 1800
"""Frames per tracking session: a 30-minute stretch at 1 fps fits a 24 GB card with the video on the CPU."""
GPUS = ["A100-40GB", "A100-80GB", "H100"]
"""A 22 GB card runs out of memory in the multiplex detector (it ran to 19 GB on an L4); 40 GB is the floor.
The multiplex width is baked into the checkpoint (16), so it can't be narrowed to save memory."""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/sam3 /opt/sam3",
        "pip install -e /opt/sam3",
    )
    # The predictor's import chain reaches into sam3's notebook and training extras.
    .pip_install(
        "huggingface_hub", "numpy<2", "pillow", "einops", "pycocotools", "opencv-python-headless",
        "scipy", "scikit-image", "hydra-core", "torchmetrics", "fvcore", "psutil",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
app = modal.App(APP)
weights = modal.Volume.from_name("nohands-hf-cache", create_if_missing=True)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _blobs(frame_masks: dict) -> bytes:
    """Packed like nohands.maskfile, written out here so the container doesn't need the repo."""
    import json

    import numpy as np

    rows = []
    for frame in sorted(frame_masks):
        for key in sorted(frame_masks[frame]):
            mask = frame_masks[frame][key]
            ys, xs = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
            if not len(ys):
                continue
            top, bottom, left, right = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
            rows.append((frame, key, int(left), int(top), mask[top:bottom, left:right]))
    bits, offsets, total = [], [], 0
    for _, _, _, _, part in rows:
        packed = np.packbits(part.astype(bool).ravel())
        bits.append(packed)
        offsets.append(total)
        total += len(packed)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        version=np.int32(1),
        meta=np.array(json.dumps({"model": f"facebook/{VERSION}", "gpu": _gpu_name()}, sort_keys=True)),
        frame=np.array([r[0] for r in rows], dtype=np.int32),
        key=np.array([r[1] for r in rows], dtype=str),
        left=np.array([r[2] for r in rows], dtype=np.int32),
        top=np.array([r[3] for r in rows], dtype=np.int32),
        height=np.array([r[4].shape[0] for r in rows], dtype=np.int32),
        width=np.array([r[4].shape[1] for r in rows], dtype=np.int32),
        offset=np.array(offsets, dtype=np.int64),
        bits=np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8),
    )
    return buffer.getvalue()


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - informational only
        return "unknown"


@app.cls(
    image=image,
    gpu=GPUS,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": weights},
    timeout=3 * 60 * 60,
    scaledown_window=180,
)
class Tracker:
    @modal.enter()
    def load(self) -> None:
        from sam3.model_builder import build_sam3_predictor

        # FlashAttention 3 is Hopper-only; the cards above are Ada and Ampere.
        self.predictor = build_sam3_predictor(version=VERSION, use_fa3=False)
        model = getattr(self.predictor, "model", None)
        if model is not None and hasattr(model, "init_state"):
            # The shared predictor passes offload_state_to_cpu on every start_session; SAM 3.1's
            # multiplex model has no such option and refuses the call. Drop it on the way through.
            original = model.init_state

            def init_state(*args, **kwargs):
                kwargs.pop("offload_state_to_cpu", None)
                return original(*args, **kwargs)

            model.init_state = init_state

    def _chunks(self, video: Path, frames: int) -> list[tuple[int, Path]]:
        """(first frame, folder of JPEG frames) for stretches of at most CHUNK_FRAMES frames.

        The predictor reads a JPEG folder or an MP4; folders need no video decoder in the
        container, and every frame is numbered so chunk boundaries are exact.
        """
        import subprocess

        chunks = []
        for start in range(0, max(frames, 1), CHUNK_FRAMES):
            folder = video.with_name(f"frames-{start}")
            folder.mkdir()
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", str(video),
                "-vf", f"select=gte(n\\,{start})*lt(n\\,{start + CHUNK_FRAMES})", "-vsync", "vfr",
                "-q:v", "2", str(folder / "%05d.jpg"),
            ], check=True)
            chunks.append((start, folder))
        return chunks

    def _session(self, path: Path, prompts: list[dict], first: int, frame_masks: dict, label: str) -> None:
        predictor = self.predictor
        session = predictor.handle_request(request=dict(
            type="start_session", resource_path=str(path), offload_video_to_cpu=True,
        ))["session_id"]
        try:
            for number, prompt in enumerate(prompts, start=1):
                if "text" in prompt:
                    request = dict(type="add_prompt", session_id=session, frame_index=0, text=prompt["text"])
                    key, direction = f"{prompt['text']}:{label}", "forward"
                else:
                    left, top, right, bottom = prompt["box"]
                    # The box is given in proxy pixels; SAM takes relative [x, y, w, h].
                    width, height = self._size(next(path.glob("*.jpg")))
                    request = dict(
                        type="add_prompt", session_id=session, frame_index=prompt["frame"] - first,
                        bounding_boxes=[[left / width, top / height, (right - left) / width, (bottom - top) / height]],
                        bounding_box_labels=[1], rel_coordinates=True, obj_id=number,
                    )
                    key, direction = f"box:{number}", "both"
                prompted = predictor.handle_request(request=request)["outputs"]
                print(f"[{key}] prompt frame: ids={_numpy(prompted['out_obj_ids']).tolist()} "
                      f"probs={[round(float(p), 3) for p in _numpy(prompted.get('out_probs', []))]}")
                yielded, found = 0, 0
                for response in predictor.handle_stream_request(request=dict(
                    type="propagate_in_video", session_id=session, propagation_direction=direction,
                )):
                    outputs = response["outputs"]
                    ids = _numpy(outputs["out_obj_ids"])
                    masks = _numpy(outputs["out_binary_masks"])
                    frame = first + int(response["frame_index"])
                    yielded += 1
                    for object_id, mask in zip(ids, masks):
                        if mask.any():
                            found += 1
                            frame_masks.setdefault(frame, {})[f"{key}:{int(object_id)}"] = mask.astype(bool)
                print(f"[{key}] propagated {yielded} frames, {found} object-frames")
                predictor.handle_request(request=dict(type="reset_session", session_id=session))
        finally:
            predictor.handle_request(request=dict(type="close_session", session_id=session))

    @staticmethod
    def _size(image: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(image) as opened:
            return opened.size

    @staticmethod
    def _frames(path: Path) -> int:
        import json
        import subprocess

        data = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-count_frames", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout)
        stream = next(s for s in data["streams"] if s["codec_type"] == "video")
        return int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)

    @modal.method()
    def track(self, video: bytes, prompts: list[dict]) -> bytes:
        """Masks for every prompt over the whole video, as maskfile bytes.

        prompts: [{"text": "hand"}, ...] or [{"frame": 120, "box": [left, top, right, bottom]}, ...]
        (boxes in the video's pixels, half-open).
        """
        import tempfile

        folder = Path(tempfile.mkdtemp())
        path = folder / "video.mp4"
        path.write_bytes(video)
        frames = self._frames(path)
        frame_masks: dict = {}
        texts = [p for p in prompts if "text" in p]
        boxes = [p for p in prompts if "box" in p]
        for start, chunk in self._chunks(path, frames):
            if texts:
                self._session(chunk, texts, start, frame_masks, f"c{start // CHUNK_FRAMES}")
            mine = [p for p in boxes if start <= p["frame"] < start + CHUNK_FRAMES]
            if mine:
                self._session(chunk, mine, start, frame_masks, f"c{start // CHUNK_FRAMES}")
        return _blobs(frame_masks)


@app.local_entrypoint()
def main(video: str, concepts: str = "hand,person", seconds: float = 30.0, fps: float = 1.0, width: int = 640) -> None:
    """Try the tracker on a clip and write an overlay, like scripts/sam_probe.py. `concepts` is comma-separated."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import numpy as np

    from nohands import maskfile, media
    from nohands.config import ROOT

    concept = [c.strip() for c in concepts.split(",") if c.strip()]
    source = Path(video)
    info = media.probe(source)
    size = media.output_size(info, width)
    work = ROOT / "out" / "probe-open" / f"{source.stem}-{fps:g}fps-{size[0]}px-{seconds:g}s"
    proxy = work / "proxy.mp4"
    if not proxy.exists():
        media.make_proxy(source, proxy, fps=fps, size=size, seconds=seconds)
    data = Tracker().track.remote(proxy.read_bytes(), [{"text": c} for c in concept])
    (work / "masks.npz").write_bytes(data)
    objects, meta = maskfile.unpack(data)
    print(f"{sum(len(b) for b in objects.values())} object-frames over {len(objects)} frames on {meta.get('gpu')}")
    with media.FrameWriter(work / "overlay.mp4", size, fps=max(fps, 4.0)) as writer:
        for index, frame in enumerate(media.read_frames(proxy, fps=fps, size=size)):
            tinted = frame.copy()
            for blob in objects.get(index, {}).values():
                target = np.zeros((size[1], size[0]), dtype=bool)
                blob.paste_into(target)
                tinted[target] = (tinted[target] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            writer.write(tinted)
    print(work / "overlay.mp4")
