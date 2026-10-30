"""Masks from the open-weight SAM, on disk: one .npz per tracking call, hashed into the receipt.

Each object in each frame is stored the way Blob keeps it in memory, as a bounding box plus
its bit-packed mask, so a two-hour session fits in a few megabytes. The file is written in a
fixed order, so the same masks always give the same bytes and the same hash.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .masks import Blob
from .occluders import Objects

VERSION = 1


def pack(objects: Objects, meta: dict | None = None) -> bytes:
    """Serialise frame -> {key -> Blob} as compressed npz bytes."""
    rows = sorted((frame, key, blob) for frame, blobs in objects.items() for key, blob in blobs.items())
    bits, offsets, total = [], [], 0
    for _, _, blob in rows:
        packed = np.packbits(blob.mask.astype(bool).ravel())
        bits.append(packed)
        offsets.append(total)
        total += len(packed)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        version=np.int32(VERSION),
        meta=np.array(json.dumps(meta or {}, sort_keys=True)),
        frame=np.array([frame for frame, _, _ in rows], dtype=np.int32),
        key=np.array([key for _, key, _ in rows], dtype=str),
        left=np.array([blob.left for _, _, blob in rows], dtype=np.int32),
        top=np.array([blob.top for _, _, blob in rows], dtype=np.int32),
        height=np.array([blob.mask.shape[0] for _, _, blob in rows], dtype=np.int32),
        width=np.array([blob.mask.shape[1] for _, _, blob in rows], dtype=np.int32),
        offset=np.array(offsets, dtype=np.int64),
        bits=np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8),
    )
    return buffer.getvalue()


def unpack(data: bytes) -> tuple[Objects, dict]:
    with np.load(io.BytesIO(data)) as saved:
        meta = json.loads(str(saved["meta"]))
        frames, keys = saved["frame"], saved["key"]
        left, top, height, width = saved["left"], saved["top"], saved["height"], saved["width"]
        offsets, bits = saved["offset"], saved["bits"]
    objects: Objects = {}
    for index in range(len(frames)):
        count = int(height[index]) * int(width[index])
        start = int(offsets[index])
        mask = np.unpackbits(bits[start:start + (count + 7) // 8])[:count].astype(bool)
        blob = Blob(int(left[index]), int(top[index]), mask.reshape(int(height[index]), int(width[index])))
        objects.setdefault(int(frames[index]), {})[str(keys[index])] = blob
    return objects, meta


def save(path: Path, objects: Objects, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack(objects, meta))
    return path


def load(path: Path) -> tuple[Objects, dict]:
    return unpack(path.read_bytes())


def from_arrays(frame_masks: dict[int, dict[str, np.ndarray]]) -> Objects:
    """Full-frame bool masks per object per frame, as Blobs (empty masks dropped)."""
    objects: Objects = {}
    for frame, masks in frame_masks.items():
        for key, mask in masks.items():
            blob = Blob.of(np.asarray(mask, dtype=bool))
            if blob is not None:
                objects.setdefault(frame, {})[key] = blob
    return objects
