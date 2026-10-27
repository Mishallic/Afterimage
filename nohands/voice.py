"""Muse Voice Transcribe: what the painter said while painting, and when.

Quotes are pinned to the moment they were spoken, so the Proof of Hand page can play your own
words next to the stroke you were making. Arabic and English can be mixed in one sentence.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import media
from .config import meta_api_key

URL = "https://api.meta.ai/v1/asr/transcribe"
MODEL = "muse-voice-transcribe-1.0"


def multipart(fields: list[tuple[str, str | None, str, bytes]]) -> tuple[bytes, str]:
    """Encode (name, filename, content type, data) parts as multipart/form-data."""
    boundary = f"nohands-{uuid.uuid4().hex}"
    body = bytearray()
    for name, filename, content_type, data in fields:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disposition}\r\nContent-Type: {content_type}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def transcribe(
    wav: Path, *, mode: str = "ENDPOINTING", languages: tuple[str, ...] = ("English", "Arabic"),
    keywords: tuple[str, ...] = (), retries: int = 3,
) -> dict:
    settings: dict = {"model": MODEL, "mode": mode, "audioEncoding": "WAV", "languageBias": list(languages)}
    if keywords:
        settings["keywords"] = list(keywords)
    body, content_type = multipart([
        ("request", None, "application/json", json.dumps(settings).encode()),
        ("audio", wav.name, "audio/wav", wav.read_bytes()),
    ])
    request = urllib.request.Request(
        f"{URL}?sessionId=nohands-{uuid.uuid4().hex}",
        data=body,
        headers={
            "Authorization": f"Bearer {meta_api_key()}",
            "Accept": "application/json",
            "Content-Type": content_type,
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            # Failed and rate-limited requests aren't billed, so retrying is free.
            if error.code in (429, 500) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def session_quotes(video: Path, work: Path, *, keywords: tuple[str, ...] = ()) -> list[dict]:
    """Every spoken turn in the session, with start and end in seconds from the start."""
    quotes = []
    for wav, offset in media.extract_audio(video, work):
        result = transcribe(wav, keywords=keywords)
        for turn in result.get("turns", []):
            text = turn.get("transcript", "").strip()
            if text:
                quotes.append({
                    "start": offset + turn["startMs"] / 1000,
                    "end": offset + turn["endMs"] / 1000,
                    "text": text,
                })
    return quotes
