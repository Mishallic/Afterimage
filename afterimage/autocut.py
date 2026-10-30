"""A rule-based director: a decent reel from the session stats alone, no API needed.

Muse Spark replaces this once the API is unlocked; both hand the renderer the same
EditDecisionList, so the rest of the pipeline doesn't change.
"""

from __future__ import annotations

from .edl import Caption, Clip, EditDecisionList
from .render import time_of


def spoken_duration(seconds: float) -> str:
    hours = seconds / 3600
    if hours >= 1.25:
        return f"{hours:.1f} HOURS".replace(".0 ", " ")
    if hours >= 0.75:
        return "AN HOUR"
    minutes = max(1, round(seconds / 60))
    return f"{minutes} MINUTE{'S' if minutes != 1 else ''}"


def clock(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def auto_edl(stats: dict, *, layout: str = "vertical", fps: float = 30.0) -> EditDecisionList:
    frames = stats["frames"]
    last = frames - 1
    hook = min(max(stats["hook_frame"], 0), last)

    opener = Clip("wipe", hook, hook + 1, 2.8, captions=[
        Caption(f"{spoken_duration(stats['duration_s'])} OF REAL FOOTAGE.", 0.0, 1.4),
        Caption("I REMOVED MYSELF.", 1.4),
    ])

    story = Clip("timelapse", 0, frames, min(16.0, max(6.0, frames / 30)), ease="inout")
    moments = []
    if stats.get("first_paint_frame") is not None:
        moments.append((stats["first_paint_frame"], "FIRST STROKE"))
    if stats["longest_fight_minutes"] >= 1:
        worked = f"{stats['longest_fight_minutes']:.0f} MINUTES ON THE {stats['longest_fight'].upper()}"
        moments.append((stats["longest_fight_frame"], worked))
    for frame, text in moments:
        at = time_of(story, fps, frame)
        story.captions.append(Caption(text, at, min(at + 1.8, story.duration)))

    reveal = Clip("freeze", last, frames, 2.2, captions=[Caption("NO AI-GENERATED PIXELS.", 0.3)])
    end_card = Clip("freeze", last, frames, 3.2, dim=0.6, captions=[
        Caption("MADE BY HAND.\nSHOWN WITHOUT THEM.", 0.15, style="card"),
        Caption(f"{clock(stats['duration_s'])} of painting · every pixel from a real frame · Afterimage", 0.15, style="title"),
    ])
    return EditDecisionList([opener, story, reveal, end_card], layout=layout, fps=fps)
