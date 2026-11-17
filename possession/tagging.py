"""Smart Tagging (SportsCode XML) -> the few facts we trust it for.

The tagging is automatic and its intervals are windows, not instants, so it is used only where the
laws of the game make the label certain: play is dead until a kick-off is taken -- before each half
starts and between a goal and the restart. Everything else is left to the model.
"""
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25


def load(path):
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): (lab.findtext("text") or "") for lab in inst.findall("label")}
        out.append({"code": (inst.findtext("code") or "").strip().lower(),
                    "start": int(round(float(inst.findtext("start")) * FPS)),
                    "end": int(round(float(inst.findtext("end")) * FPS)),
                    "type": (tags.get("Type") or "").strip().lower()})
    return out


def _kickoff_frame(ev, ball, pitch):
    """Earliest frame in the tag window at which the ball has left the centre spot; the window's
    first frame if that cannot be seen. Deliberately early: frames after it go back to the model."""
    a, b = max(ev["start"], 0), min(ev["end"] + 50, len(ball))
    bx = (ball[a:b, 0] - 0.5) * pitch[0]
    by = (ball[a:b, 1] - 0.5) * pitch[1]
    r = np.hypot(bx, by)
    at_spot = np.flatnonzero(r < 1.5)
    if len(at_spot):
        away = np.flatnonzero(r[at_spot[0]:] > 2.0)
        if len(away):
            return a + at_spot[0] + int(away[0]) - 3
    return a


def dead_intervals(events, ball, blocks, pitch):
    """[(start, stop)] half-open frame intervals that must be dead ball."""
    out = []
    goals = [e for e in events if e["code"] == "goals"]
    for ev in events:
        if ev["code"] != "set pieces" or not ev["type"].startswith("kick off"):
            continue
        block = next(((s, e) for s, e in blocks if s <= ev["start"] < e), None)
        if block is None:
            continue
        ko = max(_kickoff_frame(ev, ball, pitch), block[0])
        if "start" in ev["type"]:
            if ko - block[0] < 10 * 60 * FPS:        # a kick-off tag late in a half is not the start
                out.append((block[0], ko))
        else:
            start = ko - 15 * FPS                    # a goal is never restarted faster than this
            prior = [g for g in goals if 0 <= ko - g["end"] <= 180 * FPS]
            if prior:
                start = min(start, max(g["end"] for g in prior) + 2 * FPS)
            out.append((max(start, block[0]), ko))
    return [(a, b) for a, b in out if b > a]
