"""Read the Automatic Tracking Data (Metrica FIFA/EPTS export: metadata XML + raw text).

Raw line:  <frame_id>:<slot>;<slot>;...:<ball>     each group "x,y[,speed]", NaN where unseen.
frame_id counts from 1 and the submission from 0, so video_frame = frame_id - 1 (DATA.md).
Only positions are read. Speeds are recomputed from positions in features.py so that the exact
same computation runs on the training games and on the match.
"""
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """(team code of every track slot in file order, list of per-frame-range slot orders or None).

    The first <Team> is the home side (FIFA/EPTS convention, same as the organisers' baseline)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    team = np.array([HOME if t == team_ids[0] else AWAY for _, t in players], np.int8)
    index = {pid: i for i, (pid, _) in enumerate(players)}
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order, seen = [], set()
        for ref in spec.iter("PlayerChannelRef"):
            pid = channel.get(ref.get("playerChannelId"))
            if pid is None or pid not in index:
                order = None
                break
            if pid not in seen:                 # one entry per slot, whatever its channels are
                seen.add(pid)
                order.append(index[pid])
        if not order:
            specs = []
            break
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(order)))
    return team, (specs or None)


def _xy(group):
    """First two fields of one "x,y[,...]" group as floats; NaN for anything unreadable."""
    f = group.split(",")
    try:
        return float(f[0]), float(f[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def read(meta_path, raw_path, n_frames=None):
    """P (T, slots, 2) and ball (T, 2), normalised 0-1 with NaN for missing; team (slots,)."""
    team, specs = slot_teams(meta_path)
    n_slots = len(team)
    if n_frames is None:                       # one cheap pass to find the length
        with open(raw_path) as handle:
            n_frames = max(int(ln[:ln.index(":")]) for ln in handle if ln.count(":") == 2)
    T = n_frames
    P = np.full((T, n_slots, 2), np.nan)
    ball = np.full((T, 2), np.nan)
    si = 0
    handle = open(raw_path)
    for ln in handle:
        ln = ln.strip()
        if ln.count(":") != 2:
            continue
        fid, players, b = ln.split(":")
        fid = int(fid)
        row = fid - 1
        if not 0 <= row < T:
            continue
        groups = players.split(";")
        nf = groups[0].count(",") + 1
        try:
            vals = np.array(players.replace(";", ",").split(","), dtype=float).reshape(len(groups), nf)[:, :2]
        except ValueError:                      # groups of unequal length: parse them one by one
            vals = np.array([_xy(g) for g in groups])
        order = None
        if specs is not None:
            while si < len(specs) - 1 and fid > specs[si][1]:
                si += 1
            if specs[si][0] <= fid <= specs[si][1] and len(specs[si][2]) == len(groups):
                order = specs[si][2]
        if order is None:
            if len(groups) != n_slots:
                raise ValueError("frame {}: {} position groups but {} track slots in the metadata"
                                 .format(fid, len(groups), n_slots))
            P[row] = vals
        else:
            P[row, order] = vals
        ball[row] = _xy(b)
    handle.close()
    return P, ball, team
