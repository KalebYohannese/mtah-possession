"""Load Metrica's public sample games (other matches) and derive per-frame H/A/D labels from
their corrected tracking + synchronised events. Used only to tune/validate the method."""
import json, os, numpy as np, pandas as pd, xml.etree.ElementTree as ET
ROOT = os.environ.get("METRICA_SAMPLE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample-data", "data"))
H, A, D, X = 0, 1, 2, 3
POSS_EVENTS = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}

def _labels(T, ball, ev, halves):
    """ev: list of (start_frame0, team_code, type). D where ball is NaN; else team of last on-ball event."""
    owner = np.full(T, -1, np.int8)
    ev = sorted([e for e in ev if e[2] in POSS_EVENTS], key=lambda e: e[0])
    fr = np.array([e[0] for e in ev]); tm = np.array([e[1] for e in ev])
    idx = np.searchsorted(fr, np.arange(T), side="right") - 1
    ok = idx >= 0
    owner[ok] = tm[idx[ok]]
    lab = np.full(T, X, np.int8)
    for a, b in halves:
        seg = owner[a:b].copy()
        seg[seg < 0] = D
        dead = np.isnan(ball[a:b, 0])
        seg[dead] = D
        lab[a:b] = seg
    return lab

def load_csv_game(n):
    d = os.path.join(ROOT, f"Sample_Game_{n}")
    hs = pd.read_csv(os.path.join(d, f"Sample_Game_{n}_RawTrackingData_Home_Team.csv"), skiprows=2)
    as_ = pd.read_csv(os.path.join(d, f"Sample_Game_{n}_RawTrackingData_Away_Team.csv"), skiprows=2)
    T = len(hs)
    def players(df):
        v = df.iloc[:, 3:-2].values.astype(float)
        return v.reshape(T, -1, 2)
    hp, ap = players(hs), players(as_)
    ball = hs.iloc[:, -2:].values.astype(float)
    period = hs["Period"].values
    halves = []
    for p in (1, 2):
        w = np.flatnonzero(period == p); halves.append((int(w[0]), int(w[-1]) + 1))
    e = pd.read_csv(os.path.join(d, f"Sample_Game_{n}_RawEventsData.csv"))
    ev = [(int(r["Start Frame"]) - 1, H if r["Team"] == "Home" else A, r["Type"]) for _, r in e.iterrows()]
    P = np.concatenate([hp, ap], axis=1)
    team = np.array([H] * hp.shape[1] + [A] * ap.shape[1], np.int8)
    return dict(name=f"game{n}", P=P, team=team, ball=ball, halves=halves, labels=_labels(T, ball, ev, halves), events=e)

def load_epts_game():
    d = os.path.join(ROOT, "Sample_Game_3")
    root = ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    pteam = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    chan = {pc.get("id"): pc.get("playerId") for pc in root.iter("PlayerChannel")}
    pids = list(dict.fromkeys(chan.values())); slot = {p: i for i, p in enumerate(pids)}
    team = np.array([H if pteam[p] == teams[0] else A for p in pids], np.int8)
    params = {pp.findtext("Name"): pp.findtext("Value") for pp in root.iter("ProviderParameter")}
    halves = [(int(params["first_half_start"]) - 1, int(params["first_half_end"])),
              (int(params["second_half_start"]) - 1, int(params["second_half_end"]))]
    specs = []
    for sp in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in sp.iter("PlayerChannelRef")]
        order = [slot[chan[r]] for r in refs if r.endswith("_x")]
        specs.append((int(sp.get("startFrame")), int(sp.get("endFrame")), order))
    lines = open(os.path.join(d, "Sample_Game_3_tracking.txt")).read().splitlines()
    T = len(lines); P = np.full((T, len(pids), 2), np.nan); ball = np.full((T, 2), np.nan)
    for i, line in enumerate(lines):
        f, pl, b = line.split(":"); f = int(f)
        order = next(o for a, e, o in specs if a <= f <= e)
        vals = np.array([[float(v) for v in g.split(",")[:2]] for g in pl.split(";")])
        P[i, order] = vals
        ball[i] = [float(v) for v in b.split(",")[:2]]
    ev_raw = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
    ev = [(int(e["start"]["frame"]) - 1, H if e["team"]["id"] == teams[0] else A, e["type"]["name"]) for e in ev_raw]
    return dict(name="game3", P=P, team=team, ball=ball, halves=halves,
                labels=_labels(len(P), ball, ev, halves), events=ev_raw)

def load_all(cache=os.path.join(os.path.dirname(os.path.abspath(__file__)), "games_cache.npz")):
    import pickle
    pk = cache.replace(".npz", ".pkl")
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    gs = [load_csv_game(1), load_csv_game(2), load_epts_game()]
    for g in gs: g.pop("events")
    pickle.dump(gs, open(pk, "wb"))
    return gs
