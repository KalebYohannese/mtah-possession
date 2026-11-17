"""Degrade a clean sample game until it looks like the challenge's automatic tracking (ATD).

Metrica's public sample games are corrected tracking: the ball is always there while play is live
and absent exactly when play is dead. The challenge ATD is the opposite kind of data -- the ball is
seen on ~59% of graded frames and is lost in long runs (seconds to minutes), and a stopped ball is
often still tracked. Both facts are stated in DATA.md or can be counted in the committed
baseline_raw.csv; nothing here uses a ground truth of the challenge match.

So before training we: hide the ball in long runs during live play, give some stoppages a visible
ball (resting at the restart spot, lingering where it died, or being carried), hide it in others,
drop player tracks for a while, and add positional jitter.
"""
import numpy as np

PITCH = (105.0, 68.0)


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def degrade(g, rng, p_dead_visible=0.5, alive_miss_frac=0.2, track_drop=0.07, jitter=0.10,
            ball_jitter=0.15):
    P = g["P"].copy()
    ball = g["ball"].copy()
    T = len(ball)
    hide = np.zeros(T, bool)
    for (h0, h1) in g["halves"]:
        dead = np.isnan(g["ball"][h0:h1, 0])
        s_, e_ = _runs(dead)
        for a, b in zip(s_ + h0, e_ + h0):
            L = b - a
            prev = g["ball"][a - 1] if a > h0 else None
            nxt = g["ball"][b] if b < h1 else None
            if nxt is None and prev is None:
                hide[a:b] = True
                continue
            if nxt is None:
                nxt = prev
            if prev is None:
                prev = nxt
            if rng.random() < p_dead_visible:
                kind = rng.random()
                L1 = int(min(L * rng.uniform(0.2, 0.7), rng.uniform(25, 150)))
                L0 = int(min(L - L1, rng.uniform(0, 75))) if rng.random() < 0.5 else 0
                w = np.concatenate([np.zeros(L0), np.linspace(0, 1, max(L1, 1)),
                                    np.ones(max(L - L0 - L1, 0))])[:L]
                ball[a:b] = prev[None] * (1 - w[:, None]) + nxt[None] * w[:, None]
                if kind < 0.2:
                    # lingers where it died, is lost, and is found again shortly before the restart
                    ball[a:b] = prev[None]
                    c = a + int(rng.uniform(0, min(L, 100)))
                    e = b - int(rng.uniform(0, 25))
                    ball[e:b] = nxt[None]
                    hide[c:max(c, e)] = True
                elif kind < 0.4 and L >= 30:
                    # carried around: a slow wander on top of the drift to the restart spot
                    step = rng.normal(0, 0.06, (L, 2)).cumsum(axis=0)
                    step -= np.linspace(0, 1, L)[:, None] * step[-1]
                    k = np.ones(13) / 13
                    step = np.column_stack([np.convolve(step[:, i], k, mode="same") for i in (0, 1)])
                    ball[a:b] += step / np.array(PITCH)
                elif rng.random() < 0.3:
                    c = rng.integers(a, b)
                    hide[c:min(b, c + int(rng.uniform(25, 200)))] = True
            else:
                ball[a:b] = nxt[None]
                a2 = a + int(rng.uniform(-25, 75))
                b2 = b - int(rng.uniform(0, 25))
                if rng.random() < 0.25:
                    b2 = b + int(rng.uniform(0, 100))        # found again only after the restart
                hide[max(h0, a2):max(a2 + 1, min(h1, b2))] = True
        # long losses during live play
        alive = ~dead
        target = alive_miss_frac * alive.sum()
        done = guard = 0
        while done < target and guard < 10000:
            guard += 1
            L = int(np.clip(rng.lognormal(np.log(150), 0.8), 20, 1500))
            c = rng.integers(h0, h1 - L)
            if hide[c:c + L].any():
                continue
            hide[c:c + L] = True
            done += alive[c - h0:c - h0 + L].sum()
        for _ in range(int((h1 - h0) / 25 / 60 * 0.3)):      # a few short blips
            c = rng.integers(h0, h1 - 10)
            hide[c:c + rng.integers(1, 8)] = True
    ball = ball + rng.normal(0, ball_jitter, ball.shape) / np.array(PITCH)
    ball[hide] = np.nan
    N = P.shape[1]
    for _ in range(int(track_drop * T * 22 / 150)):          # tracks vanish for a few seconds
        j = rng.integers(0, N)
        c = rng.integers(0, T)
        P[c:c + int(rng.exponential(150)) + 5, j] = np.nan
    P = P + rng.normal(0, jitter, P.shape) / np.array(PITCH)
    return dict(name=g["name"], P=P, team=g["team"], ball=ball, halves=g["halves"], labels=g["labels"])


def swap_teams(g):
    """Mirror augmentation: swap home and away and rotate the pitch, so nothing learned can
    depend on which side is called home or on the direction of play."""
    lab = g["labels"].copy()
    h, a = lab == 0, lab == 1
    lab[h] = 1
    lab[a] = 0
    return dict(name=g["name"] + "_sw", P=1.0 - g["P"], team=1 - g["team"], ball=1.0 - g["ball"],
                halves=g["halves"], labels=lab)
