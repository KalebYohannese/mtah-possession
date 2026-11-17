"""Frame features from anonymous tracking (players with team only, ball that is often missing)."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN windows are expected
FPS = 25

def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)

def _wmean(x, w, valid=None):
    """centred window mean over +-w frames ignoring NaN / invalid; returns (mean, valid_fraction)."""
    if valid is None: valid = ~np.isnan(x)
    xs = np.where(valid, x, 0.0)
    c = np.concatenate(([0.0], np.cumsum(xs))); n = np.concatenate(([0.0], np.cumsum(valid)))
    i = np.arange(len(x)); lo = np.maximum(0, i - w); hi = np.minimum(len(x), i + w + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (c[hi] - c[lo]) / cnt
    m[cnt == 0] = np.nan
    return m, cnt / (hi - lo)

def _wmax(x, w):
    """centred window max ignoring NaN (O(T log w) via doubling)."""
    y = np.where(np.isnan(x), -np.inf, x); out = y.copy(); T = len(y)
    for s in range(1, w + 1):
        out[s:] = np.maximum(out[s:], y[:-s]); out[:-s] = np.maximum(out[:-s], y[s:])
    out[np.isinf(out)] = np.nan
    return out

def _since(flag, cap):
    """seconds since flag was last true (cap if never), and seconds until next true."""
    T = len(flag); idx = np.arange(T)
    last = np.maximum.accumulate(np.where(flag, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(flag, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum((idx - last) / FPS, cap), np.minimum((nxt - idx) / FPS, cap)

def _ffill(x):
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0); np.maximum.accumulate(idx, out=idx)
    out = x[idx]; out[: np.argmax(~np.isnan(x))] = np.nan if np.isnan(x).all() or np.isnan(x[0]) else out[0]
    return out

def _bfill(x): return _ffill(x[::-1])[::-1]

def player_speed(P, pitch, k=10):
    """Speed (m/s) of every track slot from its displacement over +-k frames (0.8 s at k=10).

    A long baseline on purpose: frame-to-frame differences of automatic tracking are dominated by
    jitter. Displacements that are physically impossible (> 11 m/s) are identity switches of the
    slot, not running, and are dropped."""
    T = len(P); S = np.full(P.shape[:2], np.nan)
    if T > 2 * k:
        d = P[2 * k:] - P[:-2 * k]
        S[k:T - k] = np.hypot(d[..., 0] * pitch[0], d[..., 1] * pitch[1]) * FPS / (2 * k)
    S[S > 11.0] = np.nan
    return S

def half_features(P, team, ball, pitch):
    """features for one half (contiguous block). P (T,N,2) normalised 0-1, ball (T,2); NaN = missing."""
    T = len(ball); Lx, Ly = pitch; F = {}
    S = player_speed(P, pitch)
    isH = team == 0; isA = team == 1
    X = P[..., 0] * Lx; Y = P[..., 1] * Ly
    present = ~np.isnan(X)
    # which way home attacks: home's average position sits on its own side of the opponents'
    with np.errstate(invalid="ignore"):
        mxH = np.nanmean(np.where(isH[None], X, np.nan), axis=1); mxA = np.nanmean(np.where(isA[None], X, np.nan), axis=1)
    sgn = 1.0 if np.nanmean(mxH - mxA) < 0 else -1.0      # +1: home attacks towards x = Lx
    Xn = X if sgn > 0 else Lx - X
    bx = ball[:, 0] * Lx; by = ball[:, 1] * Ly
    bxn = bx if sgn > 0 else Lx - bx
    vis = ~np.isnan(bx)
    # --- ball visibility structure
    F["vis"] = vis.astype(float)
    s_, e_ = _runs(~vis); gap = np.zeros(T)
    for a, b in zip(s_, e_): gap[a:b] = (b - a) / FPS
    F["gap_len"] = np.minimum(gap, 120)
    s_, e_ = _runs(vis); vr = np.zeros(T)
    for a, b in zip(s_, e_): vr[a:b] = (b - a) / FPS
    F["vis_run"] = np.minimum(vr, 120)
    F["t_since_seen"], F["t_until_seen"] = _since(vis, 120)
    for w in (25, 125, 375): F[f"visfrac_{w}"] = _wmean(vis.astype(float), w, np.ones(T, bool))[0]
    # --- ball kinematics
    bs = np.full(T, np.nan); k = 2
    d = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) * FPS / (2 * k)
    bs[k:T - k] = d; bs = np.minimum(bs, 40.0)
    F["bspeed"] = bs
    for w in (12, 50): F[f"bspeed_m{w}"] = _wmean(bs, w)[0]
    F["bspeed_x50"] = _wmax(bs, 50)
    edge = np.minimum(np.minimum(bx, Lx - bx), np.minimum(by, Ly - by))
    F["edge"] = edge; F["edge_ff"] = _ffill(edge); F["edge_bf"] = _bfill(edge)
    F["edge_min25"] = -_wmax(-edge, 25)
    F["bxn_ff"] = _ffill(bxn); F["bxn_bf"] = _bfill(bxn)
    F["bspeed_ff"] = _ffill(F["bspeed_m12"]); F["bspeed_bf"] = _bfill(F["bspeed_m12"])
    # --- nearest player of each team
    dH = np.full(T, np.nan); dA = np.full(T, np.nan); n2H = np.full(T, np.nan); n2A = np.full(T, np.nan)
    vi = np.flatnonzero(vis)
    for c0 in range(0, len(vi), 20000):
        ii = vi[c0:c0 + 20000]
        dd = np.hypot(X[ii] - bx[ii, None], Y[ii] - by[ii, None]); dd = np.where(np.isnan(dd), 1e3, dd)
        dh = np.sort(dd[:, isH], axis=1); da = np.sort(dd[:, isA], axis=1)
        dH[ii] = dh[:, 0]; dA[ii] = da[:, 0]
        n2H[ii] = (dh < 5).sum(1); n2A[ii] = (da < 5).sum(1)
    dH = np.minimum(dH, 60); dA = np.minimum(dA, 60)
    F["dH"] = dH; F["dA"] = dA; F["dmin"] = np.minimum(dH, dA)
    diff = np.clip(dA - dH, -15, 15); F["ddiff"] = diff                 # >0 : home nearer
    F["nH5"] = n2H; F["nA5"] = n2A
    nearH = np.where(vis, (dH <= dA).astype(float), np.nan)
    for w in (5, 12, 25, 50, 125, 250):
        F[f"voteH_{w}"] = _wmean(nearH, w)[0]
    for w in (12, 50): F[f"ddiff_m{w}"] = _wmean(np.clip(diff, -4, 4), w)[0]
    F["dmin_m12"] = _wmean(F["dmin"], 12)[0]; F["dmin_m50"] = _wmean(F["dmin"], 50)[0]
    with np.errstate(invalid="ignore"):
        cH = vis & (dH < 1.5) & (dH < dA); cA = vis & (dA < 1.5) & (dA < dH)
    for nm, c in (("H", cH), ("A", cA)):
        F[f"ts_c{nm}"], F[f"tn_c{nm}"] = _since(c, 30)
        for w in (12, 25, 50, 125): F[f"c{nm}_{w}"] = _wmean(c.astype(float), w, np.ones(T, bool))[0]
    F["ts_cdiff"] = F["ts_cA"] - F["ts_cH"]; F["tn_cdiff"] = F["tn_cA"] - F["tn_cH"]
    # sustained control: who had the last / has the next control stretch of >= 0.4 s
    for nm, c in (("H", cH), ("A", cA)):
        s_, e_ = _runs(c); long = np.zeros(T, bool)
        for a, b in zip(s_, e_):
            if b - a >= 10: long[a:b] = True
        F[f"ts_lc{nm}"], F[f"tn_lc{nm}"] = _since(long, 30)
    F["ts_lcdiff"] = F["ts_lcA"] - F["ts_lcH"]; F["tn_lcdiff"] = F["tn_lcA"] - F["tn_lcH"]
    # --- player kinematics (available on every frame)
    with np.errstate(invalid="ignore", all="ignore"):
        med = np.nanmedian(S, axis=1); q75 = np.nanpercentile(S, 75, axis=1)
        slow = np.nanmean(np.where(present, S < 1.5, np.nan), axis=1); fast = np.nanmean(np.where(present, S > 4.5, np.nan), axis=1)
        mH = np.nanmean(np.where(isH[None], S, np.nan), axis=1); mA = np.nanmean(np.where(isA[None], S, np.nan), axis=1)
    F["sp_med"] = med; F["sp_q75"] = q75; F["sp_slow"] = slow; F["sp_fast"] = fast; F["sp_HmA"] = mH - mA
    for w in (25, 75, 200):
        F[f"sp_med_{w}"] = _wmean(med, w)[0]; F[f"sp_slow_{w}"] = _wmean(slow, w)[0]; F[f"sp_q75_{w}"] = _wmean(q75, w)[0]
    F["sp_HmA_50"] = _wmean(mH - mA, 50)[0]
    # --- team shape, in the frame where home attacks towards +x
    with np.errstate(invalid="ignore", all="ignore"):
        XH = np.where(isH[None], Xn, np.nan); XA = np.where(isA[None], Xn, np.nan)
        YH = np.where(isH[None], Y, np.nan); YA = np.where(isA[None], Y, np.nan)
        cxH = np.nanmean(XH, 1); cxA = np.nanmean(XA, 1)
        F["cxH"] = cxH; F["cxA"] = cxA; F["cx_gap"] = cxA - cxH
        F["sxH"] = np.nanstd(XH, 1); F["sxA"] = np.nanstd(XA, 1); F["syH"] = np.nanstd(YH, 1); F["syA"] = np.nanstd(YA, 1)
        F["sx_d"] = F["sxH"] - F["sxA"]; F["sy_d"] = F["syH"] - F["syA"]
        F["cy_d"] = np.nanmean(YH, 1) - np.nanmean(YA, 1)
    for nm, c in (("H", cxH), ("A", cxA)):
        cs = _wmean(c, 6)[0]; v = np.full(T, np.nan); v[12:T - 12] = (cs[24:] - cs[:-24]) / (24 / FPS)
        F[f"vcx{nm}"] = v; F[f"vcx{nm}_50"] = _wmean(v, 50)[0]
    F["sy_d_50"] = _wmean(F["sy_d"], 50)[0]; F["sx_d_50"] = _wmean(F["sx_d"], 50)[0]
    names = list(F.keys())
    return np.column_stack([F[n] for n in names]).astype(np.float32), names

def build(P, team, ball, halves, pitch=(105.0, 68.0)):
    """Feature matrix for the graded frames. halves = [(start, stop), ...] half-open frame blocks;
    every window is computed inside one block, never across the interval."""
    Xs, idx = [], []
    for a, b in halves:
        f, names = half_features(P[a:b], team, ball[a:b], pitch)
        Xs.append(f); idx.append(np.arange(a, b))
    return np.vstack(Xs), np.concatenate(idx), names
