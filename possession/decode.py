"""Turn per-frame class probabilities into possessions: a Viterbi path with switching costs."""
import numpy as np


def viterbi(logp, c_team, c_dead):
    """Most likely H/A/D path for log-probabilities (T,3), paying c_team to hand the ball to the
    other team and c_dead to stop or restart play. The costs are what remove flicker: a switch has
    to be supported by several frames of evidence before it is worth paying for."""
    T = len(logp)
    C = np.array([[0, c_team, c_dead], [c_team, 0, c_dead], [c_dead, c_dead, 0]], float)
    score = logp[0].astype(float).copy()
    back = np.zeros((T, 3), np.int8)
    ar = np.arange(3)
    for t in range(1, T):
        cand = score[:, None] - C
        b = np.argmax(cand, axis=0)
        back[t] = b
        score = cand[b, ar] + logp[t]
    path = np.zeros(T, np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(proba, block_lengths, c_team=10.0, c_dead=12.0):
    """Decode each graded block (half) independently."""
    lp = np.log(np.clip(proba, 1e-6, 1.0))
    out, o = [], 0
    for L in block_lengths:
        out.append(viterbi(lp[o:o + L], c_team, c_dead))
        o += L
    return np.concatenate(out)
