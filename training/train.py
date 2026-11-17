"""Train the frame classifier on Metrica's public sample games and export it to models/model.npz.

    git clone --depth 1 https://github.com/metrica-sports/sample-data training/sample-data
    python -m training.train                 # full model (all three games)
    python -m training.train --holdout 2     # leave one game out and score it with the official metric

Labels come from those games' own corrected tracking and events (see sample_games.py) -- no data
about the challenge match is involved anywhere in training.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession import decode, features, gbm  # noqa: E402
from training import sample_games, simulate  # noqa: E402

# (seed, alive_miss_frac, p_dead_visible, player jitter m, ball jitter m, track drop)
TRAIN_VARIANTS = ((11, 0.10, 0.55, 0.08, 0.12, 0.05), (12, 0.30, 0.40, 0.25, 0.30, 0.10))
TEST_VARIANTS = ((101, 0.05, 0.5, 0.10, 0.15, 0.07), (102, 0.20, 0.5, 0.10, 0.15, 0.07),
                 (103, 0.35, 0.5, 0.20, 0.25, 0.10))
STEP = 4                    # keep every 4th frame: neighbouring frames are near-duplicates
GBM = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200,
           l2_regularization=1.0, random_state=0)


def variant(g, gi, v):
    seed, amf, pdv, jit, bjit, drop = v
    return simulate.degrade(g, np.random.default_rng(seed * 10 + gi), p_dead_visible=pdv,
                            alive_miss_frac=amf, jitter=jit, ball_jitter=bjit, track_drop=drop)


def xy(d):
    X, idx, names = features.build(d["P"], d["team"], d["ball"], d["halves"])
    return X, d["labels"][idx], names


def training_set(games, skip=None):
    Xs, ys, names = [], [], None
    for gi, g in enumerate(games):
        if gi == skip:
            continue
        for v in TRAIN_VARIANTS:
            d = variant(g, gi, v)
            for dd in (d, simulate.swap_teams(d)):
                X, y, names = xy(dd)
                Xs.append(X[::STEP])
                ys.append(y[::STEP])
            print("  features: game {} variant {}".format(gi + 1, v[0]), flush=True)
    return np.vstack(Xs), np.concatenate(ys), names


def official_scores(y, pred, block_lengths):
    sys.path.insert(0, os.environ.get("MTAH_EVAL_DIR", os.path.join(ROOT, "..", "evaluation")))
    import possession_metrics as pm
    cut = block_lengths[0]
    gap = np.full(10, 3, np.int8)                # an ungraded gap, so the halves stay separate
    r = pm.evaluate(np.concatenate([y[:cut], gap, y[cut:]]).astype(np.int8),
                    np.concatenate([pred[:cut], gap, pred[cut:]]).astype(np.int8))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=None, help="game index 0-2 to hold out and score")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "model.npz"))
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    games = sample_games.load_all()
    tmp = args.out.replace(".npz", "_holdout{}.npz".format(args.holdout))
    clf = None
    if args.holdout is None or not os.path.exists(tmp):
        X, y, names = training_set(games, skip=args.holdout)
        t0 = time.time()
        clf = HistGradientBoostingClassifier(**GBM).fit(X, y)
        print("fitted on {} rows x {} features in {:.0f} s".format(len(X), X.shape[1], time.time() - t0), flush=True)

    if args.holdout is None:
        gbm.export(clf, names, args.out)
        print("wrote", args.out)
    else:
        if clf is not None:
            gbm.export(clf, names, tmp)
            sys.exit(0)                          # run again to score (keeps each step short)
        m = gbm.Model(tmp)
        g = games[args.holdout]
        lengths = [b - a for a, b in g["halves"]]
        for v in TEST_VARIANTS:
            Xt, yt, _ = xy(variant(g, args.holdout, v))
            p = m.predict_proba(Xt)
            r = official_scores(yt, decode.decode(p, lengths), lengths)
            print("held-out game {}  ball lost on {:.0%} of live frames:  macroF1 {:.3f}  spellF1 {:.3f}  "
                  "possessions {} (truth {})".format(args.holdout + 1, v[1], r["macro_f1"], r["spell_f1_mean"],
                                                     r["n_pred_spells"], r["n_gt_spells"]), flush=True)
