"""Tracking in, submission out.

    python predict.py --data /path/to/data --out results/Sportify.csv

--data is the unpacked bundle (the folder that contains tactical/). Needs numpy only.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession import atd, decode, features, gbm, tagging  # noqa: E402

# Published in the challenge's DATA.md / games.py: frame count of the tactical video, the two
# graded blocks (half-open), and the real pitch size. Boundaries, not labels.
N_FRAMES = 199319
BLOCKS = ((11185, 90652), (110808, 190198))
PITCH = (105.3, 68.0)
STEM = "BAR-ATH_tactical"
LETTERS = np.array(["H", "A", "D", "X"])


def run(data_dir, model_path, out_path, use_tagging=True, c_team=10.0, c_dead=12.0,
        blocks=BLOCKS, n_frames=N_FRAMES, stem=STEM):
    t0 = time.time()
    folder = os.path.join(data_dir, "tactical")
    meta = os.path.join(folder, stem + "_atd.xml")
    raw = os.path.join(folder, stem + "_atd.txt")
    tags = os.path.join(folder, stem + "_smart_tagging.xml")
    print("reading tracking ...", flush=True)
    P, ball, team = atd.read(meta, raw, n_frames)
    print("  {} frames, {} track slots ({} home / {} away)".format(
        len(P), len(team), int((team == 0).sum()), int((team == 1).sum())))
    graded = np.zeros(len(P), bool)
    for a, b in blocks:
        graded[a:b] = True
    print("  ball seen on {:.1%} of graded frames; {:.1f} players tracked per graded frame".format(
        (~np.isnan(ball[graded, 0])).mean(), (~np.isnan(P[graded, :, 0])).sum(1).mean()))

    print("features ...", flush=True)
    X, idx, names = features.build(P, team, ball, blocks, PITCH)
    model = gbm.Model(model_path)
    if names != model.names:
        raise ValueError("feature list does not match the model file")
    print("classifying {} frames ...".format(len(X)), flush=True)
    proba = model.predict_proba(X)

    if use_tagging and os.path.isfile(tags):
        forced = tagging.dead_intervals(tagging.load(tags), ball, blocks, PITCH)
        pos = np.full(len(P), -1)
        pos[idx] = np.arange(len(idx))
        for a, b in forced:
            rows = pos[a:b]
            rows = rows[rows >= 0]
            proba[rows] = (1e-6, 1e-6, 1.0)
        print("  kick-off rule: {} dead stretches, {:.0f} s in total".format(
            len(forced), sum(b - a for a, b in forced) / 25.0))
    elif use_tagging:
        print("  no Smart Tagging file found -- kick-off rule skipped")

    path = decode.decode(proba, [b - a for a, b in blocks], c_team, c_dead)
    labels = np.full(len(P), 3, np.int8)          # X outside the graded blocks: never scored
    labels[idx] = path

    folder_out = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder_out, exist_ok=True)
    with open(out_path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, s) for i, s in enumerate(LETTERS[labels]))

    g = labels[graded]
    spells = 0
    for a, b in blocks:
        seg = labels[a:b]
        starts = np.flatnonzero(np.concatenate(([True], seg[1:] != seg[:-1])))
        spells += int((seg[starts] < 2).sum())
    alive = int((g < 2).sum())
    print("\nwrote {}  ({} rows, {:.0f} s)".format(out_path, len(labels), time.time() - t0))
    print("graded frames: H {:.1%}  A {:.1%}  D {:.1%}".format((g == 0).mean(), (g == 1).mean(), (g == 2).mean()))
    print("possessions: {}   mean duration {:.1f} s".format(spells, alive / 25.0 / max(spells, 1)))
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", os.path.join(HERE, "data")))
    ap.add_argument("--model", default=os.path.join(HERE, "models", "model.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "submission.csv"))
    ap.add_argument("--no-tagging", action="store_true", help="skip the Smart Tagging kick-off rule")
    args = ap.parse_args()
    run(args.data, args.model, args.out, use_tagging=not args.no_tagging)
