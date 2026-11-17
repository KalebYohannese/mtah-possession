# Ball possession from automatic tracking — More than a Hack 2027, Phase 1

Per-frame `H` / `A` / `D` labels for the tactical video, produced from the provided tracking (ATD)
and, for kick-offs only, the Smart Tagging. No video, no hand labels, no data about the match
other than the provided bundle.

**How it works, in three lines.** A gradient-boosted classifier reads ~90 features of each frame
(who is near the ball, who controlled it last and next, how fast the players move, how long the
ball has been lost) and gives a probability for home / away / dead. A Viterbi pass with switching
costs turns those probabilities into possessions. The classifier is trained only on Metrica's
three _public sample games_, degraded to look like automatic tracking. Details: `TECHNICAL.md`.

## Reproduce the submission

```bash
pip install -r requirements.txt            # numpy, nothing else
python predict.py --data /path/to/data --out results/Sportify.csv
```

`--data` is the unpacked bundle (the folder containing `tactical/`); `MTAH_DATA_DIR` works too.
The run takes a few minutes and is deterministic: same data, same file.

Then, from the organisers' repository:

```bash
python evaluation/validate_submission.py results/Sportify.csv
```

## Retrain the model (optional)

`models/model.npz` is committed, so this is not needed to reproduce the file.

```bash
pip install -r requirements-train.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data training/sample-data
python -m training.train                   # writes models/model.npz
python -m training.train --holdout 2       # run twice: fit without game 3, then score it
```

The hold-out mode scores with the organisers' `possession_metrics.py`; point `MTAH_EVAL_DIR` at
their `evaluation/` folder.

## Layout

```
predict.py                 tracking in, submission out
possession/atd.py          reads the EPTS tracking files
possession/features.py     frame features
possession/gbm.py          numpy-only inference for the trained trees
possession/decode.py       Viterbi smoothing into possessions
possession/tagging.py      kick-off rule from the Smart Tagging
training/                  sample-game loader, ATD-like degradation, training script
models/model.npz           the trained model
```

The match data is not in this repository and must not be added to it (`.gitignore` covers
`data/`, `results/` and every `*.csv`).

Training data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
