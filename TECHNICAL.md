# Technical document — Sportify

## Approach

The task is treated as sequence labelling in two stages.

1. **Frame classifier.** For every graded frame, 87 features computed from the tracking go into a
   gradient-boosted tree model that outputs P(home), P(away), P(dead).
2. **Possession decoder.** A Viterbi path through those probabilities, paying a fixed cost to hand
   the ball to the other team and a fixed cost to stop or restart play. The costs remove
   flicker: a change of state must be supported by several frames of evidence.

Why this rather than nearest-player plus a majority vote: the ranking metric counts whole
possessions, so the two things that matter are _where the ball is dead_ and _exactly when
possession changes_. A majority vote can express neither — it has no dead-ball concept beyond
"ball not tracked" and it moves every boundary by half its window.

Why a learned classifier rather than hand-set thresholds: thresholds on distances and speeds would
have had to be tuned on the provided match, which is not allowed. Learning them on other matches
is.

## Assets used

| asset                                                          | use                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| tactical ATD (`_atd.txt`, `_atd.xml`)                          | everything: player positions with team, ball position |
| Smart Tagging                                                  | kick-offs only (see rules below)                      |
| tactical video, homography, broadcast video / ATD / homography | **not used**                                          |

The broadcast recording was dropped because it is not synchronised to the tactical timeline. Most
Smart Tagging phases were dropped because their intervals are phases rather than possessions and
there was no other-match tagging to learn their reliability from. The speed column of the ATD is
not read: speeds are recomputed from positions so that the same code runs on the training games.

## Handling missing and noisy tracking

- **Ball missing.** The ball is lost in long runs, not frame by frame. Features describe the gap
  itself (its length, time since / until the ball is seen, where and how fast the ball was before
  and after) and the players, who are tracked throughout: median and upper-quartile speed, share
  of slow players, team centroids, their velocities and spreads, in a frame where home attacks
  towards +x. The model therefore still has evidence for dead-vs-alive and for the team in
  possession while the ball is gone.
- **No identities, broken tracks.** All features are collective (nearest player of each team,
  team aggregates). Player speed is the displacement of a slot over 0.8 s, which is robust to
  jitter; displacements above 11 m/s are treated as identity switches and ignored.
- **Team symmetry.** Every training game is also used with home and away swapped and the pitch
  rotated, so the model cannot learn a side or a direction of play.

## Training data

**Metrica Sports public sample data** — three anonymised matches, corrected tracking at 25 fps
with synchronised events (https://github.com/metrica-sports/sample-data). Labelled by Metrica; no
labels were produced by us and none concern the provided match.

Labels are derived mechanically, mirroring how the challenge describes its ground truth: a frame
is `D` where the corrected tracking has no ball (it is absent exactly when play is dead);
otherwise it takes the team of the most recent on-ball event (pass, carry, shot, recovery, set
piece, ball lost, ball out).

Because those games are corrected tracking and the challenge data is not, each game is degraded
before training (`training/simulate.py`): the ball is hidden in long runs during live play (10% and
30% of live frames in the two training variants); roughly half of the stoppages keep a visible
ball that rests at the restart spot, lingers where it died or is carried, and the rest lose it;
player tracks drop out for seconds at a time; positions get 0.08–0.30 m of jitter. The statistics
this imitates (ball seen on 59% of graded frames, lost in ~150 long runs) come from `DATA.md` and
from counting the organisers' committed `baseline_raw.csv`, which is derived from the ATD alone.

430k training rows (every 4th frame). Model: scikit-learn `HistGradientBoostingClassifier`,
250 iterations, 48 leaves, exported to plain arrays so that inference needs only numpy.

## Possession rules implemented

- Three states only; a possession is a maximal run of one team with the ball alive.
- **Loose and contested balls stay with the last team**: this is what the training labels encode
  (possession changes at the other team's recovery, not at a touch or a duel) and what the
  decoder's switching cost enforces.
- **Dead ball** is learned, not "ball not tracked": the cues are players slowing down
  collectively, the ball resting or lost near a boundary, and where the ball reappears.
- **Kick-offs** (the only hand-written rule), taken from the Smart Tagging `Kick Off` set pieces:
  - _Start of a half._ Play is dead from the start of a graded half until its tagged kick-off. In
    practice this applies to the second half, which is graded from frame 110808 but kicks off
    about two minutes later. The first half is graded from its own kick-off: its tag window begins
    before the graded block, so the rule skips it and nothing is forced there.
  - _After a goal._ The 15 s before each `Kick Off (After Goal)` are forced dead (a goal is never
    restarted faster than that); the rest of the stoppage is left to the model. The `Goals` tags
    are phases that run up to the restart rather than the instant of the goal, so they cannot mark
    where the stoppage begins. (The code would extend the dead stretch back to a goal tag that
    ends before the kick-off, but on this match none does, so that branch never applies.)
  - The kick-off instant inside the tag window is taken from the ball leaving the centre spot,
    erring early; frames after it are left to the model.
- `X` is written only outside the graded blocks.

## Validation

All choices (features, model size, decoder costs 10 / 12) were made on the sample games, by
holding one game out and scoring it with the organisers' `possession_metrics.py`:

| held-out sample game 3, ball lost on … of live frames | macroF1 | spellF1 |
| ----------------------------------------------------- | ------- | ------- |
| 5%                                                    | 0.923   | 0.680   |
| 20%                                                   | 0.918   | 0.659   |
| 35%                                                   | 0.895   | 0.561   |

Under the same simulation the organisers' `smooth50` baseline scores about 0.23–0.31 spellF1,
close to the 0.2433 it scores on the real match, which is the evidence that the simulation is a
reasonable stand-in. The public five minutes of the provided match were used once, after the
method was frozen, to confirm that frames and labels are aligned — not to choose anything.

## Limitations

- The numbers above are on simulated degradation of other matches. The real ATD will differ in
  ways the simulation does not capture (where and why the ball is lost, wrong team assignments,
  duplicated tracks), so the real score will be lower.
- Training labels come from events; the challenge ground truth comes from corrected tracking.
  The two may place a change of possession a few frames apart.
- Very short possessions (under about a second) are mostly missed, and each miss also merges its
  two neighbours.
- During long ball losses in live play the team is inferred from context and team shape only.
- Goal stoppages are only partly covered by the kick-off rule: the last 15 s before the restart
  are forced dead, and the celebration before that is left to the model. If it reads part of a
  celebration as live play, that produces short false possessions.
- Only three training matches, all from one provider.

With more time: use the video around ball losses, learn the reliability of Smart Tagging set
pieces, and replace the frame classifier with a sequence model.

## Reproduction

See `README.md`: `pip install -r requirements.txt`, then
`python predict.py --data <bundle> --out results/Sportify.csv`.
