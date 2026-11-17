# Architecture

Each stage is one module, so most PRs touch one file plus its tests.

| Stage | Module | Input | Output |
|---|---|---|---|
| Load | `io.py` | `$MTAH_DATA_DIR` | `Match` |
| Clean ball | `ball.py` | `Match` | `BallTrack` |
| Dead-ball score | `deadball.py` | cleaned `Match`, `BallTrack` | Score per frame |
| Evidence and decoder | `possession.py` | cleaned `Match` (ball already replaced by the cleaned track), score | `H`/`A`/`D` per frame |
| Final mask and write | `pipeline.py` | Labels | `outputs/Falafelona.csv` |
| Motion features | `velocity.py` | positions, `period`, `fps`, config block | ball velocity and player speed per frame (m/s) |
| Learned dead-ball score | `learned_dead.py` | cleaned `Match`, a fitted model file in `models/` | chance per frame that the ball is dead: one clue of the dead-ball score (D24, v3: D31) |
| Learned touch team | `learned_touch.py` | cleaned `Match`, the detected touches, `models/touch_*.json` | which team has the ball after each touch, as H / A costs (D32) |
| Learned team in gaps | `learned_team.py` | cleaned `Match`, `models/team_*.json` | chance that home has the ball on each gap frame, as H / A costs (D29) |
| Learned switch hazard | `learned_switch.py` | cleaned `Match`, a model file | lower switch cost where a change looks likely; in the code, **off** (D34) |
| Dev set | `devset.py` | `$IDSSE_DATA_DIR` | `Match` plus truth, scores per config |

`vendor.py` gives access to the official code, and `config.py` loads `configs/*.yaml`.

`config.py` also refuses any setting name that no stage reads, so a misspelt key cannot switch a feature off silently.

`pipeline.py` runs `ball.DataCleaner` first and passes the cleaned `Match` on: `ball_xy` in it is already the cleaned track, so `possession.py` does not take the `BallTrack` separately.

## Interfaces

```python
# io.py
Match:
  n_frames: int                # 199,319
  fps: float                   # from metadata
  period: int8[n]              # 1, 2, or 0 = ungraded
  ball_xy: float[n, 2]         # metres, NaN where missing
  players_xy: float[n, 46, 2]  # metres, NaN where slot empty
  slot_team: str[46]           # 'H' / 'A' / '?'  (slots are tracklets, not players)
  tags: list[Event]            # Smart Tagging, sorted: start, end (exclusive), code, type,
                               #   team ('H' / 'A' / None for N/A), half; evidence, not truth
  homography: float[n, 3, 3]   # normalised pitch (0-1) -> pixels; divide metres by pitch first
  homography_error: float[n]   # reprojection error, lower is better
  homography_valid: bool[n]    # matrix is NaN where False

select_tags(tags, code=, type=, team=) -> list[Event]  # code/type case-insensitive

# ball.py
clean(match, cfg) -> BallTrack:
  xy: float[n, 2]              # NaN where still unknown
  source: int8[n]              # OBSERVED / INTERPOLATED / REJECTED / UNKNOWN
  speed: float[n]

# deadball.py
detect(match, ball, cfg) -> dead_score: float[n]     # 0..1

# possession.py
predict(match, dead_score, cfg) -> labels: str[n] in {H, A, D}   # match = the cleaned Match

# velocity.py   (NaN = no evidence; windows in seconds from a config block)
ball_velocity(xy, period, fps, section) -> vxy: float[n, 2], speed: float[n]     # m/s
player_speed(xy, period, fps, section)  -> speed per slot and frame               # m/s

# pipeline.py
run(cfg) -> writes outputs/Falafelona.csv, returns {sha256, counts}

# devset.py
load_idsse(match_id) -> DevMatch(match_id, match: Match, truth: str[n] in {H, A, D, X})
degrade_ball(match, section, seed) -> (Match, report)   # ball gaps like ours, fixed seed
run_devset(configs, dev_cfg, matches) -> spellF1 / macroF1 per config and match
```

## What `configs/final.yaml` runs

1. **Ball:** `interpolate`: straight-line fill of gaps up to 0.5 s.
2. **Dead-ball score:** `combined`: `before_kickoff` 1.0, `out_of_bounds` 1.0, `before_restart` 0.6, `still_ball_at_spot` 0.6, and the blend `gap_chance` × 0.3 + `learned` × 0.7 with the v3 model (D24, D31). Capped at 1.
3. **Possession:** `hmm`: a Viterbi decoder over H / A / D, costs 10 / 18 / 18 (D26), with distance-margin evidence moved 24 frames later, the fast-ball rule, restart-taker evidence (D20), touch evidence scored by the touch model (D28, D32) and the gap team model inside ball gaps (D29). Then switch placement 1.0 (D26) and the 1 s minimum spell (D30).
4. **Output:** one label per frame, no `X` (D14).

The technical document, [Falafelona.md](Falafelona.md), explains each part and why.

## Scripts

| Script | What it does | Reads |
|---|---|---|
| `scripts/devset_eval.py` | scores configs on the dev set with the official metric; `--oracle` gives one part of the pipeline the truth (dead ball, touches, team in gaps, switch timing) and `--diagnose` explains where the score is lost | IDSSE (humans only) |
| `scripts/fit_deadball.py` | fits the learned dead-ball models and writes them to `models/` | IDSSE (humans only) |
| `scripts/touch_lomo.py` | leave-one-match-out check of a config picked from a sweep (D25) | the per-match scores written by `devset_eval.py --scores-json` |
| `scripts/label_free_check.py` | builds the file from a config and prints counts only (label shares, possessions, kickoff, tags, sha256) | our match (humans only) |
| `scripts/compare_submissions.py` | compares two possession files, aggregates only | two CSV files |
| `scripts/ball_eda.py`, `scripts/dead_eda.py` | aggregate statistics of the ball gaps and of the dead-ball clues | our match (humans only) |
| `scripts/metric_study.py` | how spellF1 reacts to flicker and late switches, on synthetic labels | nothing |
| `scripts/bootstrap_eval.py` | checks a submission file: the public five-minute pipeline check with a bootstrap interval (never used to choose, D9) and label-free counts | our outputs (humans only) |
| `scripts/fit_team.py`, `scripts/fit_touch.py`, `scripts/fit_switch.py` | fit the gap team, touch and switch-hazard models and write them to `models/` | IDSSE (humans only) |
| `scripts/cost_sweep.py` | writes a grid of decoder-cost configs and checks the winner leave-one-match-out | the per-match scores of `devset_eval.py` |
| `scripts/team_diag.py` | label-free look at the team model's inputs and outputs, on our match or on IDSSE; `--smooth-ball` compares touch statistics | our match or IDSSE (humans only) |
| `scripts/window_check.py` | how often a 5-minute window of a real match is as calm as the public five minutes | IDSSE (humans only) |
| `scripts/camera_aim_check.py`, `scripts/broadcast_sync_check.py`, `scripts/tag_team_check.py` | label-free checks of the homography, the broadcast tracking and the Smart Tagging phases as gap evidence (not used, D34) | our match (humans only) |
| `scripts/check_no_data.py` | refuses data-shaped files in the repo; runs in the tests and the git hooks | the repo |

## Baseline path

`configs/baseline.yaml` reproduces the official `baseline_smooth50`: no ball cleaning, dead wherever the ball is unseen, then a centred majority vote. It is the check that the pipeline is correct before anything else is added.

## Dev set

We have no ground truth for our own match, so versions are compared on other matches (D15), starting with the seven IDSSE matches. `devset.py` turns an IDSSE match into the same `Match` the stages read (metres, 25 fps, 46 slots, pitch from the match information) and builds the truth from the file's own `BallStatus` and `BallPossession`. The ball of IDSSE is almost complete, ours is not, so `degrade_ball` cuts gaps into it with the gap lengths measured on our match (`configs/devset.yaml`, from the ball EDA, issue #13) and a fixed seed. The gaps are placed in two ways (D17): `random` (anywhere) and `stoppage` (starting as the ball goes dead); every comparison runs under both (`make devset` and `make devset GAPS=stoppage`). The stages then run exactly as in `pipeline.run`, and the labels are scored with the official `possession_metrics`. With `configs/devset-smooth.yaml` the IDSSE ball is first smoothed (sigma 0.25 s), so its kicks look like ours (D27). The halves are placed one after the other with an ungraded pad between them, as the real halftime hole does, so a vote near a half edge never sees the other half.

`scripts/devset_eval.py` is the entry point and `experiments/README.md` says how to run it. The dev score compares versions; it does not predict the real score (simulated gaps, the provider's possession flag as truth, no tracklet swaps).

## X and the second half

No `X` is written (D14): `x_policy: none`. Ungraded frames are never scored, and on graded frames `X` cannot gain on spellF1 and always misses on macroF1.

The second half is graded from frame 110,808, before its kickoff, and those frames are dead ball. Detect the kickoff from the `Kick Off (Start)` Smart Tag rather than hard-coding a frame.