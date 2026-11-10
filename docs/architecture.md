# Architecture

Each stage is one module, so most PRs touch one file plus its tests.

| Stage | Module | Input | Output |
|---|---|---|---|
| Load | `io.py` | `$MTAH_DATA_DIR` | `Match` |
| Clean ball | `ball.py` | `Match` | `BallTrack` |
| Dead-ball score | `deadball.py` | `Match`, `BallTrack` | Score per frame |
| Evidence and decoder | `possession.py` | `Match`, `BallTrack`, score | `H`/`A`/`D` per frame |
| Final mask and write | `pipeline.py` | Labels | `outputs/Falafelona.csv` |

`vendor.py` gives access to the official code, and `config.py` loads `configs/*.yaml`.

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
predict(match, ball, dead_score, cfg) -> labels: str[n] in {H, A, D}

# pipeline.py
run(cfg) -> writes outputs/Falafelona.csv, returns {sha256, counts}
```

## Baseline path

`configs/baseline.yaml` reproduces the official `baseline_smooth50`: no ball cleaning, dead wherever the ball is unseen, then a centred majority vote. It is the check that the pipeline is correct before anything else is added.

## X and the second half

The `X` policy is not agreed yet. D14 proposes that `pipeline.py` writes `X` only on ungraded frames, using the official `games.half_video_blocks`.

The second half is graded from frame 110,808, before its kickoff, and those frames are dead ball. Detect the kickoff from the `Kick Off (Start)` Smart Tag rather than hard-coding a frame.
