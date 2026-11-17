# models/

Fitted numbers of the learned models (`src/falafelona/learned_dead.py`, `learned_team.py`,
`learned_touch.py`). Small JSON files: a few dozen coefficients each, no match data and no frame
values.

**What the submission uses (`configs/final.yaml`):**

| File | Model | Fitted on | Dev-set twins |
|---|---|---|---|
| `deadball_stoppage3_all.json` | dead ball v3 (`learned_dead.py`) | stoppage gaps, all seven matches (172,220 gap and 228,844 seen-ball frames) | `deadball_stoppage3_lomo_<match>.json` |
| `team_smooth5g_all.json` | team in gaps v5 (`learned_team.py`) | smoothed dev set, all seven matches (312,500 gap frames) | `team_smooth5g_lomo_<match>.json` |
| `touch_all.json` | touch team (`learned_touch.py`) | smoothed dev set, all seven matches (44,380 touches) | `touch_lomo_<match>.json` |

Each `*_lomo_<match>.json` is fitted on the other six matches; on the dev set a config points at
`models/..._lomo_{match_id}.json`, so every match is scored by a model that never saw it.

The other files are earlier versions and variants, not used in the submission (rows in
`experiments/RESULTS.md`): the dead-ball v2 files (`deadball_*`, `deadball_stoppage_*`, the v1.2
model, `deadball_random_*`, `deadball_nolen_*`, `deadball_smooth*_*`) and the earlier team models
(`team_*`, `team_smooth_*`, `team_smooth4g_*`).

Remake them (humans only, they read IDSSE): `scripts/fit_deadball.py`, `scripts/fit_team.py`,
`scripts/fit_touch.py`, with `IDSSE_DATA_DIR` set (the team and touch models with
`--devset configs/devset-smooth.yaml`).

Training data is the IDSSE dev set only (D15, organisers' issue #16): other matches, ball cut like
ours, never our match and no hand labels. A file made for other features is refused when loaded.
