# models/

Fitted numbers of the learned dead-ball score (`src/falafelona/learned_dead.py`). Small JSON files,
about 2 KB each: a few dozen coefficients, no match data and no frame values.

- `deadball_all.json`: fitted on all seven IDSSE matches. What a run and the submission use
  (`configs/exp/poss-learned-final.yaml`).
- `deadball_lomo_<match>.json`: fitted on the other six matches. The dev set scores each match
  with the file that never saw it (`models/deadball_lomo_{match_id}.json` in the dev-set configs).

Remake them (humans only, it reads IDSSE; same data, same files):

    export IDSSE_DATA_DIR=~/idsse-data
    .venv/bin/python scripts/fit_deadball.py

Training data is the IDSSE dev set only (D15, organisers' issue #16): other matches, ball cut like
ours, never our match and no hand labels. A file made for other features is refused when loaded.
