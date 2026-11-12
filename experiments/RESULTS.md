# Results

| Commit | Change | Dev spellF1 | Possessions | Public check | Verdict |
|---|---|---|---|---|---|
| — | Official `baseline_smooth50`: full-match spellF1 **0.2433**, macroF1 0.6724 (organisers) | — | — | — | Floor to beat |
| 1fb986f | IDSSE dev set (7 matches, ball cut like ours): baseline, ball method none | 0.212 | n/a (dev set) | n/a | Reference for the dev set |
| 1fb986f | ball interpolate, gaps up to 0.5 s (ball-interpolate.yaml) | 0.212 (+0.000) | n/a | n/a | keep as default: no effect on our gaps, simplest |
| 1fb986f | ball kalman, gaps up to 0.5 s (ball-kalman-short.yaml) | 0.212 (+0.000) | n/a | n/a | drop: same as interpolate |
| 1fb986f | ball kalman, gaps up to 1.9 s (ball-kalman.yaml) | 0.213 (+0.001) | n/a | n/a | drop: noise, only 0.1% of our missing ball is in gaps this short |
