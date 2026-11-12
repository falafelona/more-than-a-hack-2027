# Results

| Commit | Change | Dev spellF1 | Possessions | Public check | Verdict |
|---|---|---|---|---|---|
| — | Official `baseline_smooth50`: full-match spellF1 **0.2433**, macroF1 0.6724 (organisers) | — | — | — | Floor to beat |
| #23 | IDSSE dev set (7 matches, ball cut like ours): baseline, ball method none | 0.212 | n/a (dev set) | n/a | Reference for the dev set |
| #23 | ball interpolate, gaps up to 0.5 s (ball-interpolate.yaml) | 0.212 (+0.000) | n/a | n/a | keep as default: no effect on our gaps, simplest |
| #26 (not merged) | ball kalman, gaps up to 0.5 s (ball-kalman-short.yaml on #26) | 0.212 (+0.000) | n/a | n/a | drop: same as interpolate |
| #26 (not merged) | ball kalman, gaps up to 1.9 s (ball-kalman.yaml on #26) | 0.213 (+0.001) | n/a | n/a | drop: noise, only 0.1% of our missing ball is in gaps this short |
| #30 | final.yaml: ball interpolate, gaps up to 0.5 s | 0.212 | 669 (mean 7.6 s) | pending | keep: format valid, 199,319 rows |
