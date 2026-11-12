# Results

| Commit | Change | Dev spellF1 | Possessions | Public check | Verdict |
|---|---|---|---|---|---|
| — | Official `baseline_smooth50`: full-match spellF1 **0.2433**, macroF1 0.6724 (organisers) | — | — | — | Floor to beat |
| #23 | IDSSE dev set (7 matches, ball cut like ours): baseline, ball method none | 0.212 | n/a (dev set) | n/a | Reference for the dev set |
| #23 | ball interpolate, gaps up to 0.5 s (ball-interpolate.yaml) | 0.212 (+0.000) | n/a | n/a | keep as default: no effect on our gaps, simplest |
| #26 (not merged) | ball kalman, gaps up to 0.5 s (ball-kalman-short.yaml on #26) | 0.212 (+0.000) | n/a | n/a | drop: same as interpolate |
| #26 (not merged) | ball kalman, gaps up to 1.9 s (ball-kalman.yaml on #26) | 0.213 (+0.001) | n/a | n/a | drop: noise, only 0.1% of our missing ball is in gaps this short |
| #30 | final.yaml: ball interpolate, gaps up to 0.5 s | 0.212 | 669 (mean 7.6 s) | pending | keep: format valid, 199,319 rows |
| #33 | possession hmm, nearest team (poss-hmm.yaml), placeholder costs | 0.200 (-0.013) | n/a | n/a | inconclusive: noise, better on 1 of 7. alive/dead F1 -0.035 on every match: D on whole gaps, but dev-set gaps are random (mostly live play). Team side closer to truth (possession length 7-8 s vs truth 5.5-9.2 s) |
| #33 | possession hmm, distance margin (poss-hmm-margin.yaml), placeholder costs | 0.206 (-0.006) | n/a | n/a | inconclusive: noise, better on 1 of 7. Precision +0.051, recall -0.027, alive/dead F1 -0.035 (same cause) |
| #33 | possession switch_cost (poss-switch-cost.yaml) | 0.209 (-0.003) | n/a | n/a | inconclusive: noise, better on 3 of 7. Precision +0.038, recall -0.018 |
