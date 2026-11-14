# Falafelona: technical document

Team Falafelona's entry for More Than a Hack 2027: one label per frame of FC Barcelona vs
Athletic Club saying who has the ball (`H` Barcelona, `A` Athletic, `D` dead ball).
All numbers in this document are aggregates. No match data, no frame-level values.

Sections marked **[to fill]** are completed after the team freeze (15–16 Nov).

## 1. Approach

**The problem.** The ranking metric, spellF1, scores whole possessions: a predicted possession
counts only if it overlaps a true one well. A single wrong frame in the middle of a possession
splits it in two, and both halves usually miss. We measured this on synthetic labels
(`scripts/metric_study.py`): two 3-frame flicker blips per minute cut spellF1 to 0.61 while
frame accuracy (macroF1) stays at 0.997. Timing matters too, but less: the organisers' own table
(EVALUATION.md) gives 0.98 for a perfect answer 3 frames late, 0.89 for 8 frames and 0.65 for one
second. So the main job is not a better guess on each frame; it is **producing few, clean
possessions, switched at the right moment** (decision D13).

**The pipeline.**

```
tracking data
  → clean the ball track              ball.py       fill short gaps, reject false detections
  → dead-ball score per frame         deadball.py   0 = alive ... 1 = dead
  → team evidence per frame           possession.py which team is closer to the ball
  → H / A / D decoder (HMM, Viterbi)  possession.py chooses the best whole sequence
  → Falafelona.csv                    pipeline.py   199,319 rows, frame,label
```

**Why a decoder.** The official baseline gives each frame to the team of the nearest player, then
smooths with a majority vote. A vote over a fixed window must trade flicker against blurred edges.
Our decoder instead chooses the best *whole sequence* of H, A and D: every change of state has a
cost, so a change only happens when the evidence over several frames pays for it. Brief wrong
evidence (a pass flying past an opponent) cannot create a possession, and the change is placed
where the evidence fits best, using frames after it as well as before (the task is post-match).

**Why dead ball is part of the decoder.** The dead-ball stage gives a score, not a final answer.
A wrongly placed `D` splits a possession exactly like a wrong team does, so team and dead-ball
decisions are made together in one model (D13).

**How we chose settings.** We have no answers for our own match, and tuning on it is not allowed.
Every setting is chosen on other matches (the IDSSE dev set, section 3), never on this match
(D9, D15). A change goes into the submitted configuration only if, against the current one, it
wins by more than 0.04 spellF1 under both kinds of simulated ball gaps, or wins on at least 6 of
7 matches under both, on two seeds. Below 0.04 differences are noise.

**Final configuration.** [to fill after the freeze: which ball, dead-ball and possession methods
are in `configs/final.yaml`, and the fallback rule if nothing passed: baseline + interpolate.]

### 1.1 Ball track
Owner: Abdallah (task 2.7). [to fill]

### 1.2 Dead ball and restarts
Owner: Chikhani (task 3.8). [to fill: the methods and the chosen dead-ball score]

**What the data says (dead-ball EDA, task 3.3).** We compared three dead-ball clues on our match:
Smart Tagging restarts (58 set pieces), long ball gaps (2 s or more, 133) and the ball seen past
a line (67 stretches). "Nearby" means from 10 s before a restart to 2 s after it.

| Clue | Result |
|---|---|
| Restarts with a long ball gap nearby | 84% |
| Restarts with the ball seen out of bounds nearby | 26% |
| Restarts with either | 97% |
| Long gaps with a restart nearby | 37% |
| Restarts without a team (every throw-in) | 31 of 58 |

So a long ball gap is a strong hint of a stoppage but a bad rule: most stoppages come with one,
yet most long gaps are the ball lost in live play. Calling every gap `D` would split many
possessions. Restarts are frequent, so a gap can sit near one by chance; against that chance
level, gaps over 20 s are near a restart much more often (0.61 against 0.33) and gaps of 5–20 s
barely more (0.25 against 0.20) [Tala to confirm these two from the final EDA run]. And 32% of
gaps start with the ball within 2 m of a line, against 9% of seen frames. So gap length and
"last seen near a line" are better evidence than "the ball is missing". This is why dead ball is
a graded score fed to the decoder.

## 2. Assets used (and dropped)
Owner: Tala. [to fill]

## 3. Training data (required)

**From this match: none.** We use no hand labels and no annotations of this match, and nothing is
trained or tuned on it. The public five minutes of ground truth are used only to check that the
pipeline works (it reproduces the official baseline's score), never to choose between versions
(D9).

**Dev set: IDSSE.** Every setting is chosen on the IDSSE dataset [1]: seven full professional
matches from the German Bundesliga and 2. Bundesliga, 2022/23, tracked at 25 Hz, released under
CC BY 4.0. The organisers approved its use (official challenge repository, issue #16).
**Who labelled it, and how much:** the labels come with the dataset, from the league's official
tracking data; we labelled nothing ourselves. We use all seven matches, every frame of both
halves.

- **Truth.** IDSSE's own tracking states, for every frame, whether the ball is in play
  (`BallStatus`) and which team owns it (`BallPossession`). Ball alive gives `H` or `A` from that
  flag; ball dead gives `D`. This is the provider's flag, not the organisers' labelling (D17).
- **Making it as hard as our match.** IDSSE's ball is almost always tracked; ours is missing on
  about 41% of graded frames, mostly in gaps longer than 5 s. We remove the ball from IDSSE in
  gaps with the lengths measured on our match, with a fixed seed so every version sees the same
  gaps. Where the gaps go matters, so every comparison is run with two kinds of cut:
  - **random:** gaps start anywhere, like the ball hidden in open play;
  - **stoppage:** gaps start as the ball goes dead, like the ball lost as it leaves the pitch.

  Our real gaps are a mix of both, so the two bracket the truth.
- **Use.** Every candidate version runs on the seven matches and is scored with the official
  metric. The dev score is used to compare versions, not to predict our final score.

[1] Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer." *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

## 4. Possession rules

How our labels follow the official definitions (CHALLENGE.md). [Dudu to confirm against the final
decoder and settings.]

| Situation | Official rule | What we implemented |
|---|---|---|
| States | Every graded frame is `H`, `A` or `D` | The decoder has exactly these three states |
| Loose or contested ball | Possession stays with the last team that had it | When no player is near the ball there is no team evidence, so the decoder keeps the current state |
| A touch, a deflection, a pass flying past | Not a change of possession | A change has a cost, so a few frames of evidence for the other team cannot pay for it |
| Controlled possession by the other team | A change of possession | Sustained evidence (the other team clearly closer to the ball over several frames) pays the switch cost |
| Aerial duel | Not a change without control | Our tracking has no ball height, so we cannot see the ball in the air. A fast ball (ball velocity per frame) counts as a pass, not as control [Dudu: fast-ball rule, final?] |
| Restart after a stoppage | There is a clear possessor | Play resumes from `D` to the team the evidence supports after the restart |
| Dead ball | Always `D`, never a possession | The dead-ball score pulls the decoder towards `D`; second-half frames before its kickoff are `D`, found from the Kick Off (Start) tag, never a fixed frame (added to the final configuration at the freeze) |
| Abstaining (`X`) | `X` on a graded frame is a miss | We write no `X` [D14, to confirm] |

## 5. Results

We have no answers for the full match, so we report three kinds of evidence (D11): the dev-set
score, statistics on our match that need no answers, and the public five-minute check.

**The official baseline, as reference.**

| | Baseline (`baseline_smooth50`) |
|---|---|
| Full-match spellF1 / macroF1 (published by the organisers) | 0.2433 / 0.6724 |
| Public 5-minute check, spellF1 / macroF1 | 0.326 / 0.621 |
| Dev-set spellF1, random / stoppage gaps | 0.212 / 0.269 |
| Possessions on our match | 669 |
| Median possession length | 4.7 s |
| Possessions shorter than 1 s | 132 |

For comparison, the public five minutes hold 17 possessions averaging 14.2 s, and the IDSSE
matches hold 370–511 possessions of 5.5–9.2 s each. The baseline reports more possessions than
either, and one in five lasts under a second: it splits real possessions.

**Candidates on the dev set** (change in spellF1 against the reference, random / stoppage gaps,
seed 0, and on how many of the 7 matches it was better). Full rows, including seed 1, are in
`experiments/RESULTS.md`.

| Version | random | stoppage | Verdict |
|---|---|---|---|
| Ball: interpolate short gaps | +0.000 | — | kept as the simplest gap filler (no effect: only 0.1% of our missing ball is in short gaps) |
| Ball: Kalman smoother | +0.000 to +0.001 | — | dropped: same as interpolate |
| Possession: switch cost | −0.003 (3/7) | +0.008 (5/7) | dropped: noise |
| Possession: HMM, nearest team | −0.012 (1/7) | +0.008 (5/7) | dropped: noise |
| Possession: HMM, distance margin | −0.006 (1/7) | +0.038 (7/7) | dropped alone: no gain under random gaps |
| HMM margin + flat dead score | +0.006 (4/7) | +0.045 (7/7) | dropped: no gain under random gaps |
| HMM margin + graded gap score | +0.022 (7/7) | +0.047 (7/7) | best candidate so far, not in: the random-gap gain is below 0.04, and 5/7 on seed 1 |

[to fill after the freeze: final retuned results, the submitted version against the baseline,
and the same match statistics as above for the submitted file.]

## 6. Limitations

- **The dev-set truth is not the organisers' labelling.** It is the data provider's own
  possession flag, which may draw the line on loose balls and restarts differently.
- **IDSSE has no Smart Tagging,** so any clue built on restart tags cannot be tested on the dev
  set as it is.
- **IDSSE tracking is cleaner than ours.** It follows 22 identified players with no swaps; ours
  averages 20.5 tracked objects, with no identities and frequent tracklet swaps.
- **Simulated gaps only bracket our real gaps.** Random and stoppage cuts are the two extremes;
  our real gaps are a mix, so a version must win under both.
- **No ball height.** Our tracking has the ball's x, y and speed only, so high balls and aerial
  duels cannot be seen directly.
- **No full-match score before results day.** The public five minutes hold 17 possessions, so
  one possession moves the score by about 0.06; we use it only as a pipeline check (D9).
- **Smart Tagging is evidence, not truth.** It is automatic and unreviewed, sparse (5 free kicks
  in the match), and every throw-in has no team.
- [ball, dead ball, possession: one line each from their owners]

**With more time** we would: test the restart-tag clues on a dev set with simulated tags;
simulate tracklet swaps and missing players on the dev set; and retune the decoder so the
number and length of possessions match the truth (ours are still too few and too long on the
dev set).

## 7. Reproduction
Owner: Tala. [to fill: one command, SHA-256 of the file, runtime and machine]