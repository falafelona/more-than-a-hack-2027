# Falafelona: technical document

Team Falafelona's entry for More Than a Hack 2027: one label per frame of FC Barcelona vs
Athletic Club saying who has the ball (`H` Barcelona, `A` Athletic, `D` dead ball).

| | |
|---|---|
| Team | Falafelona: Chikhani, Tala, Dudu, Laila, Abdallah |
| Repository | https://github.com/falafelona/more-than-a-hack-2027 |
| Submitted version | tag `v1.2` (commit `b40b4b8`), built from `configs/final.yaml` |
| Possession file | `Falafelona.csv`, 199,319 rows, SHA-256 `ea403a67c0ce2ea67b00decb50a766781c0c1bf51762f56f7d4926263b0e43a2` |

All numbers in this document are aggregates. It contains no match data and no frame-level values.

## 1. The approach

**The problem.** The ranking metric, spellF1, scores whole possessions: a predicted possession
counts only if it overlaps a true one well. A single wrong frame in the middle of a possession
splits it in two, and both halves usually miss. We measured this on synthetic labels
(`scripts/metric_study.py`): two 3-frame flicker blips per minute cut spellF1 to 0.61 while
frame accuracy (macroF1) stays at 0.997. Timing matters too, but less: the organisers' own table
(EVALUATION.md) gives 0.98 for a perfect answer 3 frames late, 0.89 for 8 frames and 0.65 for one
second. So the main job is not a better guess on each frame; it is **producing few, clean
possessions, switched at the right moment** (decision D13).

**The pipeline.** Three stages, each one module, all settings in `configs/final.yaml`.

```
tracking data + Smart Tagging
  → clean the ball track              ball.py         fill short gaps
  → dead-ball score per frame         deadball.py     0 = alive ... 1 = dead
                                      learned_dead.py   (one clue is a small fitted model)
  → team evidence per frame           possession.py   distance margin, touches, restarts
  → H / A / D decoder (HMM, Viterbi)  possession.py   chooses the best whole sequence
  → Falafelona.csv                    pipeline.py     199,319 rows, frame,label
```

**Why a decoder, rather than smoothing.** The official baseline gives each frame to the team of
the nearest player, then smooths with a majority vote. A vote over a fixed window must trade
flicker against blurred edges. Our decoder instead chooses the best *whole sequence* of H, A
and D: every change of state has a cost, so a change only happens when the evidence over several
frames pays for it. Brief wrong evidence (a pass flying past an opponent) cannot create a
possession, and the change is placed using frames after it as well as before (the task is
post-match).

**Why dead ball is part of the decoder.** The dead-ball stage gives a score, not a final answer.
A wrongly placed `D` splits a possession exactly like a wrong team does, so team and dead-ball
decisions are made together in one model (D13).

**How we chose settings.** We have no answers for our own match, and tuning on it is not allowed.
Every setting is chosen on other matches (the IDSSE dev set, section 3), never on this match
(D9, D15). Every candidate was scored against the current configuration in four runs: two kinds
of simulated ball gaps times two random seeds. What we did with the result (D18):

- A change went in only if it was better in all four runs, on at least 6 of the 7 dev matches
  in each. All five steps we kept meet this.
- A mean gain of at least +0.04 spellF1 per run passed outright; below that, a single run is
  inside the noise. Three of the five steps cleared +0.04 in every run.
- The other two went in by an explicit team decision, recorded in `experiments/RESULTS.md`:
  switch placement (+0.022 to +0.027, but better on all 7 matches in all four runs) and the
  dead-score blend (an exception, explained in section 1.3).

### 1.1 Ball track

The ball is missing on 40.7% of graded frames. Cleaners are chosen in the `ball:` section:

| Method | What it does | Outcome |
|---|---|---|
| `none` | the raw track | baseline |
| `reject` | removes false detections: impossible jumps and a ball far from every player | available, not in the final configuration |
| `interpolate` | fills gaps up to 0.5 s with a straight line, unless the ball could not have crossed the gap at 40 m/s | **used** |
| Kalman smoother | fills and smooths gaps up to 0.5 s or 1.9 s | dropped: +0.000 to +0.001, same as `interpolate` |

Cleaning the ball does not move the score, and the ball EDA says why: 99% of our missing ball
frames are in gaps longer than 5 s (median gap 10.7 s), and only about 0.1% are in gaps short
enough to fill. So we keep the simplest filler and treat long gaps as a question for the
dead-ball score and the decoder, not for the ball stage.

### 1.2 Dead-ball score

Every frame gets a chance from 0 (alive) to 1 (dead). Each clue adds its weight where it fires,
and the total is capped at 1.

| Clue | Fires when | Weight |
|---|---|---|
| `before_kickoff` | graded second-half frames before its kickoff, found from the Kick Off (Start) tag, never a fixed frame | 1.0 |
| `out_of_bounds` | the ball is more than 1 m past a touchline or goal line | 1.0 |
| `before_restart` | the seconds before a tagged restart (throw-in 8 s, corner 15 s, goal kick 20 s, free kick 25 s, kickoff after a goal 45 s) | 0.6 |
| `still_ball_at_spot` | a seen ball slower than 0.5 m/s for 2 s at a restart spot (near a line, in the goal area, at the centre spot) | 0.6 |
| `gap_chance` | inside a ball gap: one hand-built chance for the whole gap, from its length, whether the ball was last seen or reappears near a line or the centre spot, and whether the players stand still | 0.3 × the chance |
| `learned` | a per-frame chance from a small logistic model fitted on IDSSE (section 1.3) | 0.7 × the chance |

Tried and not used: `ball_missing` alone (the official baseline: every gap is dead) and a flat
"players standing still" clue. Player and ball speeds are computed from positions with a short
line fit, not read from the tracking file's speed field.

**What the data says (aggregates only).** Three facts shaped this stage.

- **Almost all of the missing ball is in long gaps.** The ball is missing on 64,643 of the
  158,857 graded frames (40.7%), and 99% of those frames are in 131 gaps longer than 5 s
  (median 10.7 s, longest 221 s).
- **Gaps tend to start at a line.** 32% of gaps start with the ball within 2 m of a touchline
  or goal line, against 9% of seen frames. So "last seen near a line" and the gap's length are
  better evidence than "the ball is missing".
- **Restart tags are few.** Smart Tagging has 59 set pieces in the match (26 throw-ins, 12
  corners, 10 goal kicks, 5 free kicks, 6 kickoffs): fewer than half the number of long ball gaps.

So a ball gap is a hint of a stoppage but a bad rule: calling every gap `D`, as the baseline
does, would split many possessions. This is why dead ball is a graded score fed to the decoder.

### 1.3 The learned dead-ball score and the blend

**Why.** The hand-built `gap_chance` gives each ball gap one value for its whole length, so a
20 s gap comes out all dead or all alive even when it holds some of each, and its weights were
set by hand.

**What.** `learned_dead.py` gives every frame its own chance, from two small logistic
regressions (ridge, fitted with plain numpy; no extra dependency):

- **inside a ball gap** (12 inputs): the gap's length; whether the ball was last seen, or
  comes back, near a line or at the centre spot, and how fast it moved at the gap's edges; the
  players' speed around this frame (windows of 1, 4 and 12 s) and the share standing still;
- **where the ball is seen** (13 inputs): ball speed, ball standing still (and at a restart
  spot), distance to a line, out of bounds, distance to the centre spot, how long it has been
  slow, and the same player-speed windows.

No input uses tags or the truth. The fitted numbers are a 2 KB JSON file of a few dozen
coefficients, committed in `models/`, so the organisers can inspect exactly what runs. The
pipeline only applies them; section 3 says what they were trained on.

**The blend.** On its own the learned score failed our rule: it gained under random gaps only
(+0.045 / +0.008 / +0.038 / +0.004 on top of touch evidence). Blending it with the hand-built
score fixed the weak side: inside a gap the final score is `0.3 × gap_chance + 0.7 × learned`.
Where the ball is seen, the learned score (× 0.7) adds to the other clues. Of three weights
tried (0.3, 0.5, 0.7), 0.7 was best, and the model trained on stoppage gaps only was the
steadiest across runs.

**The exception.** Against the configuration before it, the blend gains +0.040 / +0.023 /
+0.037 / +0.022 (seed 0 random / stoppage, seed 1 random / stoppage) and is better on 27 of 28
match-runs; its worst single match is −0.005. That is below our +0.04 bar under stoppage gaps
(and, at +0.037, just under it in one random-gap run). We took an exception (D24) for one
reason: under stoppage gaps even a *perfect* dead-ball call inside gaps is worth only +0.041
(oracle run), so a change to the gap score has almost no room to reach +0.04 there.
We state the cost of this plainly in section 6: the blend was chosen from about 8 variants on
the same 7 matches, so part of its lead may be luck.

### 1.4 Team evidence and the decoder

A Viterbi decoder over H / A / D. Costs: a switch between H and A 3.5, into or out of D 6.0,
dead-score weight 2.0.

- **Distance margin.** Team evidence is how much closer the nearest home player is to the ball
  than the nearest away player (or the reverse), fading to nothing when nobody is within 5 m. It
  is moved 0.5 s later (12 frames), because the receiver nears the ball just before touching it.
- **Fast-ball rule.** While the ball moves faster than 12 m/s (a pass, a shot, a long ball) no
  team gets distance evidence: the player it flies past is often the nearest one.
- **Restart taker.** When a dead spell of at least 3 s ends, the team nearest the ball around
  that moment takes the restart (the other team pays an extra cost for 2 s).
- **Switch placement.** Where no team has evidence (ball missing, flying, or far from
  everybody), H and A cost the same, so the decoder can put a switch anywhere in that stretch
  for the same total cost, and by default it lands on the first frame. In reality possession
  changes when the ball arrives. We place each switch 75% of the way into its no-evidence
  stretch. Only frames where H and A cost exactly the same change label, so the decoder's
  choice of possessions is untouched; only the timing moves.
- **Touch evidence.** Our diagnosis on the dev set showed that most missed possessions are
  short spells of the other team that the decoder skips, because a few frames of distance
  evidence cannot pay the switch cost; lowering that cost would bring the flicker back. So we
  look for a *touch*: the ball's velocity changes by at least 3 m/s between 0.12 s before and
  0.12 s after a frame, while a player is within 2 m of the ball and the other team's nearest
  player is at least 0.1 m further away. For 0.8 s after a touch the other team pays an extra
  cost. On the dev set 77–79% of true switches come with such a touch, and 85–86% of the
  switches it claims are right.

**Output.** One label per frame, header `frame,label`. We write no `X` anywhere (D14): every
frame gets `H`, `A` or `D`. An `X` cannot raise spellF1, and on a graded frame it always counts
as a miss for macroF1.

## 2. Which assets we used, and which we dropped

| Asset | Used? | How, or why not |
|---|---|---|
| Tactical tracking (ATD) | **Yes** | The only source of positions: ball x, y and the 46 player slots with their team. Slots are tracklets, not players, so nothing relies on identities |
| Tracking speed field | No | Its unit is not documented and it may be mostly empty; we compute ball and player speeds from positions |
| Smart Tagging, `SET PIECES` | **Yes, as evidence** | Kick Off (Start) marks the second-half kickoff; the other restart types open a dead-ball window before the tag. Tags are automatic and unreviewed, so they only add to the dead-ball score |
| Smart Tagging, team on a tag | No | 36 tags carry no team, including every throw-in (DATA.md); the team taking a restart comes from who is nearest the ball |
| Smart Tagging, other codes (build up, transitions, …) | No | Phase-of-play tags, too coarse in time for per-frame possession, and we could not test them: the dev set has no such tags |
| Homography | Loaded, not used | Our stages work in pitch metres straight from tracking; nothing needed pixels |
| Tactical video | No | We did not attempt our own ball detection; the time went into the decoder |
| Broadcast video, tracking and homography | No | On its own timeline, with cuts and replays, and a third of its frames have no usable homography |
| Public 5 minutes of ground truth | Pipeline check only | Never used to train, tune or choose between versions (D9) |

No pre-trained model, commercial API or external service is part of the pipeline. Everything
that runs is in the repository: the code, the configuration and the fitted model file.

## 3. Training data

**From this match: none.** We use no hand labels and no annotations of this match, from us or
from any other source, and nothing is trained or tuned on it (D21). The public five minutes of
ground truth are used only to check that the pipeline works, never to choose between versions
(D9).

**The only labelled dataset: IDSSE** [1].

| | |
|---|---|
| Source | Bassek et al., *Scientific Data* 12, 195 (2025): an open dataset of seven full professional matches from the German Bundesliga, 1st and 2nd division, season 2022/23, with player and ball positions at 25 frames per second, released under CC BY 4.0 |
| Approval | The organisers approved its use (official challenge repository, issue #16) |
| Who labelled it | The labels come with the dataset: its own tracking states, on every frame, whether the ball is in play (`BallStatus`) and which team owns it (`BallPossession`). **We labelled nothing ourselves**, on this match or any other |
| How much we used | All seven matches, every frame of both halves: about 1.0 million scored frames |
| What it is not | Another league and another tracking provider, and the provider's possession flag is not the organisers' labelling |

**How it becomes our truth.** Ball alive gives `H` or `A` from the possession flag; ball dead
gives `D` (D17).

**Making it as hard as our match.** IDSSE's ball is almost always tracked; ours is missing on
40.7% of graded frames. We remove the ball from IDSSE in gaps with the lengths measured on our
match (aggregates from our ball EDA: how much is missing and how long the gaps are), with a
fixed seed so every version sees the same gaps. Where the gaps go matters, so every comparison
is run with two kinds of cut:

- **random:** gaps start anywhere, like the ball hidden in open play;
- **stoppage:** gaps start as the ball goes dead, like the ball lost as it leaves the pitch.

Our real gaps are a mix of both, so the two bracket the truth. For the tag clues, IDSSE has no
Smart Tagging, so we simulate restart tags from the dev-set truth at the rate seen on our match.

**Use 1: choosing settings.** Every candidate version runs on the seven matches and is scored
with the official metric. Thresholds and costs chosen this way are all visible in
`configs/final.yaml`.

**Use 2: fitting the learned dead-ball score.** The model in the submission
(`models/deadball_stoppage_all.json`) is fitted by `scripts/fit_deadball.py` on all seven
matches, with the ball cut by stoppage gaps under two training seeds (100 and 101, not the
seeds 0 and 1 used for scoring), every 5th frame: 172,220 gap frames and 228,844 seen-ball
frames. The target is the dataset's own flag (ball dead or not).

**Keeping the dev score honest.** Scoring a match with a model that was fitted on it would
flatter the score. So on the dev set each match is scored with a model fitted on the *other
six* (`models/deadball_stoppage_lomo_<match>.json`, in the twin config
`configs/exp/poss-blend-learned70-stopmodel.yaml`). Every dev-set number for the blend in this
document comes from those leave-one-match-out models. Only the submission itself uses the
model fitted on all seven. For the touch settings, picked from a sweep of 17 configs, we ran
the same kind of check (`scripts/touch_lomo.py`): choosing the config on six matches and
scoring the seventh gave the same gain as choosing on all seven, and every fold picked the same
config.

[1] Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer." *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

## 4. The possession rules we implemented

How our labels follow the official definitions (CHALLENGE.md), as set in `configs/final.yaml`.

| Situation | Official rule | Where we drew the line |
|---|---|---|
| States | Every graded frame is `H`, `A` or `D` | The decoder has exactly these three states |
| Loose or contested ball | Possession stays with the last team that had it | When no player is within 5 m of the ball there is no team evidence, so the decoder keeps the current state |
| A deflection, a pass flying past | Not a change of possession | A change has a cost, so a few frames of distance evidence for the other team cannot pay for it |
| Controlled possession by the other team | A change of possession | Sustained evidence (the other team clearly closer to the ball over several frames) pays the switch cost. Evidence is moved 0.5 s later, because the receiver nears the ball just before the touch |
| A short possession by the other team | A controlled touch is a possession, however short | Touch evidence: a sharp change of the ball's velocity next to one team's player gives that team strong evidence for 0.8 s |
| A challenge or duel | No change without control | If both teams' nearest players are about equally close at a touch (within 0.1 m), the touch gives no evidence to either |
| The moment of the change | When the new team takes control | Inside a stretch with no evidence (ball missing or in flight), the switch is placed 75% of the way through, close to when the ball arrives, not when it leaves |
| Aerial duel, a ball flying past | Not a change without control | Our tracking has no ball height. While the ball moves faster than 12 m/s, no player gets distance evidence, so the ball must slow down near a player, or be touched, before possession can change |
| Restart after a stoppage | There is a clear possessor | When a dead spell of at least 3 s ends, the team nearest the ball around that moment takes the restart |
| Dead ball | Always `D`, never a possession | The dead-ball score pulls the decoder towards `D`; second-half frames before its kickoff are `D` |
| Ball missing | (not a rule: the central difficulty) | Inside a ball gap there is no team evidence, so the decoder holds the last owner until the ball returns, unless the dead-ball score says the ball went dead |
| Abstaining (`X`) | `X` on a graded frame is a miss for macroF1 | We write no `X` (D14) |

## 5. Results

We have no answers for the full match, so we report three kinds of evidence (D11): the dev-set
score, statistics on our match that need no answers, and the public five-minute check.

**Dev set: the submitted version against the baseline configuration** (mean spellF1 over the 7
matches, official metric).

| Dev-set run | Baseline | Submitted (v1.2) | Gain |
|---|---|---|---|
| seed 0, random gaps | 0.212 | 0.470 | +0.258 |
| seed 0, stoppage gaps | 0.269 | 0.519 | +0.250 |
| seed 1, random gaps | 0.201 | 0.455 | +0.254 |
| seed 1, stoppage gaps | 0.266 | 0.507 | +0.241 |

The dev score compares versions; it does not predict our score on this match (section 6).

**What we kept, step by step** (change in dev-set spellF1 against the step before it; seed 0
random / stoppage, then seed 1 random / stoppage; matches better of 7).

| Step | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|
| H / A / D decoder with distance margin, graded gap score, evidence lag, fast-ball rule (against the baseline configuration) | +0.059 / +0.098 (7/7) | +0.057 / +0.094 (7/7) | kept |
| + still ball at a restart spot, restart-taker evidence | +0.080 / +0.042 (7/7) | +0.080 / +0.044 (7/7) | kept |
| + switch placement 0.75 | +0.022 / +0.027 (7/7) | +0.025 / +0.026 (7/7) | kept: under +0.04, but better on all 7 matches in all four runs |
| + touch evidence | +0.056 / +0.060 (7/7) | +0.055 / +0.055 (7/7) | kept; leave-one-match-out check gives the same gain |
| + dead-score blend (0.3 hand-built, 0.7 learned) | +0.040 / +0.023 (7/7, 7/7) | +0.037 / +0.022 (6/7, 7/7) | kept **by exception**: under +0.04 in three of the four runs (section 1.3) |

**What we tried and dropped.** Each was run under the same check; every row, with the
per-match breakdown, is in `experiments/RESULTS.md`.

| Tried | Result (spellF1 change) | Why dropped |
|---|---|---|
| Ball: Kalman smoother | +0.000 to +0.001 | Same as `interpolate`; our gaps are too long to fill |
| Decoder with nearest-team evidence, or a plain switch cost | −0.012 to +0.008 | Noise; the distance margin beats both |
| Flat dead score (every gap leans dead) | +0.006 / +0.045 | No gain under random gaps: it turns live gaps into `D` |
| Velocity matching (a player moving with the ball) | +0.000 | No effect; already covered by the nearest-player evidence |
| Landing point of a fast ball | −0.019 to −0.035 (0/7) | Adds 40–60 possessions per match and precision falls by 0.06 to 0.12 |
| Team near the ball where it reappears after a long gap | −0.004 to 0.000 | The nearest team at that moment is often not the one with the ball |
| Team shape inside ball gaps (the more spread-out team has it) | +0.000 to +0.004 | No better than keeping the last owner |
| One-step changes of single settings (switch cost, dead costs and weight, team cost, evidence lag, fast-ball speed), checked on the 15 November configuration | none better than +0.001 | The chosen values sit on a flat optimum |
| Restart windows halved, or the tag clue off (with simulated tags) | +0.002 to +0.007 | Gains in all four runs, but on only 5 of 7 matches in at least one |
| Learned dead-ball score alone, in place of the hand-built gap score | +0.045 / +0.008 / +0.038 / +0.004 | Wins under random gaps only |
| Learned score only inside gaps | +0.014 / −0.009 / +0.010 / −0.015 | Worse under stoppage gaps |
| Blend weights 0.3 and 0.5; 0.7 with the model trained on both gap kinds | up to +0.048 / +0.020 / +0.038 / +0.017 | The stoppage-gap model at 0.7 was steadier across the four runs |

**What limits the score** (diagnosis runs on the dev set, before touch evidence). Replacing our
dead-ball score with the truth adds +0.07 to +0.10: the ceiling for any dead-ball detector. Of
the true possessions that fall mostly inside a ball gap, 97% are missed. With the full ball and
a perfect dead-ball call, 87% of the possessions still missed are shorter than 5 s.

**On our match** (label-free counts from `scripts/label_free_check.py`; they show the output
is sane and never choose between versions).

| | Submitted (v1.2) |
|---|---|
| Rows | 199,319, no `X` |
| Graded halves: `H` / `A` / `D` | 32.4% / 22.9% / 44.7% |
| All frames: `H` / `A` / `D` | 25.8% / 18.3% / 55.9% |
| Possessions | 518, mean 6.8 s |
| Outside the graded halves | every frame is `D` (40,462 frames, never scored) |

Possessions in the graded halves rose from 455 (after switch placement) to 501 (with touch
evidence) to 518 (with the blend). For reference, the baseline configuration gives 669
possessions over all frames (mean 7.6 s); the public five minutes hold 17 possessions
averaging 14.2 s, and the IDSSE matches 370–511 possessions of 5.5–9.2 s each.

**Public five-minute check.** It was run on the configuration frozen on 15 November (tag
`v1.0-freeze`): spellF1 0.410 and macroF1 0.681, against 0.326 and 0.621 for the baseline
(bootstrap interval for spellF1: 0.277 to 0.544). The three later steps were deliberately not
scored on it. With 17 possessions, one possession moves the score by about 0.06, so the public
window only shows that the pipeline works (D9); it cannot rank versions.

## 6. Limitations

**Where it fails.**

- **Possessions inside ball gaps are mostly lost.** With no ball there is no team evidence, so
  the decoder holds the last owner through the whole gap. On the dev set 97% of the true
  possessions that fall mostly inside a gap are missed, and our ball is missing on 40.7% of
  graded frames, almost all of it in gaps longer than 5 s.
- **Short possessions.** Most of the remaining misses are possessions under 5 s. Touch
  evidence recovers part of them, but only where the ball is seen, and about one in seven of
  the switches it claims is wrong.
- **Dead ball is far from perfect.** A perfect dead-ball call would add +0.07 to +0.10 on the
  dev set. Our score is a mix of hand-set weights and a small model.
- **No ball height, no identities.** High balls and aerial duels cannot be seen directly; the
  fast-ball rule is a stand-in. Slots are tracklets that swap, so we cannot follow one player.
- **Outside the graded halves every frame is `D`.** This is a side effect of the blend, not a
  decision about those frames. They are never scored.

**How sure we are of the choices.**

- **The blend is our least certain step.** It was chosen from about 8 variants on the same 7
  matches, so part of its lead may be luck, and its gain under stoppage gaps (+0.022 to +0.023)
  is well below our own +0.04 bar. We accepted it by exception (section 1.3).
- **The learned model assumes our gaps look like stoppage gaps.** It was trained on gaps that
  start as the ball goes dead. How dead a ball gap is depends on where the simulated gaps are
  placed, and the real mix on our match is unknown.
- **We checked neighbours one setting at a time**, not combinations of settings.

**What the dev set cannot tell us.**

- **Its truth is not the organisers' labelling.** It is the data provider's own possession
  flag, which may draw the line on loose balls and restarts differently.
- **It is another league and another tracking provider, with cleaner tracking.** Its players
  are identified and do not swap; ours averages 20.5 tracked objects per graded frame, with no
  identities and tracklets that swap.
- **Simulated gaps only bracket our real gaps.** Random and stoppage cuts are the two extremes.
- **Simulated tags are cleaner than real ones.** Smart Tagging is automatic and unreviewed,
  sparse (5 free kicks in the match), and no throw-in carries a team.
- **Some clues gain less on our match.** The still-ball clue relies on the ball staying
  visible during a stoppage; on our match it fires on only about 930 frames. Touch evidence
  added about 10% more possessions on our match, against 20–30% on the dev set.
- **No full-match score before results day.** The dev set says the method roughly doubles the
  baseline's score there; we do not claim a number for this match.

**What we would fix with more time.** Evidence for who has the ball *inside* gaps (the largest
loss); touch inputs for the learned dead-ball model; our own ball detection on the tactical
video to shorten the gaps; tracklet swaps, missing players and noisy tags simulated on the dev
set; and a dev set from a second source, so that choices are not made on the same 7 matches.

## 7. Reproduction: how to run it

Requirements: Python 3.12, `make`, `git`, and the organisers' data bundle in a folder
**outside** the repository. Dependencies are pinned to exact versions in `requirements.txt`
(numpy, pandas, PyYAML). No training is needed to reproduce the file: the fitted model is a
small JSON file in the repository.

```bash
git clone --recurse-submodules https://github.com/falafelona/more-than-a-hack-2027
cd more-than-a-hack-2027
git checkout v1.2                         # the submitted version
make setup                                # .venv and pinned dependencies
export MTAH_DATA_DIR=/path/to/mtah-data   # the data bundle, outside the repo
make submission                           # writes outputs/Falafelona.csv from configs/final.yaml
make validate                             # the official validate_submission.py
shasum -a 256 outputs/Falafelona.csv
```

| | |
|---|---|
| Output | `outputs/Falafelona.csv`, 199,319 rows, header `frame,label` |
| Expected SHA-256 | `ea403a67c0ce2ea67b00decb50a766781c0c1bf51762f56f7d4926263b0e43a2` |
| Official validator | OK |
| Tests | `make check`: lint, and 802 tests on synthetic data only (no match data needed) |

**Determinism.** A fixed seed, no order-dependent parallelism, and the model is applied, not
trained, at run time. We built the submitted file twice and got the identical SHA-256. Earlier,
at the `v1.0-freeze` tag, a clean clone with a fresh environment gave the identical file on
macOS and on Linux (Docker `python:3.12`).

**Repeating the dev-set comparisons.** Download IDSSE [1] to a folder outside the repository,
set `IDSSE_DATA_DIR`, and run `scripts/devset_eval.py` (`experiments/README.md` explains the
setup and the options for the four runs). The reference for the submitted version on the dev set is
`configs/exp/poss-blend-learned70-stopmodel.yaml`. To refit the model files:
`scripts/fit_deadball.py` (it reads IDSSE only).

**Data handling.** The match data is not in the repository (D1, D2): it is read
from `MTAH_DATA_DIR`, and a check in the test suite refuses data-shaped files. AI coding
assistants were used to write code and documentation. Our rule is that they never read match
data and never run the pipeline on it: people run those commands and share only aggregates
(`CLAUDE.md`, D4, D16).
