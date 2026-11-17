# Falafelona: technical document

Team Falafelona's entry for More Than a Hack 2027: one label per frame of FC Barcelona vs
Athletic Club saying who has the ball (`H` Barcelona, `A` Athletic, `D` dead ball).

| | |
|---|---|
| Team | Falafelona: Mohamed Chikhani, Tala Fakhreddine, Mohamed Khaled, Laila Mansour, Abdallah Kassem |
| Repository | https://github.com/falafelona/more-than-a-hack-2027 |
| Submitted version | tag `[FILL: tag]` (commit `[FILL: commit]`), built from `configs/final.yaml` |
| Possession file | `Falafelona.csv`, 199,319 rows, SHA-256 `[FILL: sha256]` |

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

**The pipeline.** Four stages, all settings in `configs/final.yaml`.

```
tracking data + Smart Tagging
  → clean the ball track              ball.py           fill short gaps
  → dead-ball score per frame         deadball.py       0 = alive ... 1 = dead
                                      learned_dead.py   (one clue is a fitted model)
  → team evidence per frame           possession.py     distance margin, touches, restarts
                                      learned_touch.py  (which team each touch belongs to)
                                      learned_team.py   (who has the ball inside a ball gap)
  → H / A / D decoder (HMM, Viterbi)  possession.py     chooses the best whole sequence
  → clean-up                          possession.py     switch timing, very short spells merged
  → Falafelona.csv                    pipeline.py       199,319 rows, frame,label
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

**Why learned models, and only small ones.** The hand-built evidence works where the ball is
seen. Inside ball gaps (41% of our graded frames) it has nothing to say, and our diagnosis runs
showed that this is where most of the score is lost (section 5). So three parts of the pipeline
are small logistic models fitted on other matches (section 3): the dead-ball chance, the team of
each touch, and who has the ball inside a gap. They are ridge logistic regressions fitted with
plain numpy (no extra dependency), each a small JSON file in `models/` that anyone can read.
A gradient-boosted model on the same inputs did no better (section 5).

**How we chose settings.** We have no answers for our own match, and tuning on it is not allowed.
Every setting is chosen on other matches (the IDSSE dev set, section 3), never on this match
(D9, D15). Every candidate is scored against the current configuration in four runs: two kinds
of simulated ball gaps times two random seeds. What we did with the result (D18):

- A change went in only if it was better in all four runs, on at least 6 of the 7 dev matches
  in each.
- A mean gain of at least +0.04 spellF1 per run passed outright; below that, a single run is
  inside the noise.
- A smaller gain went in only by an explicit team decision, recorded in
  `experiments/RESULTS.md`. Most of our later steps were of this kind: each gained less than
  +0.04, but all four runs agreed (section 5 lists them).

### 1.1 Ball track

The ball is missing on 40.7% of graded frames. Cleaners are chosen in the `ball:` section:

| Method | What it does | Outcome |
|---|---|---|
| `none` | the raw track | baseline |
| `reject` | removes false detections: impossible jumps and a ball far from every player | available, not in the final configuration |
| `interpolate` | fills gaps up to 0.5 s with a straight line, unless the ball could not have crossed the gap at 40 m/s | **used** |
| Kalman smoother | fills and smooths gaps up to 0.5 s or 1.9 s | dropped: +0.000 to +0.001, same as `interpolate` |
| Longer `interpolate` | fills gaps up to 1.0, 1.5 or 2.0 s | dropped: +0.000 |

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
| `learned` | a per-frame chance from the learned dead-ball model (section 1.3) | 0.7 × the chance |

Why the clues are set this way:

- **Out of bounds uses a 1 m margin** because tracking is noisy near the lines, and a ball on
  the line is still in play.
- **Restart windows differ by type** because a throw-in is taken much faster than a free kick
  or a kickoff after a goal (which includes the celebration). A window that is too long
  reaches back into live play before the ball went out, so each type gets its own length.
- **The team on a tag is not used for dead ball**, so throw-ins tagged without a team still
  count: a dead ball is `D` whoever restarts it.
- **The restart-tag clue weighs 0.6, not 1.0.** That is enough on its own to make a frame lean
  dead, but Smart Tagging is automatic and unreviewed (DATA.md), so one wrong tag adds
  evidence the decoder weighs against everything else instead of overriding it. The kickoff
  clue keeps 1.0: the second half cannot have started before its kickoff.

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

### 1.3 The three learned models

All three are ridge logistic regressions. None of their inputs uses tags, the truth or anything
from the public five minutes. The pipeline only applies them; section 3 says how they were
fitted.

**Dead ball (`learned_dead.py`, version 3, `models/deadball_stoppage3_all.json`).** The
hand-built `gap_chance` gives each ball gap one value for its whole length, so a 20 s gap comes
out all dead or all alive even when it holds some of each. The learned model gives every frame
its own chance, from two parts:

- **inside a ball gap** (20 inputs): the gap's length; whether the ball was last seen, or comes
  back, near a line or at the centre spot, and how fast it moved at the gap's edges; the
  players' speed around this frame (windows of 1, 4 and 12 s) and the share standing still.
  Version 3 adds 8 inputs on timing and the crowd: time since the gap started and until it
  ends, a line clue that fades with time, the trend of the players' speed, how crowded it was
  where the ball left and where it returns, and how spread out the players are;
- **where the ball is seen** (13 inputs): ball speed, ball standing still (and at a restart
  spot), distance to a line, out of bounds, distance to the centre spot, how long it has been
  slow, and the same player-speed windows.

Inside a gap the final dead score is `0.3 × gap_chance + 0.7 × learned`; where the ball is
seen, the learned chance (× 0.7) adds to the other clues. The learned score alone gained under
random gaps only; the blend kept that gain and fixed the weak side under stoppage gaps
(section 5). It went in by exception to our +0.04 rule (D24).

**Touch team (`learned_touch.py`, `models/touch_all.json`).** A *touch* is a sharp change of
the ball's velocity: at least 3 m/s between 0.12 s before and 0.12 s after a frame. A simple
rule gives the touch to the nearest player's team, if he is within 4 m and the other team's
nearest player is at least 0.1 m further; it names a team for 91% of the touches and is right on
87.8% of those. The model decides it for every
touch, from 20 inputs: the distance margin between the two teams before, at and after the touch
and 1 s later (the receiver), how many players of each team are close, which team's nearest
player moves *with* the ball afterwards, the ball's direction along each team's attack, the
rule's answer for this touch and its neighbours, ball speeds, and the time to the neighbouring
touches. It is right on 90.2% of all touches, on matches it was not fitted on. For 0.8 s after
a touch, the team the model thinks does *not* have the ball pays `2.0 × |log-odds|` per frame,
so a confident call weighs more than an unsure one.

**Team inside ball gaps (`learned_team.py`, version 5, `models/team_smooth5g_all.json`).**
Inside a gap the distance evidence is empty, so the decoder could only hold the last owner. The
model gives the chance that home has the ball on each gap frame, from 21 inputs: where the gap
sits in time, who was closer to the ball when it was last seen and when it reappears, touches
just before and after the gap, how the two teams move (pushing up or dropping back, turning),
where play is on the pitch, and the teams' shape (width, length, height of the back line,
spread at the gap edges). It is right on about 83% of gap frames on held-out matches. It is used
**only inside gaps**, with a small weight: the other team pays `0.25 × |log-odds|` per frame. A
model used on every frame, or with a larger weight, did worse (section 5).

**Kept in the code, not used:** a *switch hazard* model (`learned_switch.py`), which lowered the
switch cost where a change looked likely. It could not tell where possession changes (AUC
0.564) and gained at most +0.001.

### 1.4 Team evidence, the decoder and the clean-up

A Viterbi decoder over H / A / D. Costs: a switch between H and A 10, into or out of D 18,
dead-score weight 2.0. These were swept in a grid with a leave-one-match-out check
(`scripts/cost_sweep.py`, section 5); the earlier costs (3.5 and 6) dated from before most of
the evidence existed.

Team evidence on each frame adds up:

- **Distance margin.** How much closer the nearest home player is to the ball than the nearest
  away player (or the reverse), fading to nothing when nobody is within 5 m; weight 0.5. It is
  moved 24 frames (0.96 s) later, because the receiver nears the ball before he touches it.
- **Fast-ball rule.** While the ball moves faster than 12 m/s (a pass, a shot, a long ball) no
  team gets distance evidence: the player it flies past is often the nearest one.
- **Touches**, scored by the touch model (section 1.3).
- **Restart taker.** When a dead spell of at least 3 s ends, the team nearest the ball around
  that moment takes the restart (the other team pays an extra cost for 2 s).
- **Inside ball gaps**, the gap team model (section 1.3).

Then two clean-up steps:

- **Switch placement.** Where no team has evidence, H and A cost the same, so the decoder can
  put a switch anywhere in that stretch for the same total cost, and by default it lands on the
  first frame. In reality possession changes when the ball arrives. We move each such switch to
  the end of its no-evidence stretch (the first frame with evidence again). Only frames where
  H and A cost exactly the same change label, so the possessions themselves are untouched.
- **Minimum spell of 1 s.** A run of one team shorter than 1 s that sits *between two runs of
  the other team* is given to that other team: a touch without control or a deflection is not a
  change of possession (CHALLENGE.md). Runs next to a dead ball are kept, because a restart has a
  clear possessor.

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
| Smart Tagging, phase codes (build up, progression, …) | Checked, not used | `scripts/tag_team_check.py` compared each code's team with our own labels: BUILD UP and PROGRESSION point to the team in possession about 90–94% of the time but cover only about 5% of gap frames, and the codes that cover gaps agree only 60–74% of the time. Weaker than our gap model, and IDSSE has no tags to test them on |
| Tactical homography | Checked, not used | `scripts/camera_aim_check.py` asked whether the camera points at the ball during gaps: the picture centre is a median 23.6 m from the ball, against 20.3 m for the players' centre, and closer on only 28–35% of frames. The wide tactical camera adds no ball information |
| Broadcast tracking | Checked, not used | `scripts/broadcast_sync_check.py` aligned it to the tactical timeline by matching the two ball tracks (no mapping is provided): 99,000 broadcast frames synced, players within 0.3–0.4 m. It would fill only 9.2% of tactical gap frames, because the broadcast mostly shows replays and close-ups when the tactical ball is lost, and the gain cannot be measured on IDSSE |
| Tactical and broadcast video | No | We did not attempt our own ball detection |
| Public 5 minutes of ground truth | Pipeline check only | Never used to train, tune or choose between versions (D9) |

No pre-trained model, commercial API or external service is part of the pipeline. Everything
that runs is in the repository: the code, the configuration and the fitted model files.

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

**Making it look like our match.** IDSSE's ball is almost always tracked; ours is missing on
40.7% of graded frames. Three steps make the dev set look like ours:

- **Gaps.** We remove the ball from IDSSE in gaps with the lengths measured on our match (from
  our ball EDA), with a fixed seed, so every version sees the same gaps. Where the gaps go
  matters, so every comparison runs with two kinds of cut: **random** (anywhere, like the ball
  hidden in open play) and **stoppage** (starting as the ball goes dead). Our real gaps are a
  mix of both.
- **Smoothing** (`configs/devset-smooth.yaml`). Our provider's ball track is smoother than
  IDSSE's: a kick's sharp turn comes out rounded. So the IDSSE ball is smoothed (Gaussian,
  sigma 0.25 s) before the gaps are cut. The sigma was chosen with label-free touch statistics,
  comparing the two tracking sets: the distance from the ball to the nearest player at a
  velocity-change peak (1.66 m on the smoothed IDSSE ball, 1.75 m on ours), the share of clear
  touchers and the time shift between ball and players. Sigma 0.2 and 0.3 led to the same
  choices. Every step after the change of decoder costs and timing was chosen on this smoothed
  dev set.
- **Tags.** IDSSE has no Smart Tagging, so for the tag clues we simulate restart tags from the
  dev-set truth, at the rate seen on our match.

**Use 1: choosing settings.** Every candidate version runs on the seven matches and is scored
with the official metric. Seed 0 is used for tuning; seed 1 only confirms.

**Use 2: fitting the three models.** Each model is fitted by its script on IDSSE with the ball
cut like ours, under both gap kinds, with training seeds 100 and 101 (never the scoring seeds 0
and 1). The team and touch models are trained on home/away-swapped copies too, so they cannot
learn that "home" is more likely. The target is always the dataset's own flag.

| Model | Script | Fitted on | Rows in the submitted model file |
|---|---|---|---|
| Dead ball v3 | `scripts/fit_deadball.py` | stoppage gaps, every 5th frame | 172,220 gap frames, 228,844 seen-ball frames |
| Team in gaps v5 | `scripts/fit_team.py` | frames whose truth is H or A, every 5th frame | 312,500 gap frames (572,428 seen-ball frames for the seen part, not used) |
| Touch team | `scripts/fit_touch.py` | every detected touch whose truth just after is H or A | 44,380 touches |

**Keeping the dev score honest.** Scoring a match with a model that was fitted on it would
flatter the score. So on the dev set each match is scored with models fitted on the *other six*
(`models/*_lomo_<match>.json`). Every dev-set number for a learned model in this document comes
from those leave-one-match-out models; only the submission uses the models fitted on all
seven. Settings picked from a sweep were checked the same way (`scripts/touch_lomo.py` for the
touch settings, the leave-one-match-out step of `scripts/cost_sweep.py` for the costs).

[1] Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer." *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

## 4. The possession rules we implemented

How our labels follow the official definitions (CHALLENGE.md), as set in `configs/final.yaml`.

| Situation | Official rule | Where we drew the line |
|---|---|---|
| States | Every graded frame is `H`, `A` or `D` | The decoder has exactly these three states |
| Loose or contested ball | Possession stays with the last team that had it | When no player is within 5 m of the ball there is no distance evidence, so the decoder keeps the current state unless a touch or the gap model says otherwise |
| A deflection, a touch without control | Not a change of possession | A change has a cost, so a few frames of evidence for the other team cannot pay for it. A spell under 1 s inside the other team's possession is given back to that team |
| Controlled possession by the other team | A change of possession | Sustained evidence (the other team clearly closer to the ball over several frames, or a confident touch) pays the switch cost. Distance evidence is moved 0.96 s later, because the receiver nears the ball before the touch |
| A short possession by the other team | A controlled touch is a possession, however short | A touch gives its team evidence for 0.8 s, weighted by how sure the touch model is. A spell must last at least 1 s, unless it borders a dead ball |
| A challenge or duel | No change without control | The touch model sees both teams' distances around the touch; when it is unsure, its evidence is close to zero |
| The moment of the change | When the new team takes control | Inside a stretch with no evidence, the switch goes to the end of the stretch: when the ball arrives, not when it leaves |
| Aerial duel, a ball flying past | Not a change without control | Our tracking has no ball height. While the ball moves faster than 12 m/s, no player gets distance evidence, so the ball must slow down near a player, or be touched, before possession can change |
| Restart after a stoppage | There is a clear possessor | When a dead spell of at least 3 s ends, the team nearest the ball around that moment takes the restart |
| Dead ball | Always `D`, never a possession | The dead-ball score pulls the decoder towards `D`; second-half frames before its kickoff are `D` |
| Ball missing | (not a rule: the central difficulty) | Inside a ball gap, the gap team model gives weak evidence; otherwise the decoder holds the last owner, unless the dead-ball score says the ball went dead |
| Abstaining (`X`) | `X` on a graded frame is a miss for macroF1 | We write no `X` (D14) |

## 5. Results

We have no answers for the full match, so we report three kinds of evidence (D11): the dev-set
score, statistics on our match that need no answers, and the public five-minute check. The dev
score compares versions; it does not predict our score on this match (section 6).

**Two dev sets, two scales.** Up to the change of decoder costs and timing we scored on the dev
set with the IDSSE ball as it is; every later step was scored on the smoothed dev set
(section 3). Smoothing lowers every score, so numbers from the two are not comparable: each step
below is compared with the step before it on the same dev set.

**The submitted version.** Mean spellF1 over the 7 matches, official metric. Because of the
change of dev set, the gain over the baseline is shown in two parts, each on one dev set.

| Dev-set run | Original dev set: baseline configuration | Original dev set: first submission (v1.2) | Smoothed dev set: after costs and timing | Smoothed dev set: submitted |
|---|---|---|---|---|
| seed 0, random gaps | 0.212 | 0.470 | 0.377 | **0.462** |
| seed 0, stoppage gaps | 0.269 | 0.519 | 0.412 | **0.513** |
| seed 1, random gaps | 0.201 | 0.455 | not reported | **0.452** |
| seed 1, stoppage gaps | 0.266 | 0.507 | not reported | **0.499** |

On the original dev set the first submission roughly doubled the baseline configuration's score.
On the smoothed dev set, the steps after it added +0.085 / +0.101 (seed 0, random / stoppage).
The costs-and-timing step in between gained +0.022 / +0.021 and then +0.009 / +0.010 on the
original dev set (table below).

**What we kept, step by step** (change in dev-set spellF1 against the step before it, mean over
the 7 matches; seed 0 random / stoppage, then seed 1 random / stoppage).

*On the original dev set:*

| Step | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|
| H / A / D decoder with distance margin, graded gap score, evidence lag, fast-ball rule (against the baseline configuration) | +0.059 / +0.098 (7/7) | +0.057 / +0.094 (7/7) | kept |
| + still ball at a restart spot, restart-taker evidence | +0.080 / +0.042 (7/7) | +0.080 / +0.044 (7/7) | kept |
| + switch placement 0.75 | +0.022 / +0.027 (7/7) | +0.025 / +0.026 (7/7) | kept by decision: under +0.04, better on 7/7 in all four runs |
| + rule-based touch evidence | +0.056 / +0.060 (7/7) | +0.055 / +0.055 (7/7) | kept; the leave-one-match-out check gives the same gain |
| + dead-score blend (0.3 hand-built, 0.7 learned v2) | +0.040 / +0.023 (7/7, 7/7) | +0.037 / +0.022 (6/7, 7/7) | kept by exception (D24): this was the first submission, v1.2 |
| + decoder costs 10 / 18 / 18 and switch timing | costs +0.022 / +0.021 (7/7); timing +0.009 / +0.010 | timing +0.010 / +0.011 | kept by exception |

*On the smoothed dev set* (seed 1 was not reported for the first four steps):

| Step | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|
| Touch distance 4 m, evidence lag 24 frames | +0.025 / +0.034 | not reported | kept by decision |
| + gap team model (first version), weight 0.25, gaps only | +0.019 / +0.015 | not reported | kept by decision |
| + gap team model v4, minimum spell 1 s | +0.015 / +0.014 | not reported | kept by decision |
| + gap team model v5 (team shape) | +0.005 / +0.006 | not reported | kept by decision |
| + dead-ball model v3 | +0.006 / +0.011 | +0.005 / +0.009 | kept by decision |
| + touch team model | +0.015 / +0.021 | +0.013 / +0.021 | kept by decision |

**What limits the score** (oracle runs: one part of the pipeline is given the truth, the rest
stays real; smoothed dev set, seed 0, random gaps, against the configuration of the time).

| Given the truth for | Gain | What it told us |
|---|---|---|
| every touch, with its team | +0.149 | touches are the biggest lever; led to the touch model |
| who has the ball inside gaps | +0.145 | the other big lever; no feature set or model got past about 83% there |
| the team of the touches we already find | +0.094 | most of the touch gain is picking the right team, not finding more touches |
| dead ball, everywhere | +0.078 | dead ball still matters |
| dead ball, inside gaps only | +0.056 | most of the dead-ball loss is in gaps |
| dead ball, where the ball is seen | +0.021 | the seen part is nearly solved |
| when possession changes (not who) | +0.009 | timing is not the problem |
| the whole ball track (no gaps) | 0.704 in total | almost all of the distance between our score and 0.70 is the missing ball |

**What we tried and dropped.** Each was run under the same check; the rows are in
`experiments/RESULTS.md`.

| Tried | Result (spellF1 change) | Why dropped |
|---|---|---|
| Ball: Kalman smoother; longer interpolation (up to 2 s) | +0.000 to +0.001 | Our gaps are too long to fill |
| Decoder with nearest-team evidence, or a plain switch cost | −0.012 to +0.008 | Noise; the distance margin beats both |
| Flat dead score (every gap leans dead) | +0.006 / +0.045 | No gain under random gaps: it turns live gaps into `D` |
| Velocity matching (a player moving with the ball) | +0.000 | No effect; already covered by the nearest-player evidence |
| Landing point of a fast ball | −0.019 to −0.035 (0/7) | Adds 40–60 possessions per match and precision falls by 0.06 to 0.12 |
| Team near the ball where it reappears after a long gap | −0.004 to 0.000 | The nearest team at that moment is often not the one with the ball |
| Team shape inside ball gaps, as a rule (the more spread-out team has it) | +0.000 to +0.004 | No better than keeping the last owner |
| Restart windows halved, or the tag clue off (with simulated tags) | +0.002 to +0.007 | Gains in all four runs, but on only 5 of 7 matches in at least one |
| Learned dead-ball score alone, or only inside gaps | wins under random gaps only | Worse under stoppage gaps; the blend fixed this |
| Dead-ball weight 0.85 instead of 0.7; dead-ball models refitted on the smoothed dev set | +0.004 / +0.001; +0.008 / −0.004 | Within noise, or mixed signs |
| Touch: smaller or larger velocity change (2.0, 1.5, 4.0 m/s); hold 0.5 s | −0.003 to −0.019 | 3 m/s and 0.8 s were best |
| Gap team model: on every frame, or as the only team evidence; weight above 0.25 | worse | The distance margin and touches are better where the ball is seen; a larger weight over-trusts an 83% model |
| Gap team model v6 (more timing inputs); the same inputs in a gradient-boosted model (LightGBM, a check only) | +0.002 / +0.002; gap accuracy 0.827 to 0.834 | Both stop at about 83%: the inputs, not the model, are the limit |
| Gap team model v4 without the team-shape inputs; minimum spell 2 s | worse | The shape inputs carry signal; 2 s removes real short possessions |
| Learned switch hazard (section 1.3) | +0.001 at best | Player movement does not show when possession changes |
| Lower decoder costs, evidence lag 12 frames | small losses | Covered by the cost sweep |
| Camera aim, broadcast fill, Smart Tagging phases as gap evidence | see section 2 | Too little new information, and none can be measured on IDSSE |

**On our match** (label-free counts from `scripts/label_free_check.py` and
`scripts/compare_submissions.py`; they show the output is sane and never choose between
versions).

| | Submitted |
|---|---|
| Rows | 199,319, no `X` |
| Graded halves: `H` / `A` / `D` | [FILL: H% / A% / D%, graded halves] |
| All frames: `H` / `A` / `D` | [FILL: H% / A% / D%, all frames] |
| Possessions | [FILL: number of possessions], mean [FILL: mean length] s |
| Outside the graded halves | every frame is `D` (40,462 frames, never scored) |

Each new version was compared with the one before it on our match. The dead-ball model v3
changed 1.5% of the frames; the touch model changed 2.3%, almost all `H` to `A` or `A` to `H`,
with the dead-ball share unchanged. For reference, the baseline configuration gives 669
possessions on our match; the public five minutes hold 17 possessions averaging 14.2 s, and
the IDSSE matches 370–511 possessions of 5.5–9.2 s each.

**Public five-minute check.** It was last scored on the configuration frozen on 15 November
(tag `v1.0-freeze`): spellF1 0.410 and macroF1 0.681, against 0.326 and 0.621 for the baseline
(bootstrap interval for spellF1: 0.277 to 0.544). Later versions were only checked to load and
score, never compared on it: with 17 possessions, one possession moves the score by about 0.06
(D9). It is also not a typical stretch of football: on the IDSSE matches, only 5.2% of 5-minute
windows are as calm (17 possessions or fewer, 14.2 s or longer on average;
`scripts/window_check.py`).

## 6. Limitations

**Where it fails.**

- **Possessions inside ball gaps.** Our ball is missing on 40.7% of graded frames, almost all of
  it in gaps longer than 5 s. The gap team model is right on about 83% of gap frames, and no
  input we tried pushed it further: the missing ball is information we do not have. A clean
  ball would take the dev score to 0.704.
- **Short possessions.** Touches recover many, but only where the ball is seen. The minimum
  spell of 1 s also removes some real possessions shorter than that.
- **Dead ball.** A perfect dead-ball call would still add +0.078 on the dev set, most of it
  inside gaps.
- **No ball height, no identities.** High balls and aerial duels cannot be seen directly; the
  fast-ball rule is a stand-in. Slots are tracklets that swap, so we cannot follow one player.
- **Outside the graded halves every frame is `D`.** This is a side effect of the dead-ball
  model, not a decision about those frames. They are never scored.

**How sure we are of the choices.**

- **Most of our later steps gained less than +0.04.** We adopted them because all four runs
  agreed, by team decision. Each is small enough that, on its own, part of it may be noise.
- **Many choices were made on the same 7 matches.** The learned models are scored
  leave-one-match-out, but the choice *between* variants (blend weights, model versions, costs,
  touch settings) was made on the same matches, so the total gain is likely somewhat flattered.
- **The dead-ball model assumes our gaps look like stoppage gaps.** The real mix on our match is
  unknown.
- **The possession count does not always move the same way on the two sets.** Some steps
  changed the count on our match in the opposite direction from the dev set: the learned team
  evidence gave more possessions on the dev set but fewer, longer ones on our match
  (`scripts/team_diag.py` compares the two, label-free). We cannot tell which is right without
  answers, and the dev set's own counts stay below its truth.

**What the dev set cannot tell us.**

- **Its truth is not the organisers' labelling.** It is the data provider's own possession
  flag, which may draw the line on loose balls and restarts differently.
- **It is another league and another tracking provider, with cleaner tracking.** Its players
  are identified and do not swap; ours averages 20.5 tracked objects per graded frame, with no
  identities and tracklets that swap. Smoothing makes its ball look more like ours, but only on
  a few statistics.
- **Simulated gaps and tags only bracket the real ones.** Random and stoppage cuts are the two
  extremes, and simulated tags are cleaner than Smart Tagging.
- **No full-match score before results day.** We do not claim a number for this match.

**What we would fix with more time.** Evidence from outside the tactical tracking for the ball
gaps: the broadcast tracking (it would fill about 9% of gap frames) and our own ball detection
on the video. Touch inputs for the dead-ball model; the decoder costs and the gap weight swept
again with the touch model on; the touch model trained on more seeds; and a dev set from a
second source, so that choices are not all made on the same 7 matches.

## 7. Reproduction: how to run it

Requirements: Python 3.12, `make`, `git`, and the organisers' data bundle in a folder
**outside** the repository. Dependencies are pinned to exact versions in `requirements.txt`
(numpy, pandas, PyYAML). No training is needed to reproduce the file: the fitted models are
small JSON files in `models/`.

```bash
git clone --recurse-submodules https://github.com/falafelona/more-than-a-hack-2027
cd more-than-a-hack-2027
git checkout [FILL: tag]                  # the submitted version
make setup                                # .venv and pinned dependencies
export MTAH_DATA_DIR=/path/to/mtah-data   # the data bundle, outside the repo
make submission                           # writes outputs/Falafelona.csv from configs/final.yaml
make validate                             # the official validate_submission.py
shasum -a 256 outputs/Falafelona.csv
```

| | |
|---|---|
| Output | `outputs/Falafelona.csv`, 199,319 rows, header `frame,label` |
| Expected SHA-256 | `[FILL: sha256]` |
| Official validator | OK |
| Tests | `make check`: lint, and 880 tests on synthetic data only (no match data needed) |

**Determinism.** A fixed seed, no order-dependent parallelism, and the models are applied, not
trained, at run time. We built the submitted file twice and got the identical SHA-256.
Earlier, at the `v1.0-freeze` tag, a clean clone with a fresh environment gave the
identical file on macOS and on Linux (Docker `python:3.12`).

**Repeating the dev-set work.** Download IDSSE [1] to a folder outside the repository, set
`IDSSE_DATA_DIR`, and run `scripts/devset_eval.py` with `--devset configs/devset-smooth.yaml`
(`experiments/README.md` explains the four runs). On the dev set, a config must point at the
leave-one-match-out model files (`models/*_lomo_{match_id}.json`), not at the `*_all.json` files
of the submission. To refit the models: `scripts/fit_deadball.py`, `scripts/fit_team.py` and
`scripts/fit_touch.py` (they read IDSSE only).

**Data handling.** The match data is not in the repository (D1, D2): it is read from
`MTAH_DATA_DIR`, and a check in the test suite refuses data-shaped files. AI coding assistants
were used to write code and documentation. Our rule is that they never read match data and
never run the pipeline on it: people run those commands and share only aggregates (`CLAUDE.md`,
D4, D16).
