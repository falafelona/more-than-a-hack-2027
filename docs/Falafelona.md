# Falafelona: technical document

Deliverable 3. A few pages. Exported to Falafelona.pdf only at submission time (PDFs are gitignored).

Each section has one owner, who writes it and opens a `docs/<ws>-...` PR.
Laila (eval) collects the sections, edits for one voice and exports the PDF (task 5.7b, 16 Nov).
Aggregates only: no rows, coordinates or screenshots of match data.

## 1. Approach
Owner: Dudu + Abdallah (task 4.8)
The pipeline end to end, and why the H/A/D decoder is the centre of it (D13).

### 1.1 Ball track
Owner: Abdallah (task 2.7)
How we clean the ball: false detections removed, gaps filled, and methods we tried and dropped.

### 1.2 Dead ball and restarts
Owner: Chikhani (task 3.8)
How the dead-ball score is built and how restarts are found.

Draft notes (to be written up by 15 Nov):
- The dead-ball stage gives every frame a score from 0 (alive) to 1 (dead). It is evidence for the
  H/A/D decoder, not a final answer (D13). The method is chosen in the `deadball:` section of the config.
- `ball_missing` (the official baseline): dead wherever the tracking has no ball. Simple, but the ball
  is also missing in live play, so it calls far too much dead ball.
- `out_of_bounds`: dead where the ball is past a touchline or goal line by more than `out_margin`
  (1 m for now). The margin is there because tracking is noisy near the lines, and a ball on the line
  is still in play. A missing ball scores 0 here: not seeing the ball says nothing about where it is.
- Still to come: players standing still, Smart Tagging restarts, the second-half kickoff, and one
  combined score.

## 2. Assets used (and dropped)
Owner: Tala
Which provided files we use (tracking, Smart Tagging, homography, video) and which we don't, and why.

## 3. Training data (required)
Owner: Laila

We use no hand labels and no annotations of this match, and nothing is trained or tuned on it.
The public five minutes are used only as a pipeline check (D9).

**Dev set.** Thresholds are chosen on the IDSSE dataset (D15): seven full professional matches
from the German Bundesliga and 2. Bundesliga, 2022/23, with tracking at 25 Hz [1]. The organisers
approved its use in the official challenge repository (issue #16).

- **Truth.** IDSSE tracking gives, for every frame, whether the ball is alive and which team
  has it. Ball alive gives H or A by that flag; ball dead gives D (D17).
- **Making it as hard as our match.** IDSSE's ball is almost always tracked; ours is missing on
  about 41% of graded frames. We remove the ball in random gaps whose lengths follow those
  measured on our match, with a fixed seed so every version sees the same gaps (D17).
- **Use.** Every candidate version runs on the seven matches and is scored with the official
  metric. The dev score is used to compare versions, not to predict our final score.

[1] Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer." Scientific Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

## 4. Possession rules
Owner: Dudu + Abdallah (task 4.8)
How our labels follow the official possession definitions in CHALLENGE.md.

## 5. Results
Owner: Laila

We have no ground truth for the full match, so we report three kinds of evidence (D11): the
dev-set score, label-free statistics on this match, and the public five-minute check.

**Reference: the official baseline (`baseline_smooth50`).**

| | Baseline |
|---|---|
| Full-match spellF1 / macroF1 (published by the organisers) | 0.2433 / 0.6724 |
| Public 5-min check, spellF1 / macroF1 | 0.326 / 0.621 |
| Possessions on this match | 669 |
| Possessions per graded minute | 5.61 |
| Median possession length | 4.7 s |
| Possessions shorter than 1 s | 132 |
| Agreement with Smart Tagging's attacking phases | 0.68 |

A real match has about 3.4 possessions per minute (the public five minutes have 17, averaging
14.2 s). The baseline reports far more, mostly short ones: it splits real possessions.

**What the metric rewards (metric study, synthetic labels).** We damaged a synthetic match one way
at a time and scored it with the official metric. Switching up to 8 frames late costs almost
nothing (spellF1 0.98), and one second late costs 0.13. Two 3-frame flicker blips per minute
cut spellF1 to 0.61 while macroF1 stays at 0.997. This is why our design centres on a decoder
that removes flicker (D13).

**Our final version.** [fill after the 14 Nov freeze: dev spellF1 vs the baseline, and the same
label-free stats as above. One row per kept change from experiments/RESULTS.md.]

## 6. Limitations
Owner: Laila (collects one or two lines from each owner)

- **The dev score is a comparison, not a prediction.** IDSSE truth is the provider's possession
  flag, not the organisers' labels, and its matches have no tracklet identity swaps.
- **Simulated ball gaps.** Our gaps are random with realistic lengths; real gaps cluster in
  crowded areas and on high balls, which the dev set does not reproduce.
- **No full-match score before results day.** The public five minutes hold 17 possessions, so
  one possession moves the score by about 0.06; it is a pipeline check only (D9).
- **Smart Tagging is evidence, not truth.** It is automatic and unreviewed, and every throw-in
  has no team.
- [ball: one or two lines from Abdallah]
- [dead ball: one or two lines from Chikhani]
- [possession: one or two lines from Dudu + Abdallah]

## 7. Reproduction
Owner: Tala
One command, the SHA-256 of the file, runtime and machine.