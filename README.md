# Falafelona: More Than a Hack 2027

Team **Falafelona**'s entry for [More Than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports × FC Barcelona / Barça Innovation Hub × Mobile World Capital Barcelona).

**Team:** Chikhani · Tala · Dudu · Laila · Abdallah

**Status:** submitted. The submission is tag [`v1.2`](https://github.com/falafelona/more-than-a-hack-2027/tree/v1.2), built from `configs/final.yaml`: 199,319 rows, official validator OK, SHA-256 `ea403a67c0ce2ea67b00decb50a766781c0c1bf51762f56f7d4926263b0e43a2`. Every setting was chosen on the IDSSE dev set, never on this match. On that dev set it scores spellF1 0.455 to 0.519 across our four runs, against 0.201 to 0.269 for the baseline configuration. The organisers' baseline on this match: spellF1 **0.2433**.

**The technical document is [docs/Falafelona.md](docs/Falafelona.md).**

## The task

For every frame of FC Barcelona vs Athletic Club (LaLiga 2024-25), say who has the ball:

| Label | Meaning |
|---|---|
| `H` | Barcelona in possession, ball alive |
| `A` | Athletic in possession, ball alive |
| `D` | Dead ball |
| `X` | No answer |

The output is one CSV with header `frame,label` and exactly 199,319 rows, frames `0 … 199318` in order.

The tracking is automatic and uncorrected, has no player identities, and the ball is missing on 41% of graded frames.

## How it is scored

- **Ranking metric: `spellF1`.** Each possession is one object, matched one-to-one with the true possessions by temporal overlap. Flicker destroys it.
- **Tie-break: `macroF1`**, per-frame F1 over H/A/D.
- **Graded frames:** `11,185–90,651` and `110,808–190,197`. Everything else is ignored.

## How it works

```
tracking data + Smart Tagging
  → clean the ball track            ball.py
  → dead-ball score                 deadball.py     (one clue is a small fitted model: learned_dead.py, models/)
  → per-frame team evidence         possession.py   (distance margin, touches, restarts; velocity.py)
  → H/A/D decoder                   possession.py
  → final labels                    pipeline.py
  → Falafelona.csv
```

**The decoder is the centre of the system.** The features say what is likely on each frame, and the decoder chooses the best whole sequence. Every switch has a cost, so brief wrong evidence cannot create a false possession, and flicker disappears.

The decoder uses three states because every graded frame is `H`, `A` or `D`. We write no `X` anywhere: it cannot raise spellF1, and on a graded frame it always counts as a miss for macroF1 (D14).

**How settings are chosen.** We have no answers for this match and may not tune on it, so every setting is chosen on the seven open IDSSE matches (dev set), with the ball cut to look like ours. A change is kept only if it is better in all four dev-set runs (two kinds of simulated gaps, two seeds), on at least 6 of the 7 matches in each (D17, D18). A mean gain of at least +0.04 spellF1 per run passes outright; a smaller one goes in only by an explicit team decision. That happened twice, and the technical document marks both: switch placement (D22) and the dead-score blend, taken by exception (D24). Every experiment, kept or dropped, is a row in [experiments/RESULTS.md](experiments/RESULTS.md).

**Training data.** The only labelled data is the open IDSSE dataset (Bassek et al., *Scientific Data* 12, 195, 2025), approved by the organisers. Its labels come with the dataset; we labelled nothing by hand, on this match or any other. It is used to choose settings and to fit one small logistic model for the dead-ball score, whose fitted numbers are in [`models/`](models/README.md).

See [docs/architecture.md](docs/architecture.md) for the stages, [docs/decisions.md](docs/decisions.md) for why, and [docs/Falafelona.md](docs/Falafelona.md) for the technical document.

## Reproduce our submission

Needs Python 3.12, `make`, `git`, and the organisers' data bundle in a folder **outside** this repo.

```bash
git clone --recurse-submodules https://github.com/falafelona/more-than-a-hack-2027
cd more-than-a-hack-2027
git checkout v1.2                         # the submitted version
make setup                                # .venv and pinned dependencies
export MTAH_DATA_DIR=/path/to/mtah-data   # the organisers' data bundle, outside this repo
make submission                           # writes outputs/Falafelona.csv from configs/final.yaml
make validate                             # the official validate_submission.py
shasum -a 256 outputs/Falafelona.csv
```

Expected: `outputs/Falafelona.csv`, 199,319 rows, validator OK, SHA-256
`ea403a67c0ce2ea67b00decb50a766781c0c1bf51762f56f7d4926263b0e43a2`.

The run is deterministic and needs no training: the fitted model is a small JSON file in `models/`. We built the file twice and got the identical SHA-256.

`make check` runs the linter and the tests (synthetic data only, no match data needed). To repeat the dev-set comparisons, see [experiments/README.md](experiments/README.md).

## Rules we work under

- **No hand labels on this match**, for any purpose, including validation.
- **The public 5 minutes check the pipeline only.** They are never used to train, tune or choose between versions.
- **Thresholds are chosen on other matches.**
- **Match data never enters this repo** and never goes to a hosted AI tool.

The official rules are the authority: [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

Working on this repo? Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Licence

Code and documentation: MIT, see [LICENSE](LICENSE). This covers our own work only. The match data is © its owners, licensed for this competition only, and not included here. `vendor/challenge/` is the organisers' code under its own licence. The fitted model files in `models/` were fitted on the IDSSE dataset (CC BY 4.0, cited in the technical document).
