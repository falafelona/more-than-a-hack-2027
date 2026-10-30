# Falafelona: More Than a Hack 2027

Team **Falafelona**'s entry for [More Than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports × FC Barcelona / Barça Innovation Hub × Mobile World Capital Barcelona).

**Team:** Chikhani · Tala · Dudu · Laila · Abdallah

**Status:** pipeline in progress. Baseline to beat: spellF1 **0.2433**.

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
tracking data
  → clean the ball track            ball.py
  → dead-ball score                 deadball.py
  → per-frame team evidence         possession.py
  → H/A/D decoder                   possession.py
  → final labels                    pipeline.py
  → Falafelona.csv
```

**The decoder is the centre of the system.** The features say what is likely on each frame, and the decoder chooses the best whole sequence. Every switch has a cost, so brief wrong evidence cannot create a false possession, and flicker disappears.

The decoder uses three states because every graded frame is `H`, `A` or `D`. Where we write `X` is still being decided (D14).

See [docs/architecture.md](docs/architecture.md) for the stages and [docs/decisions.md](docs/decisions.md) for why.

## Reproduce our submission

> Commands are placeholders until the pipeline is complete.

```bash
git clone --recurse-submodules https://github.com/falafelona/more-than-a-hack-2027
cd more-than-a-hack-2027
make setup
export MTAH_DATA_DIR=/path/to/mtah-data
make submission
make validate
```

Expected: 199,319 rows · SHA-256 `<fill in at freeze>` · runtime `<N> min` on `<machine>`.

## Rules we work under

- **No hand labels on this match**, for any purpose, including validation.
- **The public 5 minutes check the pipeline only.** They are never used to train, tune or choose between versions.
- **Thresholds are chosen on other matches.**
- **Match data never enters this repo** and never goes to a hosted AI tool.

The official rules are the authority: [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

Working on this repo? Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Licence

Code: _TBD_. The match data is © its owners, licensed for this competition only, and not included here.
