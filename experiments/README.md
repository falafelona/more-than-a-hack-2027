# Experiments

Every run that changes the output gets one row in [RESULTS.md](RESULTS.md). **Aggregate numbers only.**

| Column | What to write |
|---|---|
| Commit | Short SHA |
| Change | What changed, in a few words |
| Dev spellF1 | Score on other matches. This is what we compare (D15) |
| Possessions | Count on this match. Label-free, sanity check only |
| Public check | `ok` or `bug` from the 5-minute pipeline check, never a score to compare (D9) |
| Verdict | `keep`, `drop` or `inconclusive` |

Differences below about 0.04 spellF1 are noise.

## Running on the dev set (IDSSE)

The dev set is the seven open IDSSE matches (Bassek et al., *Scientific Data* 12, 195, 2025; approved by the organisers in official issue #16). `scripts/devset_eval.py` reads them, cuts gaps into the ball track so it looks like ours, runs the pipeline with each config and scores it with the official metric. Humans run it, in a terminal with no AI tool open, like the other scripts that read match data. It prints totals only.

**One-time setup**

1. Download the seven matches (Figshare, or the Hugging Face mirror `pysport/idsse-data`) into a folder **outside the repo**, for example `~/idsse-data`. Per match you need the match information XML and the positions XML. Both a flat folder and one folder per match work.
2. `export IDSSE_DATA_DIR=~/idsse-data`
3. Check that the files load: `.venv/bin/python scripts/devset_eval.py --summary`. It prints shapes and label counts per match. If it stops with an error about an element or attribute name, run `.venv/bin/python scripts/devset_eval.py --inspect <one positions xml>` (and the same for a match information xml). It prints the element and attribute **names** only, no values. Paste that into the issue.
4. `configs/devset.yaml` already holds the gap numbers from the ball EDA of our match (issue #13, filled in #30). Change them only if the EDA is rerun.

**Comparing versions**

```
make devset                                                    # baseline vs final
make devset CONFIGS="configs/final.yaml configs/exp/x.yaml"    # any configs; the first is the reference
```

`make devset` runs `scripts/devset_eval.py --config <CONFIGS>`. Call the script directly for the flags below.

**Two kinds of gap cuts.** By default the gaps go anywhere in a half, so they land on dead ball only as often as by chance (like an occlusion in open play). `--gaps stoppage` (or `make devset GAPS=stoppage`) starts each gap as a dead-ball spell of the truth starts, up to `stoppage_lead_s` (1 s) earlier, like a ball lost as it goes out of play; a gap longer than the stoppage runs on into the play after it. Our real gaps are a mix of both, so the two bracket the truth. The report gives the share of cut frames that are dead ball, so you can see which kind ran. **A change goes into `final.yaml` only if it passes the Go/No-Go (D18): better under both kinds of gap and both seeds, on at least 6 of 7 matches in each of the four runs. A mean gain of at least +0.04 per run passes outright; a smaller one needs an explicit team decision recorded in RESULTS.md.** Write both numbers in the RESULTS row, e.g. `0.212 / 0.230`.

**The smoothed dev set (D27).** `configs/devset-smooth.yaml` is the same dev set with the IDSSE ball smoothed (Gaussian, sigma 0.25 s) before the gaps are cut, so its kicks look like ours. Every change after the decoder-cost change was judged on it: pass `--devset configs/devset-smooth.yaml` (and a seed-1 copy of it for the seed-1 runs). Its scores are lower than on `configs/devset.yaml` and are not comparable with the earlier RESULTS rows.

**Configs with learned models.** On the dev set a config must point at the leave-one-match-out files (`models/<name>_lomo_{match_id}.json`); `devset.py` fills in the held-out match. The `*-final.yaml` copies point at the `*_all.json` files and are only for building the submission.

**Oracles and diagnosis.** `--oracle MODE` runs every config a second time with one part replaced by the truth (dead ball everywhere, in gaps or where the ball is seen; every touch with its team; the team of the detected touches; who has the ball inside gaps; when possession changes). The gain is the ceiling for improving that part. `--clean-ball` skips the cutting altogether. `--diagnose` sorts missed and fake possessions by cause. All print aggregates only.

**Simulated restart tags.** IDSSE has no Smart Tagging, so the tag clues (`before_restart`, and `restart_tag_after` in `gap_chance`) give 0 on the dev set. `--tags` (or `make devset TAGS=1`) gives each match restart tags made from its own truth: one at the end of a dead-ball spell, typed by where the ball is when play restarts, as many per minute as on our match (the `tags:` section of `configs/devset.yaml`). The tags have their own seed, so the cut ball is the same with or without them. They are cleaner than real ones (types always right, no false tags, timing assumed), so a win that needs them is weaker evidence: write `(tags)` in the RESULTS row. To judge the tag clues: `make devset TAGS=1 CONFIGS="configs/final.yaml configs/exp/poss-hmm-margin-dead-gap.yaml configs/exp/poss-hmm-margin-dead-gap-tags.yaml"`, under both gap kinds.

The first config is the reference. The report gives spellF1 and macroF1 per match, the mean, and the paired difference of every other config against the first. Every config sees the same cut ball (fixed seed), so the difference is not luck. `--clean-ball` skips the cutting and shows what the gaps cost; `--matches J03WPY J03WMX` runs a few matches for a quick check. The first run parses the XML (a minute or so per match) and caches the arrays under `$IDSSE_DATA_DIR/cache/`; later runs are quick.

**What the dev score means.** It compares versions. It does not predict the real score. The truth is the data provider's own possession flag (`BallPossession` while `BallStatus` is alive, otherwise dead ball), not the organisers' labels. The gaps are random, real gaps are not. Players come straight from tracking, with no tracklet swaps. Thresholds chosen here are still checked against these limits (D15).
