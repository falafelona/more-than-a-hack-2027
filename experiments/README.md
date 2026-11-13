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

**Two kinds of gap cuts.** By default the gaps go anywhere in a half, so mostly over live play (like an occlusion). `--gaps stoppage` (or `make devset GAPS=stoppage`) puts each gap over one dead-ball spell of the truth instead, like a ball lost while it is out of play. Our real gaps are a mix of both, so the two bracket the truth. The report gives the share of cut frames that are dead ball, so you can see which kind ran. **A change goes into `final.yaml` only if it wins by more than 0.04 under both.** Write both numbers in the RESULTS row, e.g. `0.212 / 0.230`.

The first config is the reference. The report gives spellF1 and macroF1 per match, the mean, and the paired difference of every other config against the first. Every config sees the same cut ball (fixed seed), so the difference is not luck. `--clean-ball` skips the cutting and shows what the gaps cost; `--matches J03WPY J03WMX` runs a few matches for a quick check. The first run parses the XML (a minute or so per match) and caches the arrays under `$IDSSE_DATA_DIR/cache/`; later runs are quick.

**What the dev score means.** It compares versions. It does not predict the real score. The truth is the data provider's own possession flag (`BallPossession` while `BallStatus` is alive, otherwise dead ball), not the organisers' labels. The gaps are random, real gaps are not. Players come straight from tracking, with no tracklet swaps. Thresholds chosen here are still checked against these limits (D15).
