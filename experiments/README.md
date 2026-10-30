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
