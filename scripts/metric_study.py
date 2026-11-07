# ws:eval: metric study (task 5.5). How much each kind of mistake costs on spellF1.
#
# WHAT: makes up a "true" match, damages a copy in one way at a time (switching late, flicker,
# false dead balls) and scores each copy with the official metric. The table tells the poss
# team which mistakes hurt most.
# HOW to run it:  .venv/bin/python scripts/metric_study.py


import numpy as np

from falafelona.vendor import possession_metrics as pm  # the organisers' scorer, unchanged

FPS = 25   # frames per second
SEED = 0   # same random numbers every run, so the table is always the same


# Invents the "correct answer" for a fake match: 300 possessions of about 15 s each,
# alternating Barça (H) and Athletic (A), with a dead ball (D) after 30% of them.
def make_truth(seed=SEED):
    """A made-up match: 300 possessions of ~15 s, alternating H/A, 30% followed by a dead ball."""
    rng = np.random.default_rng(seed)
    labels = []
    team = "H"
    for _ in range(300):
        labels += [team] * (FPS + int(rng.exponential(14 * FPS)))  # at least 1 s, ~15 s average
        if rng.random() < 0.3:
            labels += ["D"] * int(rng.uniform(3, 30) * FPS)       # 3-30 s stoppage
        team = "A" if team == "H" else "H"
    return np.array(labels)


# Mistake 1, reacting too slowly: slides the whole answer right, so every switch
# (H to A, A to D, ...) happens `frames` too late.
def late(truth, frames):
    """Every switch happens `frames` too late (the whole answer slides right)."""
    return np.concatenate([truth[:frames], truth[:-frames]])


# Mistakes 2 and 3, tiny wrong moments: puts short blips inside live play. With
# label="other" the blip is the wrong team (flicker); with label="D" it is a fake dead ball.
def blips(truth, label, per_minute, length, seed=SEED):
    """Put short wrong blips inside possessions: the other team (flicker) or D (false dead ball)."""
    rng = np.random.default_rng(seed)
    pred = truth.copy()
    minutes = len(truth) / FPS / 60
    for start in rng.integers(0, len(truth) - length, int(per_minute * minutes)):
        if truth[start] == "D":
            continue  # only damage live play
        other = "A" if truth[start] == "H" else "H"
        pred[start:start + length] = other if label == "other" else "D"
    return pred


# Grades a damaged copy against the truth with the official scorer. Returns spellF1
# (whole possessions right? the ranking metric) and macroF1 (frames right? the tie-break).
def score(truth, pred):
    """(spellF1, macroF1) from the official metric."""
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred))
    return result["spell_f1_mean"], result["macro_f1"]


# Runs the whole study: makes the truth, builds 1 perfect and 10 damaged copies,
# scores each one and prints the table.
def main():
    truth = make_truth()
    cases = [("perfect copy", truth)]
    cases += [(f"switches {f} frames late", late(truth, f)) for f in (3, 8, 25, 50)]
    cases += [(f"flicker: {n}/min, 3 frames", blips(truth, "other", n, 3)) for n in (0.5, 1, 2)]
    cases += [(f"false D: {n}/min, 10 frames", blips(truth, "D", n, 10)) for n in (0.5, 1, 2)]

    print(f"{'mistake':<32} spellF1  macroF1")
    for name, pred in cases:
        spell, macro = score(truth, pred)
        print(f"{name:<32} {spell:7.3f}  {macro:7.3f}")


if __name__ == "__main__":
    main()