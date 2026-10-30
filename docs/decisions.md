# Decision log

One line per technical decision we don't want to re-argue. Numbers are stable, so references in the code keep working.

**Status:** ✅ accepted · 💬 proposed · ❌ reversed

| # | Date | Decision | Why | Status |
|---|---|---|---|---|
| D1 | 2026-10-22 | Match data lives **outside the repo**, located via `MTAH_DATA_DIR` | Keeps data out of git and away from AI tools; the organisers' `games.py` already uses `MTAH_DATA_DIR` | ✅ |
| D2 | 2026-10-22 | Derived frame-level files (caches, features) go under `$MTAH_DATA_DIR/cache/`, never in the repo | They count as data under the terms | ✅ |
| D3 | 2026-10-22 | Official challenge code comes in as a **pinned git submodule** at `vendor/challenge/`, never edited | Validator and metric stay identical to the official scorer | ✅ |
| D4 | 2026-10-22 | `CLAUDE.md`, `AGENTS.md` (symlink) and per-tool deny lists in the repo | Every agent gets the same data rules. They reduce risk but do not remove it (see D16, proposed) | ✅ |
| D7 | 2026-10-30 | Results are **one table** in `experiments/RESULTS.md`, aggregates only | Simple enough to keep up with; feeds the technical doc. Replaces one YAML per run | ✅ |
| D9 | 2026-10-30 | The public 5-minute GT **never chooses between versions**: no training, tuning or comparing by CI on it | It is a rule, and it picks the wrong model (`smooth125` over `smooth50`). One possession moves it by about 0.06 | ✅ |
| D10 | 2026-10-30 | Python 3.12, exact-pinned `requirements.txt`, pip. One command: `make submission` | Reproduction must give the identical file; pip needs no extra install | ✅ |
| D11 | 2026-10-30 | Results record dev-set scores on other matches, plus label-free stats on this match; no full-match score is claimed | We have no full-match ground truth before results day | ✅ |
| D13 | 2026-10-30 | The **H/A/D decoder is the core design**, and its main purpose is removing flicker. Dead ball is a score fed to it, not a hard override | spellF1 scores whole possessions; a false `D` splits one, so team and dead-ball decisions belong in one model | ✅ |
| D14 | 2026-10-30 | `X` is written **only on ungraded frames**, as a final step | The scorer fills `X` with our last label for spellF1, so writing that label gives the same spellF1, and `X` always misses on macroF1 | 💬 |
| D15 | 2026-10-30 | All thresholds are **tuned on other matches**, starting with IDSSE | Hand labels and tuning on this match are banned, and organisers approved IDSSE (official issue #16) | ✅ |
| D16 | 2026-10-30 | Data lives on an **encrypted disk image, ejected while AI tools run** | Rules and ignore files only reduce risk; an unmounted disk cannot be read. May not be practical for every teammate | 💬 |
