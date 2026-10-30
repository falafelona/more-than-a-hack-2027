# Falafelona: instructions for AI coding agents

This file is also `AGENTS.md` (a symlink), so Claude Code, Codex, Cursor, Gemini and Copilot all read the same rules.

## 🚫 Data rules (competition T&Cs: breaking them can disqualify the team)
- **Never read, open, list, grep, cat, head or load match data**: anything under `$MTAH_DATA_DIR`, `~/mtah-data`, `/Volumes/mtah-data`, `data/`, `cache/` or `outputs/`, or any `*.mp4`, `*_atd.txt`, `*_atd.xml`, `*_smart_tagging.xml`, `*_homography.csv`, `*_possession_gt*.csv`, `*.pkl`, `*.npz` file.
- **Never run the real pipeline** (`make submission`, `python -m falafelona run`, `scripts/bootstrap_eval.py`) or any script pointed at real data. Humans run those.
- Only run code against **synthetic data** (`tests/synthetic.py`).
- If a task seems to need the real data, **stop and ask** a human to run it and report aggregate numbers.
- Never write code that prints or logs raw rows, coordinates, or per-frame values. Aggregates only.
- Never commit notebooks with outputs.

## Project
- Task: per-frame possession labels `H` (Barça) / `A` (Athletic) / `D` (dead ball) / `X` (abstain) → `outputs/Falafelona.csv`, header `frame,label`, exactly 199,319 rows.
- Ranked on **spellF1**. `vendor/challenge/evaluation/possession_metrics.py` is the spec. Baseline to beat: 0.2433.
- **Don't tune anything on the public 5-minute GT.** It's a pipeline check only (docs/decisions.md D9). Thresholds are tuned on other matches (D15).
- Stages and interfaces: `docs/architecture.md` · decisions: `docs/decisions.md` · workflow: `CONTRIBUTING.md`.

## Code conventions
- Python 3.12. Dependencies pinned exactly in `requirements*.txt`; add one only via a `chore/` PR.
- All thresholds and windows go in `configs/*.yaml`, never hard-coded.
- Deterministic: fixed seeds, no order-dependent parallelism.
- Each workstream owns its module (`ball.py`, `deadball.py`, `possession.py`, `pipeline.py`). Don't refactor other modules unless asked.
- **Never edit `vendor/challenge/`** (official code, pinned submodule).
- `make test` must pass before a PR. Use the PR template, and branch names follow `<type>/<ws>-<short-desc>`.
