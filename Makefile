# One place for every command.
#
#   make setup       create .venv (Python 3.12), install pinned deps, init the submodule,
#                    enable the git hooks and print where the data must go (CONTRIBUTING.md)
#   make test        run the tests (synthetic data only, safe for anyone and for CI)
#   make lint        ruff on our code (vendor/ excluded)
#   make submission  build outputs/Falafelona.csv from configs/final.yaml
#   make validate    run the official validate_submission.py on outputs/Falafelona.csv
#   make check       lint and test: run before every PR
#   make devset      compare configs on the IDSSE dev set (experiments/README.md). The first
#                    config is the reference. Default: baseline vs final. Another pair:
#                    make devset CONFIGS="configs/final.yaml configs/exp/x.yaml"
#
# `submission`, `validate` and `devset` read real data or real outputs: humans run them, not AI agents.

PYTHON  ?= python3.12
VENV    := .venv
PY      := $(VENV)/bin/python
CONFIG  ?= configs/final.yaml
OUT     ?= outputs/Falafelona.csv
CONFIGS ?= configs/baseline.yaml configs/final.yaml

.PHONY: setup test lint check submission validate devset clean

setup:
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), "need Python 3.12, got " + sys.version.split()[0]' \
		|| { echo "Install Python 3.12 or run: make setup PYTHON=/path/to/python3.12"; exit 1; }
	test -x $(PY) || $(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements-dev.txt
	$(PY) -m pip install --quiet --no-deps -e .
	git submodule update --init
	git config core.hooksPath .githooks
	@echo ""
	@echo "Setup done. Keep the match data OUTSIDE this repo and point to it:"
	@echo "    export MTAH_DATA_DIR=~/mtah-data"
	@echo "Never unpack it into data/ or vendor/challenge/data/ (CONTRIBUTING.md)."

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

check: lint test

submission:
	@test -n "$$MTAH_DATA_DIR" || { echo "MTAH_DATA_DIR is not set (CONTRIBUTING.md)"; exit 1; }
	$(PY) -m falafelona run --config $(CONFIG) --out $(OUT)

validate:
	$(PY) vendor/challenge/evaluation/validate_submission.py $(OUT)

devset:
	@test -n "$$IDSSE_DATA_DIR" || { echo "IDSSE_DATA_DIR is not set (experiments/README.md)"; exit 1; }
	$(PY) scripts/devset_eval.py --config $(CONFIGS)

clean:
	rm -rf .pytest_cache .ruff_cache build src/*.egg-info
	find . -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
