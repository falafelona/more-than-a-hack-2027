# One place for every command. Planned targets:
#
#   make setup       create .venv (Python 3.12), install pinned deps, init the submodule,
#                    enable the git hooks and print where the data must go (CONTRIBUTING.md)
#   make test        run the tests (synthetic data only, safe for anyone and for CI)
#   make submission  build outputs/Falafelona.csv from configs/final.yaml
#   make validate    run the official validate_submission.py on outputs/Falafelona.csv
#
# `submission` needs MTAH_DATA_DIR and is run by humans, not agents.
