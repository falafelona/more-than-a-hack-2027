# ws:int: glue the stages together and write the submission file.
#
#   load (io) -> clean ball (ball) -> dead-ball score (deadball) -> H/A/D labels (possession)
#   -> X policy (not agreed yet: docs/decisions.md D14)
#   -> outputs/Falafelona.csv: header "frame,label", exactly 199,319 rows, frames 0..199318
#
# Must be deterministic: fixed seeds from the config, no order-dependent parallelism.
# Will also compute the SHA-256 of the written file and aggregate label counts (safe to print).

from falafelona.ball import DataCleaner
from falafelona.io import load_match
from falafelona.possession import Possession


def run(cfg, n_frames=None):
    """Run the stages in order (PR #9 structure). Nothing happens at import time.

    Not finished (task 1.5): the dead-ball stage, the H/A/D labels, the X policy and the CSV
    are still to come. For now it returns the stage objects so the wiring can be tested.
    n_frames is for tests on the small synthetic bundle; the real run leaves it as None.
    """
    raw_match_data = load_match(n_frames=n_frames)
    data_cleaner = DataCleaner(raw_match_data)
    cleaned_data = data_cleaner.return_cleaned_data()
    possession_calc = Possession(cleaned_data)
    return {"match": raw_match_data, "cleaned": cleaned_data, "possession": possession_calc}
