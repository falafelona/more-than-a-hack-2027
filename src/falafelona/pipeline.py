# ws:int: glue the stages together and write the submission file.
#
#   load (io) -> clean ball (ball) -> dead-ball score (deadball) -> H/A/D labels (possession)
#   -> X policy (not agreed yet: docs/decisions.md D14)
#   -> outputs/Falafelona.csv: header "frame,label", exactly 199,319 rows, frames 0..199318
#
# Must be deterministic: fixed seeds from the config, no order-dependent parallelism.
# Returns the SHA-256 of the written file and aggregate label counts (safe to print).

import hashlib
from pathlib import Path

import numpy as np

from falafelona import deadball, possession, vendor
from falafelona.ball import DataCleaner
from falafelona.io import load_match
from falafelona.vendor import games

# The submission contract: one row per frame of the tactical video.
N_ROWS = games.GAMES[games.GAME]["n_video_frames"]
LABELS = ("H", "A", "D", "X")


def apply_x_policy(labels, match, cfg):
    """Where to abstain. Only `none` exists until D14 is agreed: no X is written."""
    policy = cfg["output"]["x_policy"]
    if policy == "none":
        return labels
    raise ValueError(f"unknown x_policy {policy!r}; available: none")


def write_submission(path, labels):
    """Write the frame,label CSV with the organisers' own writer, then return its SHA-256.

    Using starter/baseline.py's write_submission means our format is theirs by construction.
    """
    vendor.baseline.write_submission(str(path), labels)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_counts(labels):
    """Aggregate counts per label: safe to print and share."""
    return {label: int(np.sum(labels == label)) for label in LABELS}


def run(cfg, out, n_frames=None):
    """Run every stage, write the CSV to `out`, return what was built plus sha256 and counts.

    n_frames is for tests on the small synthetic bundle; the real run leaves it as None, and
    then the file must have exactly N_ROWS rows.
    """
    match = load_match(n_frames=n_frames)
    cleaner = DataCleaner(match, cfg)
    cleaned = cleaner.return_cleaned_data()

    dead_score = deadball.detect(cleaned, cleaner.track, cfg)
    labels = possession.predict(cleaned, dead_score, cfg)
    labels = apply_x_policy(labels, cleaned, cfg)

    expected = N_ROWS if n_frames is None else n_frames
    if len(labels) != expected:
        raise RuntimeError(f"pipeline produced {len(labels)} labels, expected {expected}")

    sha256 = write_submission(out, labels)
    return {
        "match": match,
        "cleaned": cleaned,
        "ball": cleaner.track,
        "dead_score": dead_score,
        "labels": labels,
        "rows": len(labels),
        "counts": label_counts(labels),
        "sha256": sha256,
        "path": Path(out),
    }
