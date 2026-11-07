"""Shared pytest fixtures. Everything here is synthetic: no test ever reads real match data.

- `bundle`: a synthetic tracking bundle in a temp folder, with MTAH_DATA_DIR pointing at it
- `official_baseline` (task 1.5): vendor/challenge/starter/baseline.py, imported unchanged
"""

import pytest
from synthetic import write_bundle


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Write a small synthetic bundle and point MTAH_DATA_DIR at it. Returns what was written."""
    expected = write_bundle(tmp_path, n_frames=500, seed=0)
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    expected["root"] = tmp_path
    return expected
