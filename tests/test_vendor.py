# The official challenge code is importable through falafelona.vendor, unchanged.
from falafelona import vendor


def test_official_modules_import():
    assert vendor.games.source_frames("tactical") == 199_319
    assert hasattr(vendor.possession_metrics, "validate_submission")
    assert hasattr(vendor.possession_metrics, "spell_scores")


def test_official_scripts_present():
    assert vendor.VALIDATOR.is_file()
    assert vendor.OFFICIAL_BASELINE.is_file()
