# ws:eval: tests for the dev set (task N5, team-board issue #23). Synthetic data only.
#
# The dev set reads other matches (IDSSE), degrades their ball track to look like ours, runs the
# pipeline stages and scores them with the official metric. Here every file is made up in the
# IDSSE layout by tests/synthetic_idsse.py, where the right labels are known:
#   - the loader must give exactly what was written (positions, truth, slots, halves)
#   - a pipeline with a complete ball must score 1.0, so the scoring path is right
#   - cutting gaps must hurt the baseline, and a cleaning method must win some of it back, so
#     the harness can tell methods apart
#   - the report must show the mean over matches and every match on its own, aggregates only

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synthetic_idsse as syn

from falafelona import devset, possession, vendor
from falafelona.io import Match

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "devset_eval.py"
spec = importlib.util.spec_from_file_location("devset_eval", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

PAD = devset.HALFTIME_PAD
SHIFT = np.array(syn.PITCH) / 2.0


def pipeline_cfg(ball=None, window=0):
    """A pipeline config for the scripted game: dead ball = ball past the line, raw labels."""
    return {
        "seed": 0,
        "ball": ball or {"method": "none"},
        "deadball": {"method": "out_of_bounds", "out_margin": 1.0},
        "possession": {"method": "nearest_majority", "window": window, "dead_threshold": 0.5},
        "output": {"x_policy": "none"},
    }


DEGRADE = {
    "missing_share": 0.3,
    "gaps": [
        {"min_s": 0.0, "max_s": 0.5, "share": 0.1},    # 1-12 frames
        {"min_s": 0.5, "max_s": 2.0, "share": 0.3},    # 13-50 frames
        {"min_s": 2.0, "max_s": 8.0, "share": 0.6},    # 51-200 frames
    ],
}
SHORT_GAPS = {   # all gaps 6-12 frames: a cleaning method can fill every one
    "missing_share": 0.25,
    "gaps": [{"min_s": 0.2, "max_s": 0.5, "share": 1.0}],
}


@pytest.fixture
def idsse(tmp_path):
    """A synthetic IDSSE match in tmp_path. Returns what was written."""
    return syn.write_idsse(tmp_path, "J03WTEST")


def long_match(n_half=9000, pad=2000, seed=0):
    """A match built directly (no files): two long halves with a pad, a seen ball everywhere."""
    rng = np.random.default_rng(seed)
    n = 2 * n_half + pad
    period = np.zeros(n, dtype=np.int8)
    period[:n_half] = 1
    period[n_half + pad:] = 2
    ball = rng.uniform([5.0, 5.0], [100.0, 63.0], size=(n, 2))
    ball[period == 0] = np.nan
    return Match(
        n_frames=n,
        fps=25.0,
        period=period,
        ball_xy=ball,
        ball_speed=np.where(np.isnan(ball[:, 0]), np.nan, 5.0),
        players_xy=np.full((n, 2, 2), np.nan),
        players_speed=np.full((n, 2), np.nan),
        slot_team=np.array(["H", "A"]),
        pitch=(105.0, 68.0),
    )


def gap_lengths(ball_xy, period):
    """Length of every run of missing ball inside the graded frames, the slow obvious way."""
    lengths, run = [], 0
    for i in range(len(ball_xy)):
        missing = period[i] > 0 and np.isnan(ball_xy[i, 0])
        if missing:
            run += 1
        else:
            if run:
                lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


# --- finding the files -----------------------------------------------------------------------


def test_find_files_by_match_id(tmp_path):
    syn.write_idsse(tmp_path / "one", "J03WAAA")
    syn.write_idsse(tmp_path / "two", "J03WBBB")

    info, positions = devset.find_files(tmp_path, "J03WBBB")

    assert info == tmp_path / "two" / syn.info_name("J03WBBB")
    assert positions == tmp_path / "two" / syn.positions_name("J03WBBB")


def test_find_files_works_in_one_flat_folder(tmp_path):
    syn.write_idsse(tmp_path, "J03WAAA")
    syn.write_idsse(tmp_path, "J03WBBB")

    info, _ = devset.find_files(tmp_path, "j03waaa")   # the id is matched ignoring case

    assert info.name == syn.info_name("J03WAAA")


@pytest.mark.parametrize("case", ["missing", "twice"])
def test_find_files_needs_exactly_one_pair(tmp_path, case):
    syn.write_idsse(tmp_path / "a", "J03WAAA")
    if case == "twice":
        syn.write_idsse(tmp_path / "b", "J03WAAA")
        wanted = "J03WAAA"
    else:
        wanted = "J03WZZZ"

    with pytest.raises(FileNotFoundError, match="expected exactly one"):
        devset.find_files(tmp_path, wanted)


# --- the match information -------------------------------------------------------------------


def test_match_info_gives_pitch_teams_and_players(idsse):
    info = devset.read_match_info(idsse["info_path"])

    assert info["pitch"] == syn.PITCH
    assert info["teams"] == {"H": syn.HOME_TEAM, "A": syn.AWAY_TEAM}
    assert info["players"][syn.HOME_IDS[3]] == "H"
    assert info["players"][syn.SUB_ID] == "H"
    assert info["players"][syn.AWAY_IDS[10]] == "A"
    assert len(info["players"]) == 23           # the referee is not in the match information


def test_match_info_without_pitch_gives_none(tmp_path):
    data = syn.write_idsse(tmp_path, pitch_in_info=False)

    assert devset.read_match_info(data["info_path"])["pitch"] is None


def test_match_info_needs_a_home_and_a_guest_team(tmp_path):
    path = tmp_path / "info.xml"
    path.write_text('<Info><Team Role="home" TeamId="X"><Player PersonId="p"/></Team></Info>')

    with pytest.raises(ValueError, match="home and a guest"):
        devset.read_match_info(path)


def test_xml_namespaces_and_child_element_values_are_understood(tmp_path):
    """The reader matches local tag names, and takes a value from a child if it is no attribute."""
    path = tmp_path / "info.xml"
    path.write_text(
        '<Info xmlns="urn:test"><Env PitchX="100" PitchY="64"/>'
        '<Team Role="home" TeamId="H"><Player PersonId="a"/></Team>'
        '<Team Role="guest" TeamId="G"><Player PersonId="b"/></Team></Info>'
    )
    info = devset.read_match_info(path)
    assert info["pitch"] == (100.0, 64.0) and info["players"] == {"a": "H", "b": "A"}

    import xml.etree.ElementTree as ET
    frameset = ET.fromstring(
        "<FrameSet><Frame><N>7</N><X>1.5</X><Y>2.5</Y><S>3</S>"
        "<BallStatus>1</BallStatus><BallPossession>2</BallPossession></Frame></FrameSet>"
    )
    frames = devset._read_frames(frameset)
    assert list(frames["n"]) == [7] and list(frames["x"]) == [1.5]
    assert list(frames["status"]) == [1] and list(frames["possession"]) == [2]


# --- loading a match -------------------------------------------------------------------------


def test_load_builds_the_match_exactly_as_written(idsse, tmp_path):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    match = dev.match
    n1, n2 = idsse["n_first"], idsse["n_second"]

    assert match.n_frames == n1 + PAD + n2 and match.fps == 25.0
    assert match.pitch == syn.PITCH
    assert (match.period[:n1] == 1).all() and (match.period[n1:n1 + PAD] == 0).all()
    assert (match.period[n1 + PAD:] == 2).all()
    assert match.players_xy.shape == (match.n_frames, 46, 2)
    assert list(match.slot_team) == ["H"] * 23 + ["A"] * 23

    # ball: centre coordinates become corner coordinates; the missing row is NaN; speed is given
    seen = np.ones(n1, dtype=bool)
    seen[syn.MISSING_BALL_ROW] = False
    assert np.allclose(match.ball_xy[:n1][seen], idsse["ball_first"][seen] + SHIFT)
    assert np.isnan(match.ball_xy[syn.MISSING_BALL_ROW]).all()
    assert np.allclose(match.ball_xy[n1 + PAD:], idsse["ball_second"] + SHIFT)
    assert np.isnan(match.ball_xy[n1:n1 + PAD]).all()
    assert match.ball_speed[0] == pytest.approx(2.0) and match.ball_speed[3] == pytest.approx(3.5)
    assert np.array_equal(np.isnan(match.ball_speed), np.isnan(match.ball_xy[:, 0]))


def test_load_puts_players_in_sorted_slots_and_ignores_referees_and_extra_time(idsse, tmp_path):
    match = devset.load_idsse("J03WTEST", root=tmp_path).match
    n1 = idsse["n_first"]

    # home player i is slot i, away player i is slot 23 + i, both at their fixed spot
    for i in (0, 4, 10):
        assert np.allclose(match.players_xy[0, i], np.array(syn._player_xy("H", i)) + SHIFT)
        assert np.allclose(match.players_xy[0, 23 + i], np.array(syn._player_xy("A", i)) + SHIFT)
    # the substitute (slot 11) is only tracked in the second half, from its middle
    assert np.isnan(match.players_xy[:n1 + PAD, 11]).all()
    second = match.players_xy[n1 + PAD:, 11, 0]
    assert np.isnan(second[:idsse["n_second"] // 2]).all()
    assert not np.isnan(second[idsse["n_second"] // 2:]).any()
    # slots 12..22 and 34..45 stay empty; the referee took no slot; extra time added no frame
    assert np.isnan(match.players_xy[:, 12:23]).all() and np.isnan(match.players_xy[:, 34:]).all()
    assert match.n_frames == n1 + PAD + idsse["n_second"]


def test_truth_comes_from_ball_status_and_possession(idsse, tmp_path):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    n1 = idsse["n_first"]

    assert np.array_equal(dev.truth[:n1], idsse["truth_first"])
    assert dev.truth[syn.MISSING_BALL_ROW] == "X"           # no BallStatus in that row
    assert (dev.truth[n1:n1 + PAD] == "X").all()            # halftime is never scored
    assert np.array_equal(dev.truth[n1 + PAD:], idsse["truth_second"])
    assert set(dev.truth) == {"H", "A", "D", "X"}


def test_corner_origin_files_give_the_same_match(tmp_path):
    syn.write_idsse(tmp_path / "centre", "J03WTEST")
    syn.write_idsse(tmp_path / "corner", "J03WTEST", origin="corner")

    centre = devset.load_idsse("J03WTEST", root=tmp_path / "centre", use_cache=False)
    corner = devset.load_idsse("J03WTEST", root=tmp_path / "corner", use_cache=False)

    assert np.allclose(centre.match.ball_xy, corner.match.ball_xy, equal_nan=True)
    assert np.allclose(centre.match.players_xy, corner.match.players_xy, equal_nan=True)
    assert np.array_equal(centre.truth, corner.truth)


def test_origin_can_be_forced_and_is_checked(idsse, tmp_path):
    forced = devset.load_idsse("J03WTEST", root=tmp_path, origin="corner", use_cache=False)
    auto = devset.load_idsse("J03WTEST", root=tmp_path, use_cache=False)
    assert np.allclose(forced.match.ball_xy[0], auto.match.ball_xy[0] - SHIFT)

    with pytest.raises(ValueError, match="origin must be"):
        devset.load_idsse("J03WTEST", root=tmp_path, origin="middle", use_cache=False)


def test_the_cache_is_not_reused_for_another_origin(idsse, tmp_path):
    auto = devset.load_idsse("J03WTEST", root=tmp_path)
    corner = devset.load_idsse("J03WTEST", root=tmp_path, origin="corner")

    assert not np.allclose(auto.match.ball_xy[0], corner.match.ball_xy[0])
    assert len(list((tmp_path / "cache").glob("J03WTEST_*.npz"))) == 2


def test_pitch_comes_from_the_match_information_or_the_fallback(tmp_path):
    syn.write_idsse(tmp_path, "J03WNOPITCH", pitch_in_info=False)

    with pytest.raises(ValueError, match="no PitchX/PitchY"):
        devset.load_idsse("J03WNOPITCH", root=tmp_path)
    dev = devset.load_idsse("J03WNOPITCH", root=tmp_path, pitch=(105.0, 68.0))
    assert dev.match.pitch == (105.0, 68.0)


def test_a_half_without_a_ball_is_an_error(tmp_path):
    data = syn.write_idsse(tmp_path)
    text = data["positions_path"].read_text().replace('GameSection="secondHalf"', 'GameSection="x"')
    data["positions_path"].write_text(text)

    with pytest.raises(ValueError, match="no ball FrameSet found for period 1|period 2"):
        devset.load_idsse("J03WTEST", root=tmp_path, use_cache=False)


def test_more_than_23_players_in_a_team_is_an_error():
    info = {"players": {f"p{i}": "H" for i in range(24)}}
    periods = {1: {"players": {f"p{i}": None for i in range(24)}}}

    with pytest.raises(ValueError, match="at most 23"):
        devset._slots(periods, info)


def test_cache_gives_the_same_match_without_parsing_again(idsse, tmp_path, monkeypatch):
    first = devset.load_idsse("J03WTEST", root=tmp_path)
    assert list((tmp_path / "cache").glob("J03WTEST_*.npz"))

    def no_parsing(*args, **kwargs):
        raise AssertionError("the cache was not used")

    monkeypatch.setattr(devset, "read_positions", no_parsing)
    second = devset.load_idsse("J03WTEST", root=tmp_path)

    assert np.array_equal(first.match.ball_xy, second.match.ball_xy, equal_nan=True)
    assert np.array_equal(first.match.players_xy, second.match.players_xy, equal_nan=True)
    assert np.array_equal(first.truth, second.truth)
    assert list(first.match.slot_team) == list(second.match.slot_team)
    assert second.match.pitch == first.match.pitch and second.match.n_frames == first.match.n_frames


def test_a_corrupt_dev_cache_is_parsed_again(idsse, tmp_path):
    first = devset.load_idsse("J03WTEST", root=tmp_path)
    (cache,) = (tmp_path / "cache").iterdir()
    cache.write_bytes(b"not a zip file")

    with pytest.warns(RuntimeWarning, match="could not be read"):
        second = devset.load_idsse("J03WTEST", root=tmp_path)

    assert np.array_equal(first.match.ball_xy, second.match.ball_xy, equal_nan=True)
    assert np.array_equal(first.truth, second.truth)
    assert len(list((tmp_path / "cache").iterdir())) == 1


def test_a_changed_file_is_parsed_again(idsse, tmp_path):
    devset.load_idsse("J03WTEST", root=tmp_path)
    path = idsse["positions_path"]
    path.write_text(path.read_text().replace('X="-60.000"', 'X="-61.000"') + "\n")

    again = devset.load_idsse("J03WTEST", root=tmp_path)

    assert len(list((tmp_path / "cache").glob("J03WTEST_*.npz"))) == 2
    dead = np.flatnonzero(again.truth == "D")[0]
    assert again.match.ball_xy[dead, 0] == pytest.approx(-61.0 + SHIFT[0])


# --- where the data folder is ----------------------------------------------------------------


def test_idsse_dir_is_checked(tmp_path, monkeypatch):
    monkeypatch.delenv("IDSSE_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        devset.idsse_dir()

    monkeypatch.setenv("IDSSE_DATA_DIR", str(vendor.REPO_ROOT / "somewhere"))
    with pytest.raises(RuntimeError, match="inside the repo"):
        devset.idsse_dir()

    monkeypatch.setenv("IDSSE_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="does not exist"):
        devset.idsse_dir()

    monkeypatch.setenv("IDSSE_DATA_DIR", str(tmp_path))
    assert devset.idsse_dir() == tmp_path.resolve()


def test_the_halftime_pad_is_longer_than_any_smoothing_window():
    """The real halftime hole is 20,156 frames, so a vote near a half edge never sees the other
    half. The pad must be long enough to give the same."""
    assert devset.HALFTIME_PAD > max(window for _, window in possession.VARIANTS) * 2


# --- degrading the ball ----------------------------------------------------------------------


def test_degrade_reaches_the_missing_share_with_gaps_of_the_asked_lengths():
    match = long_match()
    new, report = devset.degrade_ball(match, DEGRADE, seed=1)
    lengths = gap_lengths(new.ball_xy, new.period)
    graded = int((match.period > 0).sum())

    share = sum(lengths) / graded
    assert 0.298 <= share <= 0.33                     # the target, plus at most one gap per bin
    assert report["missing_share"] == pytest.approx(share)
    assert min(lengths) >= 1 and max(lengths) <= 200  # 8 s at 25 fps
    assert len(lengths) == report["gaps_placed"]      # no two gaps merged into one
    in_bin = {"short": 0, "mid": 0, "long": 0}
    for length in lengths:
        in_bin["short" if length <= 12 else "mid" if length <= 50 else "long"] += length
    for key, want in (("short", 0.1), ("mid", 0.3), ("long", 0.6)):
        assert in_bin[key] / sum(lengths) == pytest.approx(want, abs=0.05)


def test_degrade_changes_only_the_ball_and_only_inside_the_halves():
    match = long_match()
    before = match.ball_xy.copy()
    new, _ = devset.degrade_ball(match, DEGRADE, seed=1)

    assert np.array_equal(match.ball_xy, before, equal_nan=True)       # the input is untouched
    assert np.array_equal(new.period, match.period)
    assert np.array_equal(new.players_xy, match.players_xy, equal_nan=True)
    lost = np.isnan(new.ball_xy[:, 0]) & ~np.isnan(match.ball_xy[:, 0])
    assert (match.period[lost] > 0).all()                              # nothing in the pad
    kept = ~np.isnan(new.ball_xy[:, 0])
    assert np.array_equal(new.ball_xy[kept], match.ball_xy[kept])      # the rest is as tracked
    assert np.array_equal(np.isnan(new.ball_speed), np.isnan(new.ball_xy[:, 0]))


def test_degraded_gaps_never_cross_the_halftime_pad_or_touch_a_half_edge():
    match = long_match()
    new, report = devset.degrade_ball(match, {**DEGRADE, "missing_share": 0.6}, seed=3)
    first = np.flatnonzero(new.period == 1)
    second = np.flatnonzero(new.period == 2)

    for frames in (first, second):
        assert not np.isnan(new.ball_xy[frames[0], 0])     # one seen frame at each edge
        assert not np.isnan(new.ball_xy[frames[-1], 0])
    # in a crowded half, two gaps still never run into each other and become one longer gap
    assert len(gap_lengths(new.ball_xy, new.period)) == report["gaps_placed"]


def test_degrade_is_repeatable_and_the_seed_matters():
    match = long_match()
    a, _ = devset.degrade_ball(match, DEGRADE, seed=7)
    b, _ = devset.degrade_ball(match, DEGRADE, seed=7)
    c, _ = devset.degrade_ball(match, DEGRADE, seed=8)

    assert np.array_equal(a.ball_xy, b.ball_xy, equal_nan=True)
    assert not np.array_equal(a.ball_xy, c.ball_xy, equal_nan=True)


def test_the_longest_gaps_are_placed_first_so_a_crowded_half_still_fits_them():
    section = {"missing_share": 0.7, "gaps": DEGRADE["gaps"]}
    _, report = devset.degrade_ball(long_match(), section, seed=0)

    assert report["gaps_unplaced"] == 0
    assert report["missing_share"] >= 0.69


def test_speed_is_dropped_wherever_the_position_is_missing():
    match = long_match()
    match.ball_xy[100:110] = np.nan            # a position gap that still has a speed value
    assert not np.isnan(match.ball_speed[100:110]).any()
    few = {"missing_share": 0.001, "gaps": [{"min_s": 0.04, "max_s": 0.2, "share": 1.0}]}

    new, _ = devset.degrade_ball(match, few, seed=0)

    assert np.isnan(new.ball_xy[100:110]).all()
    assert np.isnan(new.ball_speed[100:110]).all()
    assert np.array_equal(np.isnan(new.ball_speed), np.isnan(new.ball_xy[:, 0]))


class Scripted:
    """Stands in for the random generator and hands out the numbers a test picked."""

    def __init__(self, integers, point=(1.0, 2.0)):
        self.queue = list(integers)
        self.point = np.array(point)

    def choice(self, n, p=None):
        return 0

    def integers(self, low, high):
        return self.queue.pop(0)

    def uniform(self, low, high):
        return self.point


def test_a_new_gap_keeps_a_seen_frame_between_itself_and_the_gap_before():
    ranges = [(0, 1000)]
    blocked = np.zeros(1000, dtype=bool)

    assert devset._place(Scripted([100]), ranges, blocked, 10) == 100       # frames 100-109
    assert devset._place(Scripted([110, 111, 90]), ranges, blocked, 10, tries=3) is None
    assert devset._place(Scripted([112]), ranges, blocked, 10) == 112       # 110 and 111 stay seen
    assert devset._place(Scripted([85]), ranges, blocked, 10) == 85         # ends at 94: fine


def test_a_false_stretch_never_fills_a_hole_the_tracker_already_had():
    ball = np.zeros((100, 2))
    ball[10:20] = np.nan
    settings = {"fault_share": 0.02, "fault_low": 1, "fault_high": 5}

    # one stretch of 5 frames from frame 8: frames 8 and 9 are seen, 10-12 are the hole
    stretches = devset._inject_false(Scripted([8, 5]), ball, [(0, 100)], 100, (105.0, 68.0),
                                     settings)

    assert stretches == 1
    assert np.isnan(ball[10:20]).all()
    assert (ball[8:10] == [1.0, 2.0]).all() and (ball[:8] == 0).all()


def test_degrade_reports_when_a_bin_runs_out_of_room():
    section = {"missing_share": 0.95, "gaps": [{"min_s": 4.0, "max_s": 8.0, "share": 1.0}]}
    new, report = devset.degrade_ball(long_match(), section, seed=0)

    assert report["gaps_unplaced"] >= 1
    assert report["missing_share"] < 0.95
    assert report["gaps_placed"] == len(gap_lengths(new.ball_xy, new.period))


def test_degrade_keeps_a_match_without_graded_frames_as_it_is():
    match = replace(long_match(), period=np.zeros(20000, dtype=np.int8))
    new, report = devset.degrade_ball(match, DEGRADE, seed=0)

    assert new is match and report["missing_share"] is None


def test_false_detections_replace_seen_positions_with_points_on_the_pitch():
    match = long_match()
    match.ball_xy[3000:3400] = np.nan          # a hole the tracker already had: it stays a hole
    section = {**DEGRADE, "false_detections": {"share": 0.02, "min_s": 0.04, "max_s": 0.2}}
    new, report = devset.degrade_ball(match, {**section, "missing_share": 0.1}, seed=2)
    plain, _ = devset.degrade_ball(match, {**DEGRADE, "missing_share": 0.1}, seed=2)

    assert report["false_stretches"] > 0
    assert np.isnan(new.ball_xy[3000:3400]).all()
    changed = ~np.isnan(new.ball_xy[:, 0]) & (new.ball_xy != match.ball_xy).any(axis=1)
    graded = int((match.period > 0).sum())
    assert changed.sum() >= 0.015 * graded * 0.5          # the asked share, minus what gaps hide
    assert (new.ball_xy[changed] >= 0).all() and (new.ball_xy[changed, 0] <= 105.0).all()
    assert (new.ball_xy[changed, 1] <= 68.0).all()
    assert (match.period[changed] > 0).all()
    assert not np.array_equal(new.ball_xy, plain.ball_xy, equal_nan=True)


@pytest.mark.parametrize(
    "section, message",
    [
        ({"gaps": DEGRADE["gaps"]}, "needs 'missing_share'"),
        ({"missing_share": 0.3}, "needs 'gaps'"),
        ({**DEGRADE, "missing_share": 0.0}, r"in \(0, 1\)"),
        ({**DEGRADE, "missing_share": 1.0}, r"in \(0, 1\)"),
        ({**DEGRADE, "missing_share": True}, r"in \(0, 1\)"),
        ({"missing_share": 0.3, "gaps": [{"min_s": 0, "max_s": 1, "share": 0.5}]}, "add up to 1"),
        ({"missing_share": 0.3, "gaps": [{"min_s": 1, "max_s": 1, "share": 1.0}]},
         "no whole frame"),
        ({"missing_share": 0.3, "gaps": [{"min_s": 0, "max_s": 1, "share": 1.5},
                                         {"min_s": 1, "max_s": 2, "share": -0.5}]}, "negative"),
        ({**DEGRADE, "false_detections": {"share": 1.0}}, "false_detections.share"),
    ],
)
def test_bad_degrade_settings_are_refused(section, message):
    with pytest.raises(ValueError, match=message):
        devset.degrade_ball(long_match(), section, seed=0)


def test_every_match_gets_its_own_fixed_seed():
    assert devset.match_seed(0, "J03WPY") == devset.match_seed(0, "J03WPY")
    assert devset.match_seed(0, "J03WPY") != devset.match_seed(0, "J03WMX")
    assert devset.match_seed(0, "J03WPY") != devset.match_seed(1, "J03WPY")


# --- running the pipeline and scoring --------------------------------------------------------


def test_a_complete_ball_gives_a_perfect_score(idsse, tmp_path):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)

    labels = devset.predict_labels(dev.match, pipeline_cfg())
    score = devset.score_labels(dev.truth, labels)

    graded = dev.truth != "X"
    assert np.array_equal(labels[graded], dev.truth[graded])
    assert score["spellF1"] == pytest.approx(1.0) and score["macroF1"] == pytest.approx(1.0)
    assert score["truth_possessions"] == score["our_possessions"] > 5
    assert score["spell_precision"] == pytest.approx(1.0)
    assert score["spell_recall"] == pytest.approx(1.0)
    assert score["latency_mean"] == pytest.approx(0.0)        # every switch exactly on time


def test_score_labels_returns_plain_numbers_and_counts_possessions():
    truth = np.array(list("HHHHDDAAAA"))
    labels = np.array(list("HHHHDDAAHH"))
    score = devset.score_labels(truth, labels)

    headline = {"spellF1", "macroF1", "truth_possessions", "our_possessions"}
    breakdown = {"spell_precision", "spell_recall", "alive_dead_f1", "team_acc",
                 "trans_f1_tight", "trans_f1_loose", "latency_mean"}
    assert headline | breakdown <= set(score)
    assert "spell_ci_low" not in score                        # no bootstrap unless asked for
    assert score["truth_possessions"] == 2 and score["our_possessions"] == 3
    assert score["count_error"] == 1 and score["spell_spurious"] >= 1
    assert all(type(v) in (int, float) for v in score.values())


def test_score_labels_adds_the_bootstrap_interval_when_asked():
    truth = np.array(list("HHHHDDAAAA"))
    labels = np.array(list("HHHHDDAAHH"))
    score = devset.score_labels(truth, labels, bootstrap_n=50)

    assert 0.0 <= score["spell_ci_low"] <= score["spell_ci_high"] <= 1.0


def test_gaps_hurt_the_baseline_and_interpolation_wins_some_back(idsse, tmp_path):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    dev_cfg = {"seed": 0, "matches": ["J03WTEST"], "degrade": SHORT_GAPS}
    configs = {
        "none": pipeline_cfg(),
        "interpolate": pipeline_cfg({"method": "interpolate", "max_gap": 0.5}),
    }

    clean = devset.run_devset(configs, dev_cfg, [dev], degrade=False)
    cut = devset.run_devset(configs, dev_cfg, [dev], degrade=True)

    perfect = clean["scores"]["none"]["J03WTEST"]["spellF1"]
    hurt = cut["scores"]["none"]["J03WTEST"]["spellF1"]
    repaired = cut["scores"]["interpolate"]["J03WTEST"]["spellF1"]
    assert perfect == pytest.approx(1.0)
    assert hurt < perfect - devset.pm.SPELL_NOISE_BAND       # the gaps cost more than noise
    assert repaired > hurt + devset.pm.SPELL_NOISE_BAND      # and filling them wins it back
    assert cut["matches"]["J03WTEST"]["missing_share"] > 0.2
    assert clean["matches"]["J03WTEST"]["missing_share"] < 0.01


def test_the_dead_ball_and_possession_stages_get_the_cleaned_ball(idsse, tmp_path, monkeypatch):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    cut, _ = devset.degrade_ball(dev.match, SHORT_GAPS, seed=0)
    raw_missing = int(np.isnan(cut.ball_xy[:, 0]).sum())
    seen = {}
    for stage in (devset.deadball, devset.possession):
        name = "detect" if stage is devset.deadball else "predict"
        real = getattr(stage, name)

        def spy(match, ball, *rest, _real=real, _name=name):
            seen[_name] = int(np.isnan(match.ball_xy[:, 0]).sum())
            return _real(match, ball, *rest)

        monkeypatch.setattr(stage, name, spy)

    devset.predict_labels(cut, pipeline_cfg({"method": "interpolate", "max_gap": 0.5}))

    assert seen["detect"] == seen["predict"] < raw_missing    # both saw the filled track


def test_every_config_sees_the_same_degraded_ball(idsse, tmp_path, monkeypatch):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    seen = []
    real = devset.predict_labels
    monkeypatch.setattr(
        devset, "predict_labels",
        lambda match, cfg, *rest: seen.append(match.ball_xy.copy()) or real(match, cfg, *rest),
    )
    dev_cfg = {"seed": 0, "matches": ["J03WTEST"], "degrade": SHORT_GAPS}

    devset.run_devset({"a": pipeline_cfg(), "b": pipeline_cfg(window=5)}, dev_cfg, [dev])

    assert len(seen) == 2 and np.array_equal(seen[0], seen[1], equal_nan=True)


def test_configs_that_differ_only_in_possession_share_the_earlier_stages(idsse, tmp_path,
                                                                         monkeypatch):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    calls = []
    real = devset.prepare_stages
    monkeypatch.setattr(devset, "prepare_stages",
                        lambda match, cfg: calls.append(1) or real(match, cfg))
    dev_cfg = {"seed": 0, "matches": ["J03WTEST"], "degrade": SHORT_GAPS}
    configs = {
        "a": pipeline_cfg(),
        "b": pipeline_cfg(window=5),                                     # possession only
        "c": pipeline_cfg({"method": "interpolate", "max_gap": 0.5}),    # a different ball
    }

    devset.run_devset(configs, dev_cfg, [dev])

    assert len(calls) == 2                       # a and b share; c needs its own cleaning


def _same(a, b):
    """Equal arrays (NaN = NaN), or equal plain values."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a == b


def test_ball_and_dead_ball_do_not_depend_on_the_possession_settings(idsse, tmp_path):
    """run_devset reuses the cleaned ball and dead score between configs that differ only in
    possession: (_stage_key). That is only right while those stages never read cfg["possession"].
    If one starts to, this test fails instead of the dev set quietly reusing the wrong stages."""
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    cut, _ = devset.degrade_ball(dev.match, SHORT_GAPS, seed=0)
    ball = {"method": "interpolate", "max_gap": 0.5}
    a = pipeline_cfg(ball)
    b = pipeline_cfg(ball, window=50)
    b["possession"]["dead_threshold"] = 0.9
    assert devset._stage_key(a) == devset._stage_key(b)

    cleaned_a, track_a, dead_a = devset.prepare_stages(cut, a)
    cleaned_b, track_b, dead_b = devset.prepare_stages(cut, b)

    assert np.array_equal(cleaned_a.ball_xy, cleaned_b.ball_xy, equal_nan=True)
    assert np.array_equal(cleaned_a.ball_speed, cleaned_b.ball_speed, equal_nan=True)
    assert np.array_equal(dead_a, dead_b, equal_nan=True)
    assert vars(track_a).keys() == vars(track_b).keys()
    assert all(_same(vars(track_a)[key], vars(track_b)[key]) for key in vars(track_a))


def test_the_same_run_gives_the_same_numbers(idsse, tmp_path):
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    dev_cfg = {"seed": 4, "matches": ["J03WTEST"], "degrade": SHORT_GAPS}

    def go():
        return devset.run_devset({"none": pipeline_cfg()}, dev_cfg, [dev])

    assert go() == go()


# --- the report ------------------------------------------------------------------------------


def two_config_result(tmp_path):
    syn.write_idsse(tmp_path, "J03WTEST")
    dev = devset.load_idsse("J03WTEST", root=tmp_path)
    dev_cfg = {"seed": 0, "matches": ["J03WTEST"], "degrade": SHORT_GAPS}
    configs = {
        "baseline": pipeline_cfg(),
        "interpolate": pipeline_cfg({"method": "interpolate", "max_gap": 0.5}),
    }
    return devset.run_devset(configs, dev_cfg, [dev])


def fake_score(spell):
    """A complete score for one match, with spellF1 set to `spell`."""
    score = devset.score_labels(np.array(list("HHHHDDAAAA")), np.array(list("HHHHDDAAAA")))
    return {**score, "spellF1": spell}


def test_report_has_the_mean_every_match_and_the_paired_difference(tmp_path):
    text = devset.format_report(two_config_result(tmp_path))

    assert text.startswith("Dev set (IDSSE), 1 matches, aggregates only")
    assert "ball: degraded to look like ours" in text
    assert "J03WTEST: ball missing" in text and "gaps" in text
    assert "config baseline" in text and "config interpolate" in text
    for part in ("spellF1", "possessions: truth", "precision", "recall", "alive/dead F1",
                 "team acc", "switches: F1", "latency"):
        assert part in text
    assert "mean: spellF1" in text and "possessions: truth" in text
    assert text.index("mean: spellF1") < text.index("  J03WTEST\n")   # mean first, then matches
    assert "paired spellF1 difference against baseline" in text
    assert "better on 1 of 1 matches (better)" in text      # interpolation wins on this match
    assert "    J03WTEST: spellF1 +" in text                 # and the same per match


def test_report_says_when_the_ball_was_not_degraded(tmp_path):
    result = two_config_result(tmp_path)

    assert "NOT degraded" in devset.format_report(result, degrade=False)


def test_a_difference_under_the_noise_band_is_called_noise():
    scores = {"a": {"m": fake_score(0.50)}, "b": {"m": fake_score(0.52)}}
    result = {"matches": {"m": {"missing_share": 0.4}}, "scores": scores}

    text = devset.format_report(result)

    assert "b: +0.020, better on 1 of 1 matches (noise)" in text   # overall
    assert "m: spellF1 +0.020 (noise)" in text                      # per match
    assert devset.paired_difference(scores["b"], scores["a"]) == (pytest.approx(0.02), 1, 1)


def test_the_mean_is_over_matches_and_skips_missing_values():
    one, two = fake_score(0.4), fake_score(0.6)
    two["latency_mean"] = None                     # no matched switch in that match

    total = devset.summarise({"m1": one, "m2": two})

    assert total["spellF1"] == pytest.approx(0.5)
    assert total["spellF1_min"] == pytest.approx(0.4) and total["spellF1_max"] == pytest.approx(0.6)
    assert total["latency_mean"] == pytest.approx(one["latency_mean"])   # only the one that exists


def test_the_overall_line_shows_the_mean_change_of_the_breakdown():
    base = {"m1": fake_score(0.4), "m2": fake_score(0.4)}
    better = {"m1": {**fake_score(0.5), "spell_precision": base["m1"]["spell_precision"] + 0.1},
              "m2": {**fake_score(0.5), "spell_precision": base["m2"]["spell_precision"] + 0.3}}
    result = {"matches": {"m1": {"missing_share": 0.3}, "m2": {"missing_share": 0.3}},
              "scores": {"base": base, "new": better}}

    text = devset.format_report(result)

    assert "new: +0.100, better on 2 of 2 matches (better); mean spell_precision +0.200" in text


def test_a_missing_latency_is_shown_as_dashes():
    score = {**fake_score(0.5), "latency_mean": None, "latency_median": None,
             "latency_abs": None}
    result = {"matches": {"m": {"missing_share": 0.4}}, "scores": {"a": {"m": score}}}

    assert "latency mean -- / median -- frames (mean |latency| --)" in devset.format_report(result)


def test_report_is_short_and_shows_no_positions_or_frame_numbers(tmp_path):
    result = two_config_result(tmp_path)
    text = devset.format_report(result)
    other = syn.write_idsse(tmp_path / "x", "J03WOTHER")
    structure = devset.inspect_structure(other["positions_path"])

    assert len(text.splitlines()) <= 20
    for forbidden in ("-40.500", "-39.500", "40.500", "10000", "10001", "60000"):
        assert forbidden not in text and forbidden not in structure
    summary = str(devset.summary(devset.load_idsse("J03WTEST", root=tmp_path)))
    assert "[" not in summary                     # no arrays, only counts


def test_inspect_structure_lists_names_and_counts_only(idsse):
    text = devset.inspect_structure(idsse["positions_path"])

    for name in ("FrameSet", "Frame", "GameSection", "BallStatus", "BallPossession", "PersonId"):
        assert name in text
    assert "attributes: BallPossession, BallStatus, N, S, X, Y, Z" in text
    info = devset.inspect_structure(idsse["info_path"])
    assert "PitchX" in info and "Role" in info and "TeamId" in info


def test_inspect_structure_stops_after_the_element_limit(idsse):
    text = devset.inspect_structure(idsse["positions_path"], max_elements=40)

    assert "first 40 elements at most" in text


# --- the command line ------------------------------------------------------------------------


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    """Data folder with one synthetic match, a dev-set config for it, and the env variable."""
    data = tmp_path / "idsse"
    syn.write_idsse(data, "J03WTEST")
    dev_cfg = tmp_path / "devset.yaml"
    dev_cfg.write_text(
        "seed: 0\nmatches: [J03WTEST]\ndegrade:\n  missing_share: 0.25\n  gaps:\n"
        "    - {min_s: 0.2, max_s: 0.5, share: 1.0}\n"
    )
    monkeypatch.setenv("IDSSE_DATA_DIR", str(data))
    return dev_cfg


def test_main_prints_the_report_for_two_configs(cli_setup, capsys):
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"
    interpolate = vendor.REPO_ROOT / "configs" / "exp" / "ball-interpolate.yaml"

    assert cli.main(["--devset", str(cli_setup), "--config", str(base), str(interpolate)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Dev set (IDSSE), 1 matches, aggregates only")
    assert "config baseline" in out and "config ball-interpolate" in out
    assert "mean: spellF1" in out
    assert "paired spellF1 difference against baseline" in out


def test_main_bootstrap_adds_the_interval(cli_setup, capsys):
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"

    assert cli.main(["--devset", str(cli_setup), "--bootstrap", "20",
                     "--config", str(base)]) == 0

    assert "spellF1 bootstrap 5-95%" in capsys.readouterr().out


def test_main_refuses_a_negative_bootstrap(cli_setup):
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"

    with pytest.raises(SystemExit):
        cli.main(["--devset", str(cli_setup), "--bootstrap", "-5", "--config", str(base)])


def test_main_clean_ball_and_match_filter(cli_setup, capsys):
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"

    assert cli.main(["--devset", str(cli_setup), "--clean-ball", "--matches", "J03WTEST",
                     "--config", str(base)]) == 0

    assert "NOT degraded" in capsys.readouterr().out


def test_main_summary_prints_counts_only(cli_setup, capsys):
    assert cli.main(["--devset", str(cli_setup), "--summary"]) == 0

    out = capsys.readouterr().out
    assert "'match': 'J03WTEST'" in out and "'truth'" in out


def test_main_inspect_needs_no_data_folder(idsse, monkeypatch, capsys):
    monkeypatch.delenv("IDSSE_DATA_DIR", raising=False)

    assert cli.main(["--inspect", str(idsse["positions_path"])]) == 0

    assert "FrameSet" in capsys.readouterr().out


def test_main_needs_a_config_and_the_data_folder_variable(cli_setup, monkeypatch):
    with pytest.raises(SystemExit):
        cli.main(["--devset", str(cli_setup)])           # nothing to score

    monkeypatch.delenv("IDSSE_DATA_DIR")
    with pytest.raises(RuntimeError, match="not set"):
        cli.main(["--devset", str(cli_setup), "--summary"])


def test_the_shipped_devset_config_is_valid():
    cfg = cli.load_dev_config(vendor.REPO_ROOT / "configs" / "devset.yaml")

    assert len(cfg["matches"]) == 7 and cfg["seed"] == 0
    settings = devset._degrade_settings(cfg["degrade"], 25.0)    # also checks the bins add up
    assert 0 < settings["missing_share"] < 1