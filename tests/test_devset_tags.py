"""Simulated restart tags on the dev set (devset.simulate_tags, review finding 2).

Made-up matches only: no file is read.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synthetic_idsse as syn
import yaml

from falafelona import deadball, devset, vendor
from falafelona.io import Match, select_tags

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "devset_eval.py"
spec = importlib.util.spec_from_file_location("devset_eval", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

FPS = 25.0
PITCH = (105.0, 68.0)
TAGS = yaml.safe_load((vendor.REPO_ROOT / "configs" / "devset.yaml").read_text())["tags"]
SETTINGS = devset._tag_settings(TAGS, FPS)

# Where the ball is when play restarts, one spot per restart type (metres from the corner).
SPOTS = {
    "kick off (after goal)": (52.5, 34.0),
    "corner kick": (104.5, 0.5),
    "throw-in": (40.0, 68.5),        # just past the touchline
    "goal kick": (5.0, 30.0),
    "free kick": (70.0, 40.0),
}


def scripted_match(n_half=60_000, pad=2_000, play_s=20, dead_s=6, spots=None):
    """Two halves of play and dead-ball spells in turn, the ball at a known spot at each restart.

    Returns (match, truth, restarts): restarts lists (first live frame, type) inside the halves.
    Each half STARTS with a dead spell (its own kickoff), which must not become a restart tag.
    """
    spots = list((spots or SPOTS).items())
    n = 2 * n_half + pad
    period = np.zeros(n, dtype=np.int8)
    period[:n_half] = 1
    period[n_half + pad:] = 2
    ball = np.full((n, 2), 30.0)                       # open play: a spot that is a free kick
    ball[period == 0] = np.nan
    truth = np.full(n, "X", dtype="<U1")
    play, dead = int(play_s * FPS), int(dead_s * FPS)
    restarts, k = [], 0
    for low, high in ((0, n_half), (n_half + pad, n)):
        i = low
        while i < high:
            truth[i:min(i + dead, high)] = "D"
            live = i + dead
            if live >= high:
                break
            kind, spot = spots[k % len(spots)]
            ball[live:live + 10] = spot
            truth[live:min(live + play, high)] = "H" if k % 2 else "A"
            if i != low:                               # the half's own kickoff is not a restart
                restarts.append((live, kind))
            k += 1
            i = live + play
    match = Match(
        n_frames=n, fps=FPS, period=period, ball_xy=ball,
        ball_speed=np.where(np.isnan(ball[:, 0]), np.nan, 5.0),
        players_xy=np.full((n, 2, 2), np.nan), players_speed=np.full((n, 2), np.nan),
        slot_team=np.array(["H", "A"]), pitch=PITCH,
    )
    return match, truth, restarts


def wanted(kind, match):
    graded = int((match.period > 0).sum())
    return round(TAGS["our_counts"][kind] * graded / TAGS["our_graded_frames"])


@pytest.mark.parametrize("kind", list(SPOTS))
def test_the_restart_type_comes_from_where_the_ball_is(kind):
    assert devset.restart_type(np.array(SPOTS[kind]), PITCH, SETTINGS) == kind


def test_an_unseen_ball_gives_no_type_and_a_goal_line_spot_outside_the_box_is_a_free_kick():
    assert devset.restart_type(np.array([np.nan, np.nan]), PITCH, SETTINGS) is None
    assert devset.restart_type(None, PITCH, SETTINGS) is None
    assert devset.restart_type(np.array([2.0, 20.0]), PITCH, SETTINGS) == "free kick"


def test_tags_sit_at_restarts_with_the_right_type_and_at_our_rate():
    match, truth, restarts = scripted_match()
    tags, report = devset.simulate_tags(match, truth, TAGS, seed=0)
    by_frame = dict(restarts)
    low, high = SETTINGS["offset"]

    assert report["tags"] == len(tags) > 0
    assert [t.start for t in tags] == sorted(t.start for t in tags)
    for tag in tags:
        live = [f for f in by_frame if low <= tag.start - f <= high]
        assert len(live) == 1                          # one restart, within the offset
        assert by_frame[live[0]] == tag.type           # of the right type
        assert tag.code == "SET PIECES" and tag.end == tag.start + 1
        assert tag.half == match.period[live[0]]
        assert tag.team is None if tag.type == "throw-in" else tag.team == truth[live[0]]
    for kind in SPOTS:                                 # plenty of candidates: exactly our rate
        assert len(select_tags(tags, type=kind)) == wanted(kind, match)
    assert report["tags_short"] == 0


def test_a_half_starts_with_its_own_kickoff_which_is_never_a_restart_tag():
    match, truth, _ = scripted_match(spots={"kick off (after goal)": SPOTS["kick off (after goal)"]})
    tags, _ = devset.simulate_tags(match, truth, TAGS, seed=0)
    first_live = {int(np.flatnonzero((match.period == h) & (truth != "D"))[0]) for h in (1, 2)}

    assert tags
    assert not any(abs(tag.start - f) <= SETTINGS["offset"][1] for tag in tags for f in first_live)


def test_a_match_with_too_few_stoppages_of_a_type_uses_all_and_reports_the_shortfall():
    match, truth, restarts = scripted_match(spots={"corner kick": SPOTS["corner kick"],
                                                   "free kick": SPOTS["free kick"]},
                                            n_half=20_000, dead_s=6, play_s=200)
    tags, report = devset.simulate_tags(match, truth, TAGS, seed=0)
    corners = sum(kind == "corner kick" for _, kind in restarts)

    assert len(select_tags(tags, type="corner kick")) == min(corners, wanted("corner kick", match))
    expected_short = sum(max(0, wanted(k, match) - sum(r == k for _, r in restarts))
                         for k in devset.TAG_TYPES)
    assert report["tags_short"] == expected_short > 0


def test_short_dead_spells_and_restarts_with_no_ball_give_no_tag():
    match, truth, _ = scripted_match(dead_s=0.6)       # every spell under min_dead_s (1 s)
    assert devset.simulate_tags(match, truth, TAGS, seed=0)[0] == []

    match, truth, _ = scripted_match()
    no_ball = replace(match, ball_xy=np.full_like(match.ball_xy, np.nan))
    assert devset.simulate_tags(no_ball, truth, TAGS, seed=0)[0] == []


def test_tags_are_repeatable_and_the_seed_matters():
    match, truth, _ = scripted_match()
    first, _ = devset.simulate_tags(match, truth, TAGS, seed=3)

    assert devset.simulate_tags(match, truth, TAGS, seed=3)[0] == first
    assert devset.simulate_tags(match, truth, TAGS, seed=4)[0] != first


def test_simulated_tags_switch_on_the_tag_clues():
    match, truth, _ = scripted_match()
    tags, _ = devset.simulate_tags(match, truth, TAGS, seed=0)
    # Windows shorter than the 6 s stoppages: the shipped ones (8-45 s) would reach into play.
    section = {"method": "before_restart", "restart_seconds": {k: 4 for k in devset.TAG_TYPES}}

    without = deadball.DeadBall(match, None, {"deadball": section}).score()
    with_tags = deadball.DeadBall(replace(match, tags=tags), None, {"deadball": section}).score()

    assert without.sum() == 0
    assert with_tags.sum() > 0
    assert (truth[with_tags > 0] == "D").mean() > 0.7      # mostly dead ball (offset +-1 s)


@pytest.mark.parametrize(("change", "message"), [
    ({"our_counts": {"throw-in": 26}}, "our_counts"),
    ({"our_counts": {**TAGS["our_counts"], "penalty": 1}}, "our_counts"),
    ({"our_counts": {**TAGS["our_counts"], "throw-in": -1}}, "throw-in"),
    ({"our_graded_frames": 0}, "our_graded_frames"),
    ({"offset_s": [1.0, -1.0]}, "offset_s"),
    ({"offset_s": 1.0}, "offset_s"),
    ({"touchline_m": "2"}, "touchline_m"),
    ({"min_dead_s": True}, "min_dead_s"),
])
def test_bad_tag_settings_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        devset._tag_settings({**TAGS, **change}, FPS)


def test_no_tags_section_is_refused():
    with pytest.raises(ValueError, match="tags"):
        devset._tag_settings(None, FPS)


# --- in the dev-set run ----------------------------------------------------------------------


def dev_cfg():
    return {"seed": 0, "matches": ["SCRIPTED"], "tags": TAGS,
            "degrade": {"missing_share": 0.2, "gaps": [{"min_s": 2.0, "max_s": 8.0, "share": 1.0}]}}


PIPE = {
    "seed": 0,
    "ball": {"method": "none"},
    "deadball": {"method": "before_restart", "restart_seconds": {k: 5 for k in devset.TAG_TYPES}},
    "possession": {"method": "nearest_majority", "window": 0, "dead_threshold": 0.5},
    "output": {"x_policy": "none"},
}


def test_tags_leave_the_degraded_ball_as_it_was_and_reach_the_stages(monkeypatch):
    match, truth, _ = scripted_match()
    dev = devset.DevMatch("SCRIPTED", match, truth)
    seen = []
    real = devset.prepare_stages

    def spy(m, cfg):
        out = real(m, cfg)
        seen.append((m, out[2]))
        return out

    monkeypatch.setattr(devset, "prepare_stages", spy)
    off = devset.run_devset({"p": PIPE}, dev_cfg(), [dev])
    on = devset.run_devset({"p": PIPE}, dev_cfg(), [dev], tags=True)

    (m_off, dead_off), (m_on, dead_on) = seen
    assert m_off.tags == [] and len(m_on.tags) == on["matches"]["SCRIPTED"]["tags"] > 0
    np.testing.assert_array_equal(m_off.ball_xy, m_on.ball_xy)          # same cut ball
    assert "tags" not in off["matches"]["SCRIPTED"]
    assert dead_off.sum() == 0 < dead_on.sum()                          # the tag clue now fires
    report = devset.format_report(on)
    assert "restart tags: SIMULATED" in report and "restart tags" in report.splitlines()[3]
    assert "SIMULATED" not in devset.format_report(off)


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    data = tmp_path / "idsse"
    syn.write_idsse(data, "J03WTEST")
    cfg = yaml.safe_load((vendor.REPO_ROOT / "configs" / "devset.yaml").read_text())
    cfg["matches"] = ["J03WTEST"]
    path = tmp_path / "devset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("IDSSE_DATA_DIR", str(data))
    return path


def test_main_tags_flag_adds_simulated_tags(cli_setup, capsys):
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"

    assert cli.main(["--devset", str(cli_setup), "--config", str(base)]) == 0
    assert "SIMULATED" not in capsys.readouterr().out
    assert cli.main(["--devset", str(cli_setup), "--tags", "--config", str(base)]) == 0
    assert "restart tags: SIMULATED" in capsys.readouterr().out


def test_main_tags_flag_needs_a_tags_section(cli_setup, tmp_path):
    cfg = yaml.safe_load(cli_setup.read_text())
    del cfg["tags"]
    cli_setup.write_text(yaml.safe_dump(cfg))
    base = vendor.REPO_ROOT / "configs" / "baseline.yaml"

    with pytest.raises(SystemExit):
        cli.main(["--devset", str(cli_setup), "--tags", "--config", str(base)])


def test_the_shipped_tag_config_and_its_experiment_pass_their_checks():
    from falafelona.config import load_config

    assert sum(TAGS["our_counts"].values()) == 58                   # dead-ball EDA, #28
    cfg = load_config(vendor.REPO_ROOT / "configs" / "exp" / "poss-hmm-margin-dead-gap-tags.yaml")
    deadball.check_config(cfg["deadball"], FPS)
    assert cfg["deadball"]["gap"]["weights"]["restart_tag_after"] > 0


def test_tags_come_from_the_undegraded_ball_with_their_own_seed(monkeypatch):
    """The cut ball would hide where restarts are taken: tags are made from the full ball."""
    match, truth, _ = scripted_match()
    dev = devset.DevMatch("SCRIPTED", match, truth)
    seen = []
    real = devset.prepare_stages
    monkeypatch.setattr(devset, "prepare_stages", lambda m, cfg: seen.append(m) or real(m, cfg))
    cfg = {**dev_cfg(), "degrade": {"missing_share": 0.6,
                                    "gaps": [{"min_s": 5.0, "max_s": 30.0, "share": 1.0}]}}
    devset.run_devset({"p": PIPE}, cfg, [dev], tags=True)

    expected, _ = devset.simulate_tags(match, truth, TAGS,
                                       devset.match_seed(0, "SCRIPTED/tags"))
    assert seen[0].tags == expected
