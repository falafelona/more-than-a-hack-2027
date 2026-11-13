"""ws:eval: a dev set from OTHER matches, so we can tune and compare without our own match.

Task N5 (team-board issue #23). Decisions D11 and D15: all thresholds are tuned on other matches,
starting with IDSSE (Bassek et al., Scientific Data 12, 195, 2025), which the organisers approved
in official issue #16. Cite it in the technical doc.

WHAT IT DOES
1. Reads one IDSSE match (DFL XML: match information + positions) into the same `Match` the
   pipeline uses: metres, 25 fps, 46 slots (23 per team), pitch from the match information.
2. Builds per-frame truth H / A / D from the tracking file itself: `BallStatus` (alive or dead)
   and `BallPossession` (home or away). A frame where the file says nothing is X (not scored).
3. Degrades the ball track to look like ours: gaps are drawn from the gap-length distribution of
   our real match (configs/devset.yaml, numbers from the ball EDA, issue #13), with a fixed seed,
   and optionally some false ball positions. IDSSE's own ball is almost complete, ours is not.
4. Runs the pipeline stages (ball -> dead ball -> possession -> X policy) with any config and
   scores the labels with the OFFICIAL metric (vendor possession_metrics: spellF1, macroF1, plus
   the breakdown that explains them: possession precision/recall, alive/dead F1, team accuracy,
   transition F1 and latency). The report gives the mean over matches first (the one "Dev
   spellF1" per run that RESULTS.md asks for, D15), then every match on its own.

Assumptions that differ from the competition match, so the dev score is a COMPARISON tool and
not a prediction of the real score:
- The truth is the data provider's own possession flag (DFL), not the challenge provider's.
  Both are mechanical from tracking, but their rules for "dead" and "owning" may differ a bit.
- Gaps are placed at random. Real gaps are not random (they happen when the ball is hidden).
- Players come straight from tracking: no tracklet swaps, no empty slots at random.

Rules this module follows (CLAUDE.md):
- The data location comes ONLY from IDSSE_DATA_DIR: a folder outside the repo. Nothing is read
  at import time. IDSSE is public, but we treat it like match data: humans run it (see
  experiments/README.md).
- Only aggregates are printed or returned for printing: no frame values, no coordinates.
- Tests run on SYNTHETIC files in the IDSSE layout (tests/synthetic_idsse.py).

THE IDSSE FILE LAYOUT this reader expects (DFL XML, as read by floodlight and kloppy):
- match information: an element with the attributes PitchX and PitchY (pitch size in metres),
  `Team` elements with `Role` ("home" or "guest") and `TeamId`, and `Player` elements below them
  with `PersonId`.
- positions: one `FrameSet` per person (or the ball) per half, with the attributes `GameSection`
  ("firstHalf", "secondHalf"), `TeamId` ("BALL" for the ball) and `PersonId`. Its `Frame` elements
  carry `N` (frame number), `X`, `Y`, `S` (speed) and, for the ball, `BallStatus` (1 alive, 0
  dead) and `BallPossession` (1 home, 2 away).
`inspect_structure()` prints the element and attribute NAMES of a real file (no values), so a
human can check these names against the real files in one minute.
"""

import hashlib
import json
import os
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from falafelona import deadball, possession, vendor
from falafelona.ball import DataCleaner
from falafelona.io import Match, read_cache, write_cache
from falafelona.pipeline import apply_x_policy

pm = vendor.possession_metrics

FPS = 25.0                        # TRACAB tracking in IDSSE, same as our match
SLOTS_PER_TEAM = 23               # slots 0..22 home, 23..45 away, like the competition ATD
N_SLOTS = 2 * SLOTS_PER_TEAM
BALL_TEAM_ID = "BALL"
PERIODS = {"firstHalf": 1, "secondHalf": 2}   # extra time and penalties are not used
HALFTIME_PAD = 500                # ungraded frames between the halves; longer than any window
# Bump when the parsing changes, so old caches are not reused.
LOADER_VERSION = 1

ALIVE, DEAD = 1, 0
POSSESSION_HOME, POSSESSION_AWAY = 1, 2

TIGHT_TAU, LOOSE_TAU = min(pm.TAUS), max(pm.TAUS)   # transition tolerances: 12 and 25 frames
HEADLINE_IOU = pm.SPELL_IOU_HEADLINE                # possession precision/recall at IoU 0.5
# Besides spellF1, these get a paired difference in the report: they say WHY spellF1 moved.
PAIRED_EXTRA = ("spell_precision", "spell_recall", "alive_dead_f1", "team_acc", "trans_f1_tight")


@dataclass
class DevMatch:
    """One dev-set match: the data the stages need plus the truth they are scored against."""

    match_id: str
    match: Match
    truth: np.ndarray        # str[n] in {H, A, D, X}; X = not scored (halftime, no flag)


# --- where the files are -----------------------------------------------------------------------


def idsse_dir():
    """The IDSSE folder from IDSSE_DATA_DIR. Checked: set, exists, and outside the repo."""
    raw = os.environ.get("IDSSE_DATA_DIR")
    if not raw:
        raise RuntimeError("IDSSE_DATA_DIR is not set. See experiments/README.md.")
    path = Path(raw).expanduser().resolve()
    repo = vendor.REPO_ROOT
    if path == repo or repo in path.parents:
        raise RuntimeError(f"IDSSE_DATA_DIR ({path}) is inside the repo. Move the data outside.")
    if not path.is_dir():
        raise RuntimeError(f"IDSSE_DATA_DIR ({path}) does not exist.")
    return path


def find_files(root, match_id):
    """(match_information_path, positions_path) of one match, found by its id in the file name.

    Looks in the whole folder, so both a flat folder and one folder per match work. The two
    files are told apart by "matchinformation" / "positions" in the name (case ignored).
    """
    root = Path(root)
    wanted = match_id.lower()
    found = {"info": [], "positions": []}
    for path in sorted(root.rglob("*.xml")):
        name = path.name.lower()
        if wanted not in name:
            continue
        if "matchinformation" in name or "match_information" in name:
            found["info"].append(path)
        elif "position" in name:
            found["positions"].append(path)
    for kind, paths in found.items():
        if len(paths) != 1:
            raise FileNotFoundError(
                f"match {match_id}: expected exactly one {kind} XML under {root}, "
                f"found {len(paths)}"
            )
    return found["info"][0], found["positions"][0]


# --- reading the XML ---------------------------------------------------------------------------


def _local(tag):
    """Tag name without an XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _field(element, name):
    """A value stored as an attribute, or else as the text of a child element."""
    value = element.get(name)
    if value is None:
        for child in element:
            if _local(child.tag) == name:
                return child.text
    return value


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_match_info(path):
    """Pitch size, team ids and the team of every player, from the match information XML.

    Returns {"pitch": (length, width) or None, "teams": {"H": id, "A": id},
    "players": {person_id: "H" or "A"}}.
    """
    root = ET.parse(path).getroot()
    pitch = None
    teams, players = {}, {}
    for element in root.iter():
        if element.get("PitchX") is not None and element.get("PitchY") is not None:
            pitch = (_float(element.get("PitchX")), _float(element.get("PitchY")))
        if _local(element.tag) != "Team":
            continue
        role = (element.get("Role") or "").lower()
        side = {"home": "H", "guest": "A", "away": "A"}.get(role)
        if side is None or element.get("TeamId") is None:
            continue
        teams[side] = element.get("TeamId")
        for node in element.iter():
            if _local(node.tag) == "Player" and node.get("PersonId"):
                players[node.get("PersonId")] = side
    if set(teams) != {"H", "A"}:
        raise ValueError(f"{path}: could not find a home and a guest Team with a TeamId")
    if pitch is not None and not all(np.isfinite(pitch)):
        pitch = None
    return {"pitch": pitch, "teams": teams, "players": players}


def _read_frames(frameset):
    """The Frame rows of one FrameSet as arrays."""
    rows = [f for f in frameset if _local(f.tag) == "Frame"]
    return {
        "n": np.array([_int(_field(f, "N")) for f in rows], dtype=np.int64),
        "x": np.array([_float(_field(f, "X")) for f in rows]),
        "y": np.array([_float(_field(f, "Y")) for f in rows]),
        "s": np.array([_float(_field(f, "S")) for f in rows]),
        "possession": np.array([_int(_field(f, "BallPossession")) for f in rows], dtype=np.int8),
        "status": np.array([_int(_field(f, "BallStatus")) for f in rows], dtype=np.int8),
    }


def read_positions(path, info):
    """Ball and players of both halves: {period: {"ball": frames, "players": {id: frames}}}.

    Persons that are not in the match information (referees) are ignored, and so is everything
    that is not the first or second half.
    """
    periods = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if _local(element.tag) != "FrameSet":
            continue
        period = PERIODS.get(element.get("GameSection"))
        if period is not None:
            part = periods.setdefault(period, {"ball": None, "players": {}})
            person = element.get("PersonId")
            if element.get("TeamId") == BALL_TEAM_ID:
                part["ball"] = _read_frames(element)
            elif person in info["players"]:
                part["players"][person] = _read_frames(element)
        element.clear()
    for period in PERIODS.values():
        if period not in periods or periods[period]["ball"] is None:
            raise ValueError(f"{path}: no ball FrameSet found for period {period}")
    return periods


# --- turning it into a Match -------------------------------------------------------------------


def _slots(players, info):
    """Slot of every person: home sorted by id in 0..22, away sorted by id in 23..45."""
    seen = {"H": set(), "A": set()}
    for part in players.values():
        for person in part["players"]:
            seen[info["players"][person]].add(person)
    slot_of = {}
    for side, offset in (("H", 0), ("A", SLOTS_PER_TEAM)):
        people = sorted(seen[side])
        if len(people) > SLOTS_PER_TEAM:
            raise ValueError(f"{len(people)} tracked {side} players, at most {SLOTS_PER_TEAM}")
        slot_of.update({person: offset + i for i, person in enumerate(people)})
    slot_team = np.array(["H"] * SLOTS_PER_TEAM + ["A"] * SLOTS_PER_TEAM)
    return slot_of, slot_team


def _rows(frames, start, length):
    """Row of each frame number inside its half, dropping numbers outside it."""
    row = frames["n"] - start
    return row, (row >= 0) & (row < length)


def build_match(periods, info, pitch, origin="auto", pad=HALFTIME_PAD):
    """(Match, truth) from the parsed positions. Halves are placed one after the other.

    origin: "centre" if X/Y count from the middle of the pitch (the DFL convention), "corner" if
    they already count from a corner, "auto" to decide from the data (the median X of all
    positions is near 0 for centre, near half the length for corner). The Match always counts
    from a corner, like ours.
    """
    length_m, width_m = pitch
    spans = {}
    for period, part in periods.items():
        start = int(part["ball"]["n"].min())
        spans[period] = (start, int(part["ball"]["n"].max()) - start + 1)
    n_first, n_second = spans[1][1], spans[2][1]
    n = n_first + pad + n_second
    offset = {1: 0, 2: n_first + pad}

    slot_of, slot_team = _slots(periods, info)
    period_col = np.zeros(n, dtype=np.int8)
    period_col[:n_first] = 1
    period_col[offset[2]:] = 2

    ball_xy = np.full((n, 2), np.nan)
    ball_speed = np.full(n, np.nan)
    players_xy = np.full((n, N_SLOTS, 2), np.nan)
    players_speed = np.full((n, N_SLOTS), np.nan)
    poss = np.full(n, -1, dtype=np.int8)
    status = np.full(n, -1, dtype=np.int8)

    for period, part in periods.items():
        start, length = spans[period]
        base = offset[period]
        ball = part["ball"]
        row, keep = _rows(ball, start, length)
        row = base + row[keep]
        ball_xy[row] = np.column_stack([ball["x"][keep], ball["y"][keep]])
        ball_speed[row] = ball["s"][keep]
        poss[row], status[row] = ball["possession"][keep], ball["status"][keep]
        for person, frames in part["players"].items():
            row, keep = _rows(frames, start, length)
            row = base + row[keep]
            slot = slot_of[person]
            players_xy[row, slot] = np.column_stack([frames["x"][keep], frames["y"][keep]])
            players_speed[row, slot] = frames["s"][keep]

    if origin == "auto":
        # Centre origin: the middle of everything is near x = 0. Corner origin: near length / 2.
        # The median, not the minimum: a ball a metre past the line is normal in both.
        middle = np.nanmedian(np.concatenate([ball_xy[:, 0], players_xy[..., 0].ravel()]))
        origin = "centre" if abs(middle) < length_m / 4.0 else "corner"
    if origin == "centre":
        shift = np.array([length_m / 2.0, width_m / 2.0])
        ball_xy = ball_xy + shift
        players_xy = players_xy + shift
    elif origin != "corner":
        raise ValueError(f"origin must be 'auto', 'centre' or 'corner', not {origin!r}")

    truth = np.full(n, "X", dtype="<U1")
    truth[status == DEAD] = "D"
    truth[(status == ALIVE) & (poss == POSSESSION_HOME)] = "H"
    truth[(status == ALIVE) & (poss == POSSESSION_AWAY)] = "A"

    match = Match(
        n_frames=n,
        fps=FPS,
        period=period_col,
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        players_xy=players_xy,
        players_speed=players_speed,
        slot_team=slot_team,
        pitch=(float(length_m), float(width_m)),
    )
    return match, truth


def _cache_file(root, match_id, info_path, positions_path, pitch, origin, pad):
    """Cache named by the source files' size and mtime, so a changed file is re-parsed."""
    key = hashlib.sha256()
    for path in (info_path, positions_path):
        stat = os.stat(path)
        key.update(f"{stat.st_size}:{stat.st_mtime_ns};".encode())
    key.update(f"{pitch};{origin};{pad};{LOADER_VERSION}".encode())
    return Path(root) / "cache" / f"{match_id}_{key.hexdigest()[:16]}.npz"


def _read_dev_cache(path):
    """The cached arrays of one match, or None (then it is parsed again; see io.read_cache).

    The frame count is not known before parsing, so it is taken from `period` and every other
    array must agree with it.
    """
    probe = read_cache(path, {"period": (None,)}) if Path(path).is_file() else None
    if probe is None:
        return None
    n = probe["period"].size
    return read_cache(path, {
        "period": (n,), "ball_xy": (n, 2), "ball_speed": (n,), "players_xy": (n, N_SLOTS, 2),
        "players_speed": (n, N_SLOTS), "slot_team": (N_SLOTS,), "truth": (n,),
    })


def load_idsse(match_id, root=None, pitch=None, origin="auto", pad=HALFTIME_PAD, use_cache=True):
    """Load one IDSSE match into a DevMatch.

    pitch: (length, width) in metres to use when the match information has none. The pitch of
    the match information always wins when it is there.
    """
    root = Path(root) if root is not None else idsse_dir()
    info_path, positions_path = find_files(root, match_id)
    info = read_match_info(info_path)
    size = info["pitch"] or (tuple(pitch) if pitch else None)
    if size is None:
        raise ValueError(
            f"match {match_id}: no PitchX/PitchY in the match information; pass pitch=(105, 68)"
        )
    cache = _cache_file(root, match_id, info_path, positions_path, size, origin, pad)
    stored = _read_dev_cache(cache) if use_cache else None
    if stored is not None:
        match = Match(
            n_frames=int(stored["period"].size),
            fps=FPS,
            period=stored["period"],
            ball_xy=stored["ball_xy"],
            ball_speed=stored["ball_speed"],
            players_xy=stored["players_xy"],
            players_speed=stored["players_speed"],
            slot_team=stored["slot_team"],
            pitch=(float(size[0]), float(size[1])),
        )
        return DevMatch(match_id, match, stored["truth"])
    periods = read_positions(positions_path, info)
    match, truth = build_match(periods, info, size, origin, pad)
    if use_cache:
        write_cache(
            cache,
            period=match.period,
            ball_xy=match.ball_xy,
            ball_speed=match.ball_speed,
            players_xy=match.players_xy,
            players_speed=match.players_speed,
            slot_team=match.slot_team,
            truth=truth,
        )
    return DevMatch(match_id, match, truth)


# --- degrading the ball track ------------------------------------------------------------------


def _degrade_settings(section, fps):
    """Check the `degrade:` section of configs/devset.yaml. Returns plain numbers."""
    for key in ("missing_share", "gaps"):
        if key not in section:
            raise ValueError(f"devset config: degrade needs '{key}'")
    share = section["missing_share"]
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share < 1:
        raise ValueError(f"devset config: degrade.missing_share must be in (0, 1), got {share!r}")
    bins = []
    for item in section["gaps"]:
        low, high = item["min_s"], item["max_s"]
        frames_low = int(np.floor(low * fps + 1e-9)) + 1   # same bin edges as scripts/ball_eda.py
        frames_high = int(np.floor(high * fps + 1e-9))
        if frames_high < frames_low:
            raise ValueError(f"devset config: gap bin {low}-{high} s holds no whole frame")
        if item["share"] < 0:
            raise ValueError("devset config: a gap bin share is negative")
        bins.append((frames_low, frames_high, float(item["share"])))
    total = sum(b[2] for b in bins)
    if not bins or abs(total - 1.0) > 1e-6:
        raise ValueError(f"devset config: gap bin shares must add up to 1, they add up to {total}")
    faults = section.get("false_detections") or {}
    fault_share = faults.get("share", 0.0)
    if not 0 <= fault_share < 1:
        raise ValueError("devset config: false_detections.share must be in [0, 1)")
    fault_low = int(np.floor(faults.get("min_s", 0.04) * fps + 1e-9)) or 1
    fault_high = int(np.floor(faults.get("max_s", 0.2) * fps + 1e-9))
    if fault_share and fault_high < fault_low:
        raise ValueError("devset config: false_detections holds no whole frame")
    return {
        "missing_share": float(share),
        "bins": bins,
        "fault_share": float(fault_share),
        "fault_low": fault_low,
        "fault_high": fault_high,
    }


def _half_ranges(period):
    """[(start, stop)] of the two graded halves."""
    ranges = []
    for half in (1, 2):
        frames = np.flatnonzero(period == half)
        if frames.size:
            ranges.append((int(frames[0]), int(frames[-1]) + 1))
    return ranges


def _place(rng, ranges, blocked, length, tries=200):
    """Start of a free spot for `length` frames inside one half, or None. Marks it blocked.

    A gap keeps one free frame on each side, so two gaps never merge into one longer gap and
    the length distribution stays as asked.
    """
    sizes = np.array([stop - start for start, stop in ranges], dtype=float)
    for _ in range(tries):
        half = rng.choice(len(ranges), p=sizes / sizes.sum())
        low, high = ranges[half]
        first, last = low + 1, high - 1 - length      # one frame of margin at each edge
        if last < first:
            continue
        start = int(rng.integers(first, last + 1))
        if not blocked[start - 1:start + length + 1].any():
            blocked[start - 1:start + length + 1] = True
            return start
    return None


def degrade_ball(match, section, seed):
    """A copy of `match` whose ball track looks like ours. Returns (new_match, report).

    Step 1 (only if false_detections.share > 0): short stretches of seen frames get a wrong
    position, one fixed random point on the pitch per stretch, like a detector locked onto the
    wrong object. These are what the `reject` ball method is for.
    Step 2: gaps are cut until `missing_share` of the graded frames have no ball. The gap lengths
    follow the bins of `section["gaps"]` (each bin's share of the MISSING frames, like the EDA
    report), the length inside a bin is uniform, and the longest bin is placed first. Each bin
    may overshoot its target by at most one gap. Gaps stay inside one half.
    The truth and the players are untouched. Same seed, same result.
    The report holds counts only: wanted, placed and unplaced gaps, and the missing share reached.
    """
    settings = _degrade_settings(section, match.fps)
    rng = np.random.default_rng(seed)
    ranges = _half_ranges(match.period)
    graded = int(sum(stop - start for start, stop in ranges))
    ball_xy = match.ball_xy.copy()
    ball_speed = match.ball_speed.copy()
    report = {"false_stretches": 0, "gaps_placed": 0, "gaps_unplaced": 0}
    if not graded:
        return match, {**report, "missing_share": None}

    if settings["fault_share"]:
        report["false_stretches"] = _inject_false(rng, ball_xy, ranges, graded, match.pitch,
                                                   settings)

    blocked = np.zeros(match.n_frames, dtype=bool)
    wanted_frames = settings["missing_share"] * graded
    for low, high, share in sorted(settings["bins"], reverse=True):
        target, done = round(wanted_frames * share), 0
        while done < target:
            length = int(rng.integers(low, high + 1))
            start = _place(rng, ranges, blocked, length)
            if start is None:
                report["gaps_unplaced"] += 1
                break                              # this bin has no room left
            ball_xy[start:start + length] = np.nan
            ball_speed[start:start + length] = np.nan
            done += length
            report["gaps_placed"] += 1

    ball_speed[np.isnan(ball_xy[:, 0])] = np.nan
    graded_mask = match.period > 0
    report["missing_share"] = float(np.isnan(ball_xy[graded_mask, 0]).mean())
    return replace(match, ball_xy=ball_xy, ball_speed=ball_speed), report


def _inject_false(rng, ball_xy, ranges, graded, pitch, settings):
    """Overwrite short stretches of seen ball positions with one wrong point each."""
    wanted = round(settings["fault_share"] * graded)
    length_m, width_m = pitch
    done = stretches = 0
    for _ in range(100 * wanted + 100):           # bounded: a ball that is never seen ends it
        if done >= wanted:
            break
        start = int(rng.integers(0, ranges[-1][1]))
        length = int(rng.integers(settings["fault_low"], settings["fault_high"] + 1))
        point = rng.uniform([0.0, 0.0], [length_m, width_m])    # drawn first: keeps draws fixed
        inside = [r for r in ranges if r[0] <= start < r[1]]
        if not inside:
            continue
        stop = min(start + length, inside[0][1])
        seen = ~np.isnan(ball_xy[start:stop, 0])
        if not seen.any():
            continue
        ball_xy[start:stop][seen] = point
        done += int(seen.sum())
        stretches += 1
    return stretches


def match_seed(base_seed, match_id):
    """A different but fixed seed for every match."""
    sequence = np.random.SeedSequence([base_seed, zlib.crc32(match_id.encode())])
    return int(sequence.generate_state(1)[0])


# --- running the pipeline and scoring ----------------------------------------------------------


def prepare_stages(match, cfg):
    """The stages before possession: cleaned ball and dead score. Shared between configs."""
    cleaner = DataCleaner(match, cfg)
    cleaned = cleaner.return_cleaned_data()
    dead_score = deadball.detect(cleaned, cleaner.track, cfg)
    return cleaned, cleaner.track, dead_score


def predict_labels(match, cfg, prepared=None):
    """The pipeline stages on one Match: the same calls as pipeline.run, without the file.

    prepared: the output of prepare_stages for this match and config, to skip rerunning them.
    """
    cleaned, _track, dead_score = prepared or prepare_stages(match, cfg)
    labels = possession.predict(cleaned, dead_score, cfg)
    return apply_x_policy(labels, cleaned, cfg)


def _num(value):
    """A plain float, or None when there was nothing to measure (no matched switch).

    None rather than NaN, so two identical runs compare equal (NaN != NaN).
    """
    return None if value is None else float(value)


def score_labels(truth, labels, bootstrap_n=0):
    """The official metric on one match. Plain numbers only: no frame numbers leave here.

    Never copy the transitions' pairs / missed / spurious lists out of the result: they hold
    frame numbers, which the aggregates-only rule does not allow.
    """
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=bootstrap_n)
    spell = result["spell_f1"][HEADLINE_IOU]
    tight = result["transitions"][TIGHT_TAU]
    loose = result["transitions"][LOOSE_TAU]
    per_class = result["per_class"]
    score = {
        # headline
        "spellF1": float(result["spell_f1_mean"]),
        "macroF1": float(result["macro_f1"]),
        # possessions
        "truth_possessions": int(result["n_gt_spells"]),
        "our_possessions": int(result["n_pred_spells"]),
        "count_error": int(result["spell_count_error"]),
        "spell_precision": float(spell["precision"]),
        "spell_recall": float(spell["recall"]),
        "spell_missed": int(spell["n_missed"]),
        "spell_spurious": int(spell["n_spurious"]),
        "duration_s": float(result["mean_spell_duration_s"]),
        "truth_duration_s": float(result["gt_mean_spell_duration_s"]),
        "fragmentation": float(result["fragmentation"]),
        # where the errors come from
        "alive_dead_f1": float(result["alive_dead_f1"]),
        "team_acc": float(result["team_acc_both_alive"]),
        "f1_H": float(per_class["H"]["f1"]),
        "f1_A": float(per_class["A"]["f1"]),
        "f1_D": float(per_class["D"]["f1"]),
        # switch timing (latency > 0 means our switches come late)
        "trans_f1_tight": float(tight["f1"]),
        "trans_f1_loose": float(loose["f1"]),
        "latency_mean": _num(loose["mean_latency"]),
        "latency_median": _num(loose["median_latency"]),
        "latency_abs": _num(loose["mean_abs_latency"]),
    }
    if bootstrap_n:  # the spellF1 interval exists only when asked for
        score["spell_ci_low"] = _num(result["spell_bootstrap_low"])
        score["spell_ci_high"] = _num(result["spell_bootstrap_high"])
    return score


def _stage_key(cfg):
    """The config without its possession section, as one string.

    Configs with the same key get the same cleaned ball and dead score, so those stages run
    once per match. Assumes DataCleaner and deadball.detect do not read cfg["possession"].
    """
    rest = {key: value for key, value in cfg.items() if key != "possession"}
    return json.dumps(rest, sort_keys=True, default=str)


def run_devset(configs, dev_cfg, matches, degrade=True, bootstrap_n=0):
    """Score every config on every match. The degraded ball is the same for every config.

    configs: {name: loaded pipeline config}. matches: list of DevMatch.
    bootstrap_n: spellF1 bootstrap draws per match (0 = off, it is slow).
    Returns {"matches": {id: {"ball_missing": share, **report}}, "scores": {name: {id: score}}}.
    """
    out = {"matches": {}, "scores": {name: {} for name in configs}}
    for dev in matches:
        match = dev.match
        if degrade:
            match, report = degrade_ball(
                match, dev_cfg["degrade"], match_seed(dev_cfg["seed"], dev.match_id)
            )
        else:
            graded = match.period > 0
            report = {"missing_share": float(np.isnan(match.ball_xy[graded, 0]).mean())}
        out["matches"][dev.match_id] = report
        prepared = {}  # stage key -> stages, for this match only
        for name, cfg in configs.items():
            key = _stage_key(cfg)
            if key not in prepared:
                prepared[key] = prepare_stages(match, cfg)
            labels = predict_labels(match, cfg, prepared[key])
            out["scores"][name][dev.match_id] = score_labels(dev.truth, labels, bootstrap_n)
    return out


# --- the report --------------------------------------------------------------------------------


def _mean(values):
    """Mean of the values that exist (None = nothing measured), or None if there are none."""
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def summarise(scores):
    """Mean over matches of every metric of one config, plus the spellF1 range."""
    keys = next(iter(scores.values())).keys()
    total = {key: _mean([s.get(key) for s in scores.values()]) for key in keys}
    spell = [s["spellF1"] for s in scores.values()]
    total["spellF1_min"], total["spellF1_max"] = min(spell), max(spell)
    return total


def paired_difference(scores, base_scores, metric="spellF1"):
    """Mean difference in `metric` over the matches, one pair per match, and the matches won.

    Matches where either side has no value for the metric are left out.
    """
    diffs = [
        scores[m][metric] - base_scores[m][metric]
        for m in scores
        if scores[m].get(metric) is not None and base_scores[m].get(metric) is not None
    ]
    return _mean(diffs), int(sum(d > 0 for d in diffs)), len(diffs)


def _verdict(diff):
    """'noise' inside the spellF1 noise band, else 'better' or 'worse'."""
    if abs(diff) < pm.SPELL_NOISE_BAND:
        return "noise"
    return "better" if diff > 0 else "worse"


def _signed(value):
    """'+4.2' / '-3.0' / '--', so a lag reads as a lag at a glance."""
    return "--" if value is None else f"{value:+.1f}"


def _signed3(value):
    """'+0.031' / '--' for a difference that can be missing."""
    return "--" if value is None else f"{value:+.3f}"


def _diff(score, base, metric):
    """score[metric] - base[metric], or None if either is missing."""
    if score.get(metric) is None or base.get(metric) is None:
        return None
    return score[metric] - base[metric]


def _plain(value):
    """'7.1' / '--' for a value that can be missing."""
    return "--" if value is None else f"{value:.1f}"


def _match_lines(s):
    """The full breakdown of one config on one match, as report lines."""
    headline = f"    spellF1 {s['spellF1']:.3f}, macroF1 {s['macroF1']:.3f}"
    possessions = (
        f"    possessions: truth {s['truth_possessions']} / ours {s['our_possessions']} "
        f"({s['count_error']:+d}), precision {s['spell_precision']:.3f}, "
        f"recall {s['spell_recall']:.3f} (IoU {HEADLINE_IOU}), "
        f"missed {s['spell_missed']}, fake {s['spell_spurious']}, "
        f"fragmentation {s['fragmentation']:.2f}, "
        f"duration {s['duration_s']:.1f} s vs truth {s['truth_duration_s']:.1f} s"
    )
    errors = (
        f"    errors: alive/dead F1 {s['alive_dead_f1']:.3f}, team acc {s['team_acc']:.3f}, "
        f"F1 H/A/D {s['f1_H']:.3f}/{s['f1_A']:.3f}/{s['f1_D']:.3f}"
    )
    switches = (
        f"    switches: F1 {s['trans_f1_tight']:.3f} at +/-{TIGHT_TAU}, "
        f"{s['trans_f1_loose']:.3f} at +/-{LOOSE_TAU} frames, "
        f"latency mean {_signed(s['latency_mean'])} "
        f"/ median {_signed(s['latency_median'])} frames "
        f"(mean |latency| {_plain(s['latency_abs'])})"
    )
    lines = [headline, possessions, errors, switches]
    if s.get("spell_ci_low") is not None:
        lines.append(
            f"    spellF1 bootstrap 5-95%: {s['spell_ci_low']:.3f} to {s['spell_ci_high']:.3f}"
        )
    return lines


def format_report(result, degrade=True):
    """One block per config: the mean over matches, then every match on its own.

    With several configs, the paired difference against the first one follows: one overall
    line (mean difference, matches won, verdict, and the mean change of the PAIRED_EXTRA
    metrics, which say why spellF1 moved), then one line per match.
    This is what gets pasted into the issue; the mean spellF1 is the "Dev spellF1" of RESULTS.md.
    """
    names = list(result["scores"])
    lines = [f"Dev set (IDSSE), {len(result['matches'])} matches, aggregates only"]
    lines.append(
        "ball: degraded to look like ours (fixed seed)" if degrade
        else "ball: as tracked, NOT degraded (an upper bound)"
    )
    for match_id, report in result["matches"].items():
        extra = ""
        if "gaps_placed" in report:
            extra = f", {report['gaps_placed']} gaps"
            if report["false_stretches"]:
                extra += f", {report['false_stretches']} false stretches"
            if report["gaps_unplaced"]:
                extra += f", {report['gaps_unplaced']} bins ran out of room"
        share = report["missing_share"]
        shown = "n/a" if share is None else f"{100 * share:.1f}%"
        lines.append(f"  {match_id}: ball missing {shown}{extra}")

    for name in names:
        scores = result["scores"][name]
        total = summarise(scores)
        lines.append(f"config {name}")
        lines.append(
            f"  mean: spellF1 {total['spellF1']:.3f} "
            f"(matches {total['spellF1_min']:.3f} to {total['spellF1_max']:.3f}), "
            f"macroF1 {total['macroF1']:.3f}"
        )
        for match_id, s in scores.items():
            lines.append(f"  {match_id}")
            lines.extend(_match_lines(s))

    if len(names) > 1:
        base_name = names[0]
        base = result["scores"][base_name]
        lines.append(
            f"paired spellF1 difference against {base_name} "
            f"(differences under {pm.SPELL_NOISE_BAND} are noise):"
        )
        for name in names[1:]:
            scores = result["scores"][name]
            diff, wins, count = paired_difference(scores, base)
            extra = ", ".join(
                f"{metric} {_signed3(paired_difference(scores, base, metric)[0])}"
                for metric in PAIRED_EXTRA
            )
            lines.append(
                f"  {name}: {diff:+.3f}, better on {wins} of {count} matches "
                f"({_verdict(diff)}); mean {extra}"
            )
            for match_id, s in scores.items():
                b = base[match_id]
                one = s["spellF1"] - b["spellF1"]
                per_metric = ", ".join(
                    f"{metric} {_signed3(_diff(s, b, metric))}" for metric in PAIRED_EXTRA
                )
                lines.append(f"    {match_id}: spellF1 {one:+.3f} ({_verdict(one)}), {per_metric}")
    return "\n".join(lines)


# --- checking the file layout ------------------------------------------------------------------


def inspect_structure(path, max_elements=300_000):
    """Element and attribute NAMES of an XML file, with counts. No values, no rows.

    For a human to run on one real IDSSE file, to confirm the names this module reads
    (PitchX, Team/Role, FrameSet/GameSection, Frame/N/X/Y/S/BallStatus/BallPossession).
    Stops after `max_elements` elements, so a 150 MB file is quick.
    """
    stack, seen, count = [], {}, 0
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "end":
            stack.pop()
            continue
        stack.append(_local(element.tag))
        entry = seen.setdefault("/".join(stack), {"count": 0, "attributes": set()})
        entry["count"] += 1
        entry["attributes"].update(element.attrib)
        count += 1
        if count >= max_elements:
            break
    lines = [f"structure of one XML file (names only, first {count:,} elements at most)"]
    for tree_path, entry in seen.items():
        names = ", ".join(sorted(entry["attributes"])) or "-"
        lines.append(f"  {tree_path}  x{entry['count']:,}  attributes: {names}")
    return "\n".join(lines)


def summary(dev):
    """Shapes and counts only: safe to print and to share."""
    match = dev.match
    graded = match.period > 0
    return {
        "match": dev.match_id,
        "frames": match.n_frames,
        "graded_frames": int(graded.sum()),
        "ball_missing_share": round(float(np.isnan(match.ball_xy[graded, 0]).mean()), 3),
        "truth": {label: int(np.sum(dev.truth == label)) for label in "HADX"},
        "pitch": match.pitch,
    }