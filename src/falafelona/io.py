"""ws:int: locate and load the match data into a `Match`.

    from falafelona.io import load_match
    match = load_match()          # reads $MTAH_DATA_DIR; only pipeline.py should call this

Rules this module enforces:
- The data location comes ONLY from MTAH_DATA_DIR (D1). There is no fallback, because the
  organisers' games.py falls back to vendor/challenge/data/, which is inside this repo.
- MTAH_DATA_DIR must resolve to a folder outside this repo.
- Nothing is loaded at import time. Importing this module reads no data.
- File paths come from games.require(); no file name is spelled out here.
- Parsed arrays are cached under $MTAH_DATA_DIR/cache/, never in the repo (D2).
- Nothing here prints rows or values. summary() gives shapes and counts only.

The raw ATD is parsed with numpy only, the same way the organisers' starter/baseline.py reads it
(no kloppy). Alignment: submission row = ATD frame_id - 1 (starter/load_atd.py). The 46 slots are
tracklets, not players: identities swap between them.

Smart Tagging and the homography are read with the organisers' own starter loaders (task 1.7):
- Smart Tagging times are seconds from the first tactical video frame, so seconds x fps is a
  submission row (DATA.md, "Time base -- verified"). Tags are evidence, not ground truth.
- Homography row i is submission row i. It maps NORMALISED pitch coordinates (0-1) to pixels, so
  divide Match positions (metres) by Match.pitch before projecting.
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from falafelona import vendor

games = vendor.games

# Bump when the parsing changes, so old caches are not reused.
# 2: ball_speed and players_speed added (PR #9).
LOADER_VERSION = 2

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN"

# Smart Tagging names clubs; the submission wants a side. Keys are lower case and matched
# case-insensitively. Names from vendor/challenge/DATA.md (Atletic is spelled that way in the
# file). Anything else, including "N/A" (every throw-in), becomes team None.
TAG_TEAMS = {"BAR-ATH": {"barcelona": "H", "atletic club": "A"}}
TAG_HALVES = {"1st half": 1, "2nd half": 2}


@dataclass(frozen=True)
class Event:
    """One Smart Tagging instance, in submission rows. `end` is exclusive (first row after it)."""

    start: int
    end: int
    code: str                 # e.g. "SET PIECES", as in the file
    type: str | None          # the Type tag, e.g. "Kick Off (Start)", None if absent
    team: str | None          # 'H' / 'A', or None for N/A and unknown names
    half: int | None          # 1 / 2 from the Half tag, None if absent
    tags: dict = field(default_factory=dict, compare=False, hash=False)  # every tag group


@dataclass
class Match:
    """Everything the stages need, in METRES, one row per submission frame."""

    n_frames: int
    fps: float
    period: np.ndarray       # int8[n]: 1, 2, or 0 = outside the graded halves
    ball_xy: np.ndarray      # float[n, 2], NaN where the ball is missing
    ball_speed: np.ndarray   # float[n], speed field of the raw ATD as given, NaN where missing
    players_xy: np.ndarray   # float[n, n_slots, 2], NaN where a slot is empty
    players_speed: np.ndarray  # float[n, n_slots], speed field as given, NaN where empty
    slot_team: np.ndarray    # str[n_slots]: 'H' / 'A' / '?'
    pitch: tuple[float, float]
    tags: list = field(default_factory=list)   # list[Event], Smart Tagging, sorted by start
    homography: np.ndarray | None = None       # float[n, 3, 3], normalised pitch -> pixels
    homography_error: np.ndarray | None = None  # float[n], reprojection error, lower is better
    homography_valid: np.ndarray | None = None  # bool[n], the matrix is NaN where False


def data_dir():
    """The data folder from MTAH_DATA_DIR. Checked: set, exists, and outside the repo.

    Based on Mohamed Khaled's version in PR #4.
    """
    raw = os.environ.get("MTAH_DATA_DIR")
    if not raw:
        raise RuntimeError("MTAH_DATA_DIR is not set. See CONTRIBUTING.md.")
    path = Path(raw).expanduser().resolve()
    repo = vendor.REPO_ROOT
    if path == repo or repo in path.parents:
        raise RuntimeError(f"MTAH_DATA_DIR ({path}) is inside the repo. Move the data outside.")
    if not path.is_dir():
        raise RuntimeError(f"MTAH_DATA_DIR ({path}) does not exist.")
    return path


def period_of_frames(n_frames, game=games.GAME):
    """int8[n]: 1 or 2 inside each graded half (official games.half_video_blocks), else 0."""
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in games.half_video_blocks(game):
        lo, hi = max(first, 0), min(last + 1, n_frames)
        if lo < hi:
            period[lo:hi] = half
    return period


def slot_teams(meta_path):
    """str[n_slots]: 'H' for slots of the first team in the metadata (home), 'A' for the second.

    Same convention as the organisers' baseline.py: the first Team listed is the home side.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{Path(meta_path).name}: expected 2 teams, found {len(team_ids)}")
    home, away = team_ids
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{Path(meta_path).name}: no Player entries")
    return np.array(["H" if t == home else "A" if t == away else "?" for t in slots])


def parse_raw(raw_path, n_frames, n_slots, pitch):
    """Raw ATD text -> (ball_xy, players_xy, ball_speed, players_speed).

    Positions are scaled to metres; speeds are kept as given in the file. Each group is
    "x,y,speed". Lines outside 0..n_frames-1 are skipped.
    """
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    ball_speed = np.full(n_frames, np.nan)
    players_speed = np.full((n_frames, n_slots), np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1          # frame_id counts from 1, submission rows from 0
            if not 0 <= row < n_frames:
                continue

            group = parts[2]
            if group and group[0] != MISSING:
                x, y, s = group.split(",", 2)
                ball[row] = float(x), float(y)
                ball_speed[row] = float(s)

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError(
                    f"{Path(raw_path).name}: a line has {len(groups)} slots, "
                    f"metadata lists {n_slots}"
                )
            for slot, group in enumerate(groups):
                if group and group[0] != MISSING:
                    x, y, s = group.split(",", 2)
                    players[row, slot] = float(x), float(y)
                    players_speed[row, slot] = float(s)

    scale = np.array([length, width])
    return ball * scale, players * scale, ball_speed, players_speed


def _norm(text):
    return (text or "").strip().lower()


def load_tags(game, root, n_frames):
    """Smart Tagging -> list[Event] in submission rows, sorted by start.

    Parsing is the organisers' starter/load_smart_tagging.load(). Events starting at or after
    n_frames are dropped and ends are clipped to n_frames. Raises if no tag maps to H or A,
    because that means the club names changed and every team would silently be None.
    """
    teams = TAG_TEAMS[game]
    events = []
    for raw in vendor.smart_tagging.load(game, str(root)):
        start, end = raw["start_frame"], min(raw["end_frame"], n_frames)
        if start >= n_frames or end <= start:
            continue
        tags = raw["tags"]
        events.append(Event(
            start=start,
            end=end,
            code=raw["code"] or "",
            type=tags.get("Type"),
            team=teams.get(_norm(tags.get("Team"))),
            half=TAG_HALVES.get(_norm(tags.get("Half"))),
            tags=dict(tags),
        ))
    if events and not any(event.team for event in events):
        names = sorted({_norm(e.tags.get("Team")) for e in events} - {""})
        raise ValueError(f"no Smart Tagging Team value maps to H or A; TAG_TEAMS has "
                         f"{sorted(teams)}, the file has {names}")
    events.sort(key=lambda event: (event.start, event.end))
    return events


def select_tags(tags, code=None, type=None, team=None):
    """Events matching every given field. code and type compare case-insensitively.

        select_tags(match.tags, code="set pieces", type="kick off (start)")
    """
    return [
        event for event in tags
        if (code is None or _norm(event.code) == _norm(code))
        and (type is None or _norm(event.type) == _norm(type))
        and (team is None or event.team == team)
    ]


def load_homography(game, root, n_frames):
    """(matrices float[n,3,3], error float[n], valid bool[n]), row i = submission row i.

    Parsing is the organisers' starter/load_homography.load(). Raises if the file has fewer
    rows than n_frames or its frame column is not 0, 1, 2, ... (the stated alignment).
    """
    path = games.require(["homography"], game, str(root))[0]
    frames = pd.read_csv(path, usecols=["frame"])["frame"].to_numpy()
    if len(frames) < n_frames or not np.array_equal(frames[:n_frames], np.arange(n_frames)):
        raise ValueError(f"{Path(path).name}: expected frames 0..{n_frames - 1} in order")
    matrices, error, valid = vendor.homography.load(game, str(root))
    return matrices[:n_frames], error[:n_frames].astype(float), valid[:n_frames]


def _cache_path(root, game, meta_path, raw_path, n_frames):
    """Cache file named by the source files' size and mtime, so a changed file is re-parsed."""
    key = hashlib.sha256()
    for path in (meta_path, raw_path):
        stat = os.stat(path)
        key.update(f"{stat.st_size}:{stat.st_mtime_ns};".encode())
    key.update(f"{n_frames};{LOADER_VERSION}".encode())
    return root / "cache" / f"{game}_match_{key.hexdigest()[:16]}.npz"


def load_match(game=games.GAME, n_frames=None, use_cache=True):
    """Load the tactical ATD of `game` from MTAH_DATA_DIR into a Match.

    n_frames defaults to the official submission length (199,319). Tests pass a small number.
    """
    root = data_dir()
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, str(root))
    n = n_frames if n_frames is not None else games.source_frames("tactical", game)
    info = games.GAMES[game]

    slot_team = slot_teams(meta_path)
    cache = _cache_path(root, game, meta_path, raw_path, n) if use_cache else None

    if cache is not None and cache.is_file():
        with np.load(cache) as stored:
            ball_xy, players_xy = stored["ball_xy"], stored["players_xy"]
            ball_speed, players_speed = stored["ball_speed"], stored["players_speed"]
    else:
        ball_xy, players_xy, ball_speed, players_speed = parse_raw(
            raw_path, n, len(slot_team), info["pitch"]
        )
        if cache is not None:
            cache.parent.mkdir(exist_ok=True)
            np.savez(
                cache,
                ball_xy=ball_xy,
                players_xy=players_xy,
                ball_speed=ball_speed,
                players_speed=players_speed,
            )

    homography, homography_error, homography_valid = load_homography(game, root, n)
    return Match(
        n_frames=n,
        fps=float(info["fps"]),
        period=period_of_frames(n, game),
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        players_xy=players_xy,
        players_speed=players_speed,
        slot_team=slot_team,
        pitch=games.GAMES[game]["pitch"],
        tags=load_tags(game, root, n),
        homography=homography,
        homography_error=homography_error,
        homography_valid=homography_valid,
    )


def summary(match):
    """Shapes and counts only: safe to print and to share."""
    ball_seen = ~np.isnan(match.ball_xy[:, 0])
    return {
        "n_frames": match.n_frames,
        "fps": match.fps,
        "ball_xy": match.ball_xy.shape,
        "ball_speed": match.ball_speed.shape,
        "players_xy": match.players_xy.shape,
        "players_speed": match.players_speed.shape,
        "slots_H": int(np.sum(match.slot_team == "H")),
        "slots_A": int(np.sum(match.slot_team == "A")),
        "frames_period_1": int(np.sum(match.period == 1)),
        "frames_period_2": int(np.sum(match.period == 2)),
        "ball_seen_pct": round(100.0 * float(ball_seen.mean()), 1) if match.n_frames else 0.0,
        "tags": len(match.tags),
        "tags_H": sum(event.team == "H" for event in match.tags),
        "tags_A": sum(event.team == "A" for event in match.tags),
        "tags_no_team": sum(event.team is None for event in match.tags),
        "homography_valid_pct": (
            round(100.0 * float(match.homography_valid.mean()), 1)
            if match.homography_valid is not None and match.n_frames else 0.0
        ),
    }
