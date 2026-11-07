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
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from falafelona import vendor

games = vendor.games

# Bump when the parsing changes, so old caches are not reused.
# 2: ball_speed and players_speed added (PR #9).
LOADER_VERSION = 2

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN"


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
    tags: list = field(default_factory=list)   # Smart Tagging, added in task 1.7


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

    return Match(
        n_frames=n,
        fps=float(info["fps"]),
        period=period_of_frames(n, game),
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        players_xy=players_xy,
        players_speed=players_speed,
        slot_team=slot_team,
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
    }
