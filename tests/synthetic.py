"""Generator for a small SYNTHETIC tracking bundle in the official file layout.

Tests never touch real match data. write_bundle() writes random numbers in the right format
(metadata XML + raw ATD text) to a temporary folder at test time, so no data-shaped file is ever
committed. It returns the values it wrote, in metres, so tests can check the loader exactly.

Built in on purpose, because the real tracking has them:
- ball gaps: stretches where the ball is NaN
- ball out of bounds: a stretch with the ball outside 0..1
- slot swaps: a player's tracklet ends and a different slot picks the player up
- empty slots: 46 slots for 22 players, most slots empty most of the time
"""

from pathlib import Path

import numpy as np

from falafelona.vendor import games

N_SLOTS = 46
HOME_SLOTS = range(23)    # slots 1..23 belong to the home team in the metadata
AWAY_SLOTS = range(23, 46)   # slots 24..46 to the away team


def _fmt(value):
    return "NaN" if np.isnan(value) else f"{value:.5f}"


def write_bundle(root, n_frames=500, seed=0, game=games.GAME):
    """Write a synthetic tactical ATD pair under root/ and return what was written.

    Returns a dict with ball_xy and players_xy in METRES (NaN where missing), slot_team, and
    the frame ranges of each built-in feature.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    paths = games.asset_paths(game, str(root), "tactical")
    meta_path, raw_path = Path(paths["atd_meta"]), Path(paths["atd_raw"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Positions normalised 0..1, like the real file.
    ball = rng.uniform(0.05, 0.95, size=(n_frames, 2))
    players = np.full((n_frames, N_SLOTS, 2), np.nan)
    active = [*HOME_SLOTS[:11], *AWAY_SLOTS[:11]]
    players[:, active] = rng.uniform(0.0, 1.0, size=(n_frames, len(active), 2))

    gap = (n_frames // 5, n_frames // 5 + 40)
    ball[gap[0]:gap[1]] = np.nan

    out = (n_frames // 2, n_frames // 2 + 15)
    ball[out[0]:out[1], 0] = 1.03          # past the far goal line

    swap = (3 * n_frames) // 4
    players[swap:, 11] = players[swap:, 0]  # home slot 1 tracklet ends, slot 12 continues
    players[swap:, 0] = np.nan

    _write_meta(meta_path)
    with open(raw_path, "w") as handle:
        for row in range(n_frames):
            slots = ";".join(
                f"{_fmt(x)},{_fmt(y)},{_fmt(np.nan if np.isnan(x) else 1.0)}"
                for x, y in players[row]
            )
            bx, by = ball[row]
            ball_txt = "NaN,NaN,NaN" if np.isnan(bx) else f"{_fmt(bx)},{_fmt(by)},0.00000"
            handle.write(f"{row + 1}:{slots}:{ball_txt}\n")   # frame_id counts from 1

    scale = np.array(games.GAMES[game]["pitch"])
    return {
        "n_frames": n_frames,
        "ball_xy": ball * scale,
        "players_xy": players * scale,
        "slot_team": np.array(["H"] * len(HOME_SLOTS) + ["A"] * len(AWAY_SLOTS)),
        "ball_gap": gap,
        "ball_out": out,
        "slot_swap": swap,
    }


def _write_meta(path):
    """Minimal FIFA/EPTS-style metadata: two teams, then 46 track slots with their team."""
    players = "\n".join(
        f'      <Player id="P{slot + 1}" teamId="{"T1" if slot in HOME_SLOTS else "T2"}">'
        f"<Name>Track {slot + 1}</Name><ShirtNumber>{slot + 1}</ShirtNumber></Player>"
        for slot in range(N_SLOTS)
    )
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<main>\n  <Metadata>\n"
        '    <Teams>\n      <Team id="T1"><Name>Synthetic Home</Name></Team>\n'
        '      <Team id="T2"><Name>Synthetic Away</Name></Team>\n    </Teams>\n'
        f"    <Players>\n{players}\n    </Players>\n"
        "  </Metadata>\n</main>\n"
    )
