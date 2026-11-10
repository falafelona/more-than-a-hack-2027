"""Generator for a small SYNTHETIC tracking bundle in the official file layout.

Tests never touch real match data. write_bundle() writes random numbers in the right format
(metadata XML + raw ATD text) to a temporary folder at test time, so no data-shaped file is ever
committed. It returns the values it wrote, in metres, so tests can check the loader exactly.

Built in on purpose, because the real tracking has them:
- ball gaps: stretches where the ball is NaN
- ball out of bounds: a stretch with the ball outside 0..1
- slot swaps: a player's tracklet ends and a different slot picks the player up
- empty slots: 46 slots for 22 players, most slots empty most of the time
- Smart Tagging (task 1.7): a kickoff per half, a throw-in tagged Team N/A, club names in odd
  case, one tag running past the end and one starting after it
- homography (task 1.7): one matrix per frame, with a short invalid (NaN) stretch
"""

from pathlib import Path

import numpy as np

from falafelona.vendor import games

N_SLOTS = 46
HOME_SLOTS = range(23)    # slots 1..23 belong to the home team in the metadata
AWAY_SLOTS = range(23, 46)   # slots 24..46 to the away team

# Synthetic Smart Tagging instances: (start_s, end_s, code, {group: text}). At 25 fps and the
# default 500 frames (20 s), the fifth is clipped at the end and the sixth is dropped.
TAGS = [
    (0.0, 2.0, "SET PIECES", {"Team": "Barcelona", "Half": "1st Half", "Type": "Kick Off (Start)"}),
    (2.0, 6.4, "BUILD UP", {"Team": "Barcelona", "Half": "1st Half"}),
    (8.0, 9.0, "SET PIECES", {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}),
    (10.0, 14.0, "Progression", {"Team": "ATLETIC CLUB", "Half": "1st Half"}),
    (18.0, 25.0, "SET PIECES", {"Team": "Atletic Club", "Half": "2nd Half",
                                "Type": "Kick Off (Start)"}),
    (30.0, 31.0, "GOALS", {"Team": "Barcelona", "Half": "2nd Half"}),
]
HOMOGRAPHY_INVALID = (300, 310)   # rows with no matrix, like broadcast cut-aways


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

    _write_tags(Path(paths["smart_tagging"]))
    homography, homography_error, homography_valid = _write_homography(
        Path(paths["homography"]), n_frames, rng
    )

    scale = np.array(games.GAMES[game]["pitch"])
    return {
        "n_frames": n_frames,
        "homography": homography,
        "homography_error": homography_error,
        "homography_valid": homography_valid,
        "ball_xy": ball * scale,
        "players_xy": players * scale,
        "slot_team": np.array(["H"] * len(HOME_SLOTS) + ["A"] * len(AWAY_SLOTS)),
        "ball_gap": gap,
        "ball_out": out,
        "slot_swap": swap,
    }


def _write_tags(path):
    """SportsCode XML in the shape starter/load_smart_tagging.py reads."""
    instances = []
    for i, (start, end, code, labels) in enumerate(TAGS, start=1):
        label_xml = "".join(
            f"<label><group>{group}</group><text>{text}</text></label>"
            for group, text in labels.items()
        )
        instances.append(
            f"    <instance><ID>{i}</ID><start>{start}</start><end>{end}</end>"
            f"<code>{code}</code>{label_xml}</instance>"
        )
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n<file>\n  <ALL_INSTANCES>\n"
        + "\n".join(instances)
        + "\n  </ALL_INSTANCES>\n</file>\n"
    )


def _write_homography(path, n_frames, rng):
    """One plausible pitch->pixel matrix per frame; NaN and valid=0 on HOMOGRAPHY_INVALID."""
    base = np.array([[1800.0, -500.0, 45.0], [2.0, 170.0, 255.0], [0.0, -0.5, 1.0]])
    matrices = base + rng.normal(0.0, 0.5, size=(n_frames, 3, 3))
    matrices[:, 2, 2] = 1.0
    error = rng.uniform(0.1, 0.5, size=n_frames)
    valid = np.ones(n_frames, dtype=bool)
    lo, hi = HOMOGRAPHY_INVALID
    matrices[lo:hi] = np.nan
    valid[lo:hi] = False

    columns = [f"h{r}{c}" for r in range(3) for c in range(3)]
    lines = [",".join(["frame", *columns, "reprojection_error", "valid"])]
    for row in range(n_frames):
        values = [_fmt(v) for v in matrices[row].ravel()]
        lines.append(f"{row},{','.join(values)},{error[row]:.5f},{int(valid[row])}")
    path.write_text("\n".join(lines) + "\n")
    # Return what a reader gets back from the 5-decimal text.
    return np.round(matrices, 5), np.round(error, 5), valid


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
