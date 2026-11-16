"""A scripted SYNTHETIC match with two graded halves, for end-to-end tests (task M2).

The small synthetic bundle (tests/synthetic.py) has 500 frames, all outside the graded halves,
so ball speed is never computed and the still-ball, fast-ball, restart and half logic never run
on it. This match is built in memory, in metres, with every one of those situations scripted:

    frames      period  what happens
    0-199       0       before the first half (ungraded)
    200-1799    1       kickoff tag at 250, home and away passes, fast balls, a ball gap,
                        the ball out of play past a touchline
    1800-1999   0       half-time (ungraded)
    2000-3799   2       the ball still on the centre spot until the kickoff tag at 2200,
                        play, then the ball still at a touchline until a throw-in tag at 2900
    3800-3999   0       after the match (ungraded)

Made-up numbers only. No real match data is read.
"""

import numpy as np

from falafelona.io import Event, Match

FPS = 25.0
N = 4000
PITCH = (105.0, 68.0)
CENTRE = (52.5, 34.0)
HALF_1 = (200, 1800)
HALF_2 = (2000, 3800)
KICKOFF_1, KICKOFF_2 = 250, 2200
GAP = (1000, 1060)                 # ball missing
OUT = (1500, 1520)                 # ball past the bottom touchline
STILL_AT_LINE = (2780, 2900)       # ball lying still at the touchline before the throw-in
THROW_IN = 2900
HOME_SLOTS, AWAY_SLOTS = range(11), range(23, 34)

# (frame, x, y): the ball moves in a straight line between waypoints. Long hops in few frames
# are passes and clearances (faster than the 12 m/s fast-ball threshold).
WAYPOINTS = [
    (0, *CENTRE), (KICKOFF_1, *CENTRE),
    (350, 62.0, 36.0), (390, 85.0, 50.0),          # home dribble, then a fast pass
    (560, 90.0, 45.0), (600, 60.0, 20.0),          # away wins it and clears it fast
    (800, 40.0, 25.0), (840, 15.0, 40.0),          # away dribble, fast pass forward
    (GAP[0], 10.0, 40.0), (GAP[1], 25.0, 32.0),    # (missing during the gap)
    (1300, 50.0, 30.0), (1480, 55.0, 3.0),         # home plays it towards the touchline
    (OUT[0], 55.0, -2.5), (OUT[1], 55.0, -2.5),    # out of play
    (1560, 55.0, 1.0), (1800, 70.0, 30.0),         # away throw-in and play
    (HALF_2[0], *CENTRE), (KICKOFF_2, *CENTRE),    # still on the centre spot until kickoff
    (2500, 40.0, 30.0), (2540, 70.0, 45.0),        # away dribble, then a fast pass lost to home
    (STILL_AT_LINE[0], 60.0, 0.5), (STILL_AT_LINE[1], 60.0, 0.5),   # out for a throw-in
    (3300, 30.0, 20.0), (3340, 60.0, 50.0),        # away play, fast pass to home
    (N - 1, 80.0, 40.0),
]
# Who has the ball: (first, stop, team). Outside these spells nobody is close to it.
SPELLS = [
    (KICKOFF_1, 560, "H"), (560, GAP[0], "A"), (GAP[1], OUT[0], "H"), (1560, HALF_1[1], "A"),
    (KICKOFF_2, 2540, "A"), (2540, STILL_AT_LINE[0], "H"), (THROW_IN, 3340, "A"),
    (3340, HALF_2[1], "H"),
]
# Stretches where play has stopped: the players barely move.
STOPPED = [(0, KICKOFF_1), OUT, (HALF_1[1], KICKOFF_2), STILL_AT_LINE]


def ball_track():
    frames, xs, ys = (np.array(column, dtype=float) for column in zip(*WAYPOINTS, strict=True))
    t = np.arange(N)
    xy = np.column_stack([np.interp(t, frames, xs), np.interp(t, frames, ys)])
    xy[GAP[0]:GAP[1]] = np.nan
    return xy


def player_tracks(ball, seed=0):
    """22 players swaying around their own spot; they freeze while play is stopped. The team in
    possession has a player 0.5 m from the ball, the nearest opponent is 6 m away."""
    rng = np.random.default_rng(seed)
    moving = np.ones(N)
    for first, stop in STOPPED:
        moving[first:stop] = 0.0
    clock = np.cumsum(moving) / FPS                  # stands still while play is stopped
    players = np.full((N, 46, 2), np.nan)
    slots = [*HOME_SLOTS, *AWAY_SLOTS]
    base = rng.uniform([15.0, 10.0], [90.0, 58.0], size=(len(slots), 2))
    phase = rng.uniform(0, 2 * np.pi, size=(len(slots), 2))
    sway = 4.0 * np.sin(0.5 * clock[:, None, None] + phase[None])   # about 2 m/s at most
    players[:, slots] = base[None] + sway

    for first, stop, team in SPELLS:
        own, other = (HOME_SLOTS, AWAY_SLOTS) if team == "H" else (AWAY_SLOTS, HOME_SLOTS)
        players[first:stop, own[0]] = ball[first:stop] + [0.4, 0.3]
        players[first:stop, other[0]] = ball[first:stop] + [5.0, 3.5]
    return players


def tags():
    def tag(start, type_, team, half):
        return Event(start=start, end=start + 10, code="SET PIECES", type=type_, team=team,
                     half=half)
    return [tag(KICKOFF_1, "Kick Off (Start)", "H", 1),
            tag(KICKOFF_2, "Kick Off (Start)", "A", 2),
            tag(THROW_IN, "Throw-in", None, 2)]


def make_match(with_second_kickoff=True):
    ball = ball_track()
    period = np.zeros(N, dtype=np.int8)
    period[HALF_1[0]:HALF_1[1]] = 1
    period[HALF_2[0]:HALF_2[1]] = 2
    events = tags()
    if not with_second_kickoff:
        events = [e for e in events if not (e.half == 2 and e.type == "Kick Off (Start)")]
    return Match(
        n_frames=N, fps=FPS, period=period, ball_xy=ball, ball_speed=np.full(N, np.nan),
        players_xy=player_tracks(ball), players_speed=np.full((N, 46), np.nan),
        slot_team=np.array(["H"] * 23 + ["A"] * 23), pitch=PITCH, tags=events,
    )
