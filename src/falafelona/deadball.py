# ws:dead: score how likely each frame is dead ball.
#
# Output: score in 0..1 per frame, fed to the decoder in possession.py (docs/decisions.md D13).
#
# Ideas:
# - ball out of bounds (pitch is 105.3 x 68 m)
# - players static / slow for a while
# - Smart Tagging set pieces and restarts (evidence, not truth; throw-ins have team N/A)
# - official baseline: dead wherever the ball is not observed. Start here.
# - second half: graded before its kickoff, and those frames are dead. Detect the kickoff from
#   the Kick Off (Start) Smart Tag, never a hard-coded frame.
#
# Methods are selected via the `deadball:` section of the config.
