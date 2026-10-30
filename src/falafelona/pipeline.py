# ws:int: glue the stages together and write the submission file.
#
#   load (io) -> clean ball (ball) -> dead-ball score (deadball) -> H/A/D labels (possession)
#   -> X on ungraded frames (official games.half_video_blocks; docs/decisions.md D14)
#   -> outputs/Falafelona.csv: header "frame,label", exactly 199,319 rows, frames 0..199318
#
# Must be deterministic: fixed seeds from the config, no order-dependent parallelism.
# Will also compute the SHA-256 of the written file and aggregate label counts (safe to print).
