# ws:eval: check a submission file and print AGGREGATE metrics.
#
#   python scripts/bootstrap_eval.py outputs/Falafelona.csv [--bootstrap 1000]
#
# Will:
# - score against the public 5-minute GT with vendor possession_metrics.evaluate()
#   as a PIPELINE CHECK ONLY, never to tune or choose between versions (D9)
# - report label-free stats for the whole match: number of possessions, mean/median duration,
#   share of H / A / D / X, flicker (possessions shorter than 1 s)
#
# We have no full-match ground truth, so no full-match score is possible before results day.
# Versions are compared on other matches (D15).
