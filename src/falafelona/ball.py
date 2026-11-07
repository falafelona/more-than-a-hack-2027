# ws:ball: clean the ball track.
#
# Input: raw ball positions (NaN where missing: ~41% of graded frames).
# Output: cleaned ball track, same shape, NaN where still unknown.
#
# Ideas:
# - fill short gaps (linear interpolation, Kalman filter / smoother)
# - remove false detections (impossible speeds, jumps, ball far from all players)
# - keep long gaps as unknown, since they often mean dead ball
#
# Methods are selected and tuned via the `ball:` section of the config.

from falafelona.io import Match


class DataCleaner:
    """Cleans the ball track of a Match (PR #9 structure).

    For now there is no cleaning: the cleaned data is the raw Match, unchanged (method `none`,
    task 2.2). Abdallah replaces this with the real methods.
    """

    def __init__(self, raw_data: Match):
        self.raw_data = raw_data
        self.clean_data = raw_data

    def return_cleaned_data(self):
        return self.clean_data
