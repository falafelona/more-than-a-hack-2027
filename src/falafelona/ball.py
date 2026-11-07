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

from io import Match

class DataCleaner:
    def __init__(self, raw_data: Match):
        self.raw_data = raw_data
        self. clean_data = None

    def return_cleaned_data(self):
        return self.clean_data