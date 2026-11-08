"""Public API for macula center localization in retinal angiography images."""
from bs.white_spot_predictor import PointPredictor

MaculaLocalizer = PointPredictor

__all__ = ["MaculaLocalizer"]
