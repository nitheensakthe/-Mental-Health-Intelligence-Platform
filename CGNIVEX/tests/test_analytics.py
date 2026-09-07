import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.sentiment import SentimentAnalyzer
from analytics.risk_score import RiskScoreCalculator
from analytics.anomaly import AnomalyDetector

class TestAnalytics(unittest.TestCase):

    def test_sentiment_analyzer(self):
        df = pd.DataFrame([{"label": "Joy"}, {"label": "Anxiety/Stress"}, {"label": "Neutral"}])
        res = SentimentAnalyzer.analyze_dataframe(df)
        self.assertEqual(res["counts"]["Positive"], 1)
        self.assertEqual(res["counts"]["Negative"], 1)
        self.assertEqual(res["counts"]["Neutral"], 1)

    def test_risk_score_calculator(self):
        res = RiskScoreCalculator.calculate_risk_indicator("Negative", "Anxiety/Stress", "Academic pressure", "exam panic helpless")
        self.assertIn(res["risk_indicator"], ["Moderate", "High"])
        self.assertGreater(res["risk_score"], 5.0)

    def test_anomaly_detector(self):
        trend_df = pd.DataFrame({
            "stress_signal_count": [5, 6, 5, 25, 6, 4]
        })
        detector = AnomalyDetector(z_threshold=1.5, window=3)
        res = detector.detect_spikes(trend_df)
        self.assertGreaterEqual(res["is_anomaly"].sum(), 1)

if __name__ == "__main__":
    unittest.main()
