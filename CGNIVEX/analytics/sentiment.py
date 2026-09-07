import pandas as pd

class SentimentAnalyzer:
    """
    Analyzes sentiment distributions (Positive, Neutral, Negative) across dataset.
    """

    @staticmethod
    def analyze_dataframe(df: pd.DataFrame, label_col: str = "label") -> dict:
        """Computes sentiment breakdown counts and percentages."""
        if df.empty:
            return {"counts": {}, "percentages": {}}

        # Map emotions to sentiments
        sentiment_map = {
            "Joy": "Positive",
            "Positive": "Positive",
            "Neutral": "Neutral",
            "Sadness": "Negative",
            "Anxiety/Stress": "Negative",
            "Fear": "Negative",
            "Anger": "Negative"
        }

        sentiments = df[label_col].map(lambda x: sentiment_map.get(str(x), "Neutral"))
        counts = sentiments.value_counts().to_dict()
        total = len(sentiments)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        return {
            "counts": counts,
            "percentages": percentages
        }
