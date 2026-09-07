import pandas as pd
from config import EMOTION_LABELS

class EmotionAnalyzer:
    """
    Emotion distribution analysis engine supporting multi-class categories:
    Positive, Neutral, Sadness, Anxiety/Stress, Anger, Fear, Joy.
    """

    @staticmethod
    def analyze_emotions(df: pd.DataFrame, label_col: str = "label") -> dict:
        """Calculates emotion distribution counts and percentages."""
        if df.empty or label_col not in df.columns:
            return {e: 0 for e in EMOTION_LABELS}

        counts = df[label_col].value_counts().to_dict()
        total = len(df)
        
        result = {}
        for emotion in EMOTION_LABELS:
            cnt = counts.get(emotion, 0)
            pct = round((cnt / total) * 100, 2) if total > 0 else 0.0
            result[emotion] = {
                "count": cnt,
                "percentage": pct
            }

        return result
