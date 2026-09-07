import pandas as pd

class TrendAnalyzer:
    """
    Temporal Trend Analysis engine for CGNIVEX.
    Aggregates posts across Daily, Weekly, and Monthly timeframes to track:
    - Post volume trends
    - Topic frequency progression
    - Emotion and sentiment shifts over time
    """

    @staticmethod
    def analyze_temporal_trends(df: pd.DataFrame, timestamp_col: str = "timestamp", freq: str = "D") -> pd.DataFrame:
        """
        Aggregates post volume, dominant emotions, and stress signals over time.
        freq options: 'D' (Daily), 'W' (Weekly), 'M' (Monthly).
        """
        if df.empty or timestamp_col not in df.columns:
            return pd.DataFrame()

        data = df.copy()
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        data = data.sort_values(timestamp_col)

        # Set datetime index for resampling
        data = data.set_index(timestamp_col)

        # Resample post count
        resampled = data.resample(freq).size().to_frame(name="post_count")

        # Count stress/anxiety posts
        if "label" in data.columns:
            stress_mask = data["label"].isin(["Anxiety/Stress", "Sadness", "Fear"])
            resampled["stress_signal_count"] = data[stress_mask].resample(freq).size()
            resampled["stress_signal_count"] = resampled["stress_signal_count"].fillna(0).astype(int)
        else:
            resampled["stress_signal_count"] = 0

        resampled = resampled.reset_index()
        resampled["date_str"] = resampled[timestamp_col].dt.strftime("%Y-%m-%d")
        return resampled
