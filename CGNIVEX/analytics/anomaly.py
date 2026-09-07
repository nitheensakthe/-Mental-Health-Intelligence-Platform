import pandas as pd
import numpy as np

class AnomalyDetector:
    """
    Anomaly Detection engine using Z-score and Moving Average Deviation
    to identify unusual surges or spikes in mental-health-related discussions.
    """

    def __init__(self, z_threshold: float = 1.5, window: int = 3):
        self.z_threshold = z_threshold
        self.window = window

    def detect_spikes(self, trend_df: pd.DataFrame, value_col: str = "stress_signal_count") -> pd.DataFrame:
        """
        Detects anomalies in time-series trend data using Z-score and rolling mean.
        Returns trend_df with added columns: 'z_score', 'rolling_avg', and 'is_anomaly'.
        """
        if trend_df.empty or value_col not in trend_df.columns:
            return trend_df

        df = trend_df.copy()
        values = df[value_col].astype(float).values

        # Rolling statistics
        df["rolling_avg"] = df[value_col].rolling(window=self.window, min_periods=1).mean()
        df["rolling_std"] = df[value_col].rolling(window=self.window, min_periods=1).std().fillna(1.0)
        
        # Z-score computation
        mean_val = np.mean(values) if len(values) > 0 else 0.0
        std_val = np.std(values) if len(values) > 0 and np.std(values) > 0 else 1.0
        
        df["z_score"] = (df[value_col] - mean_val) / std_val
        df["is_anomaly"] = df["z_score"] >= self.z_threshold

        anomaly_count = df["is_anomaly"].sum()
        print(f"[AnomalyDetector] Identified {anomaly_count} anomaly spikes in mental health discussion volume.")

        return df
