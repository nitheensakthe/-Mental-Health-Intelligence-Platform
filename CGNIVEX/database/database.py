import sqlite3
import pandas as pd
import json
from config import DATABASE_PATH

class CGNIVEXDatabase:
    """
    SQLite Database persistence layer for CGNIVEX.
    Stores posts, predictions, topic distributions, emotions, trends, and model evaluation metrics.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes tables for posts, predictions, topics, emotions, trend_results, and model_metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Posts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TEXT,
                    platform TEXT,
                    raw_text TEXT,
                    cleaned_text TEXT,
                    topic TEXT,
                    label TEXT
                );
            """)

            # 2. Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT,
                    text TEXT,
                    predicted_sentiment TEXT,
                    predicted_emotion TEXT,
                    predicted_topic TEXT,
                    risk_indicator TEXT,
                    explanation JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Topics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_name TEXT,
                    frequency INTEGER,
                    keywords TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. Emotions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotions (
                    emotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion_name TEXT,
                    count INTEGER,
                    percentage REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 5. Trend Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_results (
                    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    topic TEXT,
                    emotion TEXT,
                    count INTEGER,
                    anomaly_flag INTEGER DEFAULT 0
                );
            """)

            # 6. Model Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    confusion_matrix TEXT,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.commit()

    def save_posts(self, df: pd.DataFrame):
        """Saves processed posts dataframe into the SQLite posts table."""
        if df.empty:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO posts (post_id, user_id, timestamp, platform, raw_text, cleaned_text, topic, label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get("post_id", "")),
                    str(row.get("user_id", "")),
                    str(row.get("timestamp", "")),
                    str(row.get("platform", "")),
                    str(row.get("text", "")),
                    str(row.get("cleaned_text", row.get("text", ""))),
                    str(row.get("topic", "")),
                    str(row.get("label", ""))
                ))
            conn.commit()

    def save_prediction(self, post_id: str, text: str, sentiment: str, emotion: str, topic: str, risk: str, explanation: dict):
        """Saves a prediction record with explanation metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (post_id, text, predicted_sentiment, predicted_emotion, predicted_topic, risk_indicator, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (post_id, text, sentiment, emotion, topic, risk, json.dumps(explanation)))
            conn.commit()

    def save_metrics(self, model_name: str, metrics: dict):
        """Saves evaluation metrics for baseline and proposed hybrid CGNIVEX models."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score, confusion_matrix)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1_score", 0.0),
                json.dumps(metrics.get("confusion_matrix", []))
            ))
            conn.commit()

    @staticmethod
    def get_all_posts() -> pd.DataFrame:
        """Retrieves all posts stored in SQLite."""
        import sqlite3
        with sqlite3.connect(DATABASE_PATH) as conn:
            return pd.read_sql("SELECT * FROM posts", conn)

    @staticmethod
    def get_predictions() -> pd.DataFrame:
        """Retrieves recent predictions."""
        import sqlite3
        with sqlite3.connect(DATABASE_PATH) as conn:
            return pd.read_sql("SELECT * FROM predictions ORDER BY prediction_id DESC", conn)

    @staticmethod
    def get_metrics() -> pd.DataFrame:
        """Retrieves stored model metrics."""
        import sqlite3
        with sqlite3.connect(DATABASE_PATH) as conn:
            return pd.read_sql("SELECT * FROM model_metrics ORDER BY metric_id DESC", conn)

