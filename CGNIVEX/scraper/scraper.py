import time
import os
import pandas as pd
import hashlib
from datetime import datetime
from config import SAMPLE_POSTS_PATH
from preprocessing.privacy import PrivacyScrubber

class IntelligentScraper:
    """
    Intelligent Web Scraping & Ingestion Layer for CGNIVEX.
    Features rate limiting, deduplication, incremental updates, privacy scrubbing,
    and a robust MOCK/SAMPLE data mode for offline execution.
    """

    def __init__(self, rate_limit_seconds: float = 1.0, mock_mode: bool = True):
        self.rate_limit = rate_limit_seconds
        self.mock_mode = mock_mode
        self.privacy = PrivacyScrubber()
        self.seen_post_hashes = set()

    def _hash_post(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()

    def fetch_data(self, source_path: str = None, limit: int = 50) -> pd.DataFrame:
        """
        Fetches data from mock sample dataset or custom source path with rate-limiting and deduplication.
        """
        print(f"[IntelligentScraper] Ingesting data (Mock Mode={self.mock_mode})...")

        if self.mock_mode:
            target_path = source_path if source_path and os.path.exists(source_path) else SAMPLE_POSTS_PATH
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Sample dataset file not found at: {target_path}")

            df = pd.read_csv(target_path)
        else:
            # Simulate real-world scraping with rate limiting
            time.sleep(self.rate_limit)
            if source_path and os.path.exists(source_path):
                df = pd.read_csv(source_path)
            else:
                df = pd.read_csv(SAMPLE_POSTS_PATH)

        # Deduplication & Privacy Scrubbing during ingestion
        ingested_rows = []
        for idx, row in df.iterrows():
            text = str(row.get("text", ""))
            post_hash = self._hash_post(text)

            if post_hash in self.seen_post_hashes:
                continue # Skip duplicate

            self.seen_post_hashes.add(post_hash)
            
            # Privacy scrub
            scrubbed_text = self.privacy.scrub_text(text)
            anonymized_user = self.privacy.anonymize_user_id(str(row.get("user_id", f"user_{idx}")))

            row_dict = row.to_dict()
            row_dict["text"] = scrubbed_text
            row_dict["user_id"] = anonymized_user
            if "timestamp" not in row_dict or pd.isna(row_dict["timestamp"]):
                row_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ingested_rows.append(row_dict)

            if len(ingested_rows) >= limit:
                break

            time.sleep(0.01) # Simulate minor delay per post

        result_df = pd.DataFrame(ingested_rows)
        print(f"[IntelligentScraper] Successfully ingested {len(result_df)} unique records.")
        return result_df
