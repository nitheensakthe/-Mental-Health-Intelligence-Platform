import pandas as pd
from .privacy import PrivacyScrubber
from .language import LanguageDetector
from .tokenizer import TextTokenizer

class PreprocessingPipeline:
    """
    Unified Preprocessing Pipeline for CGNIVEX.
    Used seamlessly by both batch training and real-time inference.
    """

    def __init__(self):
        self.privacy = PrivacyScrubber()
        self.language = LanguageDetector()
        self.tokenizer = TextTokenizer()

    def process_text(self, text: str) -> dict:
        """
        Runs full preprocessing sequence on a single raw text string.
        """
        # 1. Scrub PII
        scrubbed_text = self.privacy.scrub_text(text)

        # 2. Check Language
        lang = self.language.detect_language(scrubbed_text)

        # 3. Tokenize and Normalize
        cleaned_text = self.tokenizer.process(scrubbed_text)
        tokens = cleaned_text.split()

        return {
            "raw_text": text,
            "scrubbed_text": scrubbed_text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "language": lang
        }

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "text", user_id_col: str = "user_id") -> tuple:
        """
        Processes a pandas DataFrame, removing duplicates, missing values,
        scrubbing PII, and generating clean tokenized columns.
        Returns (processed_df, quality_stats_dict).
        """
        stats = {
            "total_records": len(df),
            "missing_text_count": 0,
            "duplicate_count": 0,
            "valid_records": 0
        }

        if df.empty or text_col not in df.columns:
            return df, stats

        # Copy to avoid mutating original
        data = df.copy()

        # 1. Missing text check
        stats["missing_text_count"] = int(data[text_col].isna().sum())
        data = data.dropna(subset=[text_col])

        # 2. Duplicate check
        stats["duplicate_count"] = int(data.duplicated(subset=[text_col]).sum())
        data = data.drop_duplicates(subset=[text_col])

        # 3. Privacy & Text Cleaning
        if user_id_col in data.columns:
            data[user_id_col] = data[user_id_col].apply(self.privacy.anonymize_user_id)

        processed_results = data[text_col].apply(self.process_text)
        data["scrubbed_text"] = [r["scrubbed_text"] for r in processed_results]
        data["cleaned_text"] = [r["cleaned_text"] for r in processed_results]
        data["language"] = [r["language"] for r in processed_results]

        # Filter out empty cleaned text
        data = data[data["cleaned_text"].str.strip() != ""]
        stats["valid_records"] = len(data)

        return data, stats
