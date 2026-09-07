class LanguageDetector:
    """
    Language detection module to identify language of input text.
    Provides robust fallback handling for non-English or corrupted text.
    """

    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang

    def detect_language(self, text: str) -> str:
        """
        Detects language. In lightweight/prototype mode, uses simple character distribution
        and common English stopword checking.
        """
        if not text or not isinstance(text, str):
            return "unknown"

        english_stopwords = {"the", "and", "is", "in", "to", "of", "it", "you", "that", "was", "for", "on", "are", "with", "as", "I", "my", "feel", "stressed"}
        words = set(text.lower().split())

        # If any common English word is present, classify as English
        if words.intersection(english_stopwords) or any(c.isalpha() and ord(c) < 128 for c in text):
            return "en"
        
        return "non_en"

    def is_target_language(self, text: str) -> bool:
        """Checks if text matches the target language (default 'en')."""
        return self.detect_language(text) == self.target_lang
