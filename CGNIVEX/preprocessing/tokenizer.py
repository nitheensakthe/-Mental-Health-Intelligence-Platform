import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources gracefully
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

SLANG_DICTIONARY = {
    "u": "you",
    "r": "are",
    "cant": "cannot",
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "dont": "do not",
    "don't": "do not",
    "im": "i am",
    "i'm": "i am",
    "idk": "i do not know",
    "tbh": "to be honest",
    "btw": "by the way",
    "af": "extremely",
    "fyi": "for your information"
}

EMOJI_DICTIONARY = {
    "😊": " happy ",
    "😃": " joyful ",
    "😭": " crying sad ",
    "😢": " sad ",
    "😔": " depressed ",
    "😡": " angry ",
    "😠": " annoyed ",
    "😱": " fearful scared ",
    "😴": " tired sleepy ",
    "💔": " heartbroken "
}

class TextTokenizer:
    """
    Tokenizer and text normalizer engine for CGNIVEX.
    Handles emojis, slang dictionary, stop-words, and lemmatization.
    """

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def normalize_emojis_and_slang(self, text: str) -> str:
        """Translates emojis and common slang words to standard English terms."""
        for emoji, word in EMOJI_DICTIONARY.items():
            text = text.replace(emoji, word)

        words = text.split()
        normalized_words = [SLANG_DICTIONARY.get(w.lower(), w) for w in words]
        return " ".join(normalized_words)

    def clean_text(self, text: str) -> str:
        """Performs noise filtering and character cleaning."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> list:
        """
        Tokenizes text into words, removes stop-words, and applies lemmatization.
        Returns a list of clean tokens.
        """
        normalized = self.normalize_emojis_and_slang(text)
        cleaned = self.clean_text(normalized)
        tokens = cleaned.split()

        filtered_tokens = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 1:
                if self.lemmatizer:
                    try:
                        t = self.lemmatizer.lemmatize(t)
                    except Exception:
                        pass
                filtered_tokens.append(t)

        return filtered_tokens

    def process(self, text: str) -> str:
        """Returns clean tokenized text as a joined string."""
        tokens = self.tokenize_and_lemmatize(text)
        return " ".join(tokens)
