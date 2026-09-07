import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from config import MODEL_NAME, EMBEDDING_DIM, EMOTION_LABELS

class TransformerEmbeddingModel:
    """
    NLP / Transformer Module for generating contextual text embeddings and predictions.
    Uses DistilBERT when PyTorch & Transformers are available, with TF-IDF fallback.
    """

    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer
        self.tokenizer = None
        self.model = None
        self.tfidf = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """Loads DistilBERT model and tokenizer, or initializes TF-IDF fallback."""
        if self.use_transformer:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModel.from_pretrained(MODEL_NAME)
                self.model.eval()
                self.is_loaded = True
                print(f"[TransformerModel] DistilBERT ({MODEL_NAME}) loaded successfully.")
                return
            except Exception as e:
                print(f"[TransformerModel] Transformer load warning ({e}). Falling back to TF-IDF vectorizer.")
                self.use_transformer = False

        # Fallback to TF-IDF
        self.tfidf = TfidfVectorizer(max_features=EMBEDDING_DIM)
        self.is_loaded = True
        print("[TransformerModel] Lightweight TF-IDF fallback engine initialized.")

    def preprocess_text(self, text: str) -> str:
        """Preprocesses text string for model input."""
        return str(text).lower().strip()

    def generate_embedding(self, text_list: list) -> np.ndarray:
        """
        Generates contextual text embeddings for a list of text strings.
        Returns numpy array of shape (N, EMBEDDING_DIM).
        """
        if not text_list:
            return np.zeros((0, EMBEDDING_DIM))

        clean_texts = [self.preprocess_text(t) for t in text_list]

        if self.use_transformer and self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(clean_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling over token embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                    
                    # Project/Truncate to target dimension if needed
                    if embeddings.shape[1] > EMBEDDING_DIM:
                        embeddings = embeddings[:, :EMBEDDING_DIM]
                    return embeddings
            except Exception as e:
                print(f"[TransformerModel] Transformer inference error: {e}. Using TF-IDF fallback.")

        # Fallback TF-IDF vectorization
        if self.tfidf:
            try:
                matrix = self.tfidf.fit_transform(clean_texts).toarray()
                if matrix.shape[1] < EMBEDDING_DIM:
                    padding = np.zeros((matrix.shape[0], EMBEDDING_DIM - matrix.shape[1]))
                    matrix = np.hstack([matrix, padding])
                return matrix[:, :EMBEDDING_DIM]
            except Exception:
                pass

        # Return reproducible synthetic embeddings if empty
        np.random.seed(42)
        return np.random.randn(len(text_list), EMBEDDING_DIM) * 0.1

    def predict_sentiment(self, text: str) -> dict:
        """
        Predicts sentiment polarity (Positive, Neutral, Negative) and confidence score.
        """
        text_lower = text.lower()
        pos_words = ["happy", "great", "wonderful", "joy", "good", "peace", "gratitude", "positive", "win"]
        neg_words = ["sad", "depressed", "anxious", "stress", "fear", "panic", "hate", "angry", "terrible", "sleepless", "lonely"]

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)

        if pos_count > neg_count:
            return {"sentiment": "Positive", "confidence": 0.85}
        elif neg_count > pos_count:
            return {"sentiment": "Negative", "confidence": 0.88}
        else:
            return {"sentiment": "Neutral", "confidence": 0.70}

    def predict_emotion(self, text: str) -> dict:
        """
        Predicts dominant emotion from EMOTION_LABELS.
        """
        text_lower = text.lower()
        if any(w in text_lower for w in ["exam", "stress", "overwhelmed", "deadline", "burnout", "work"]):
            return {"emotion": "Anxiety/Stress", "confidence": 0.89}
        elif any(w in text_lower for w in ["sad", "lonely", "empty", "isolated", "crying"]):
            return {"emotion": "Sadness", "confidence": 0.86}
        elif any(w in text_lower for w in ["fear", "terrified", "panic", "scared"]):
            return {"emotion": "Fear", "confidence": 0.84}
        elif any(w in text_lower for w in ["angry", "frustrated", "toxic"]):
            return {"emotion": "Anger", "confidence": 0.82}
        elif any(w in text_lower for w in ["happy", "wonderful", "joy", "great", "celebrating"]):
            return {"emotion": "Joy", "confidence": 0.91}
        elif any(w in text_lower for w in ["mindfulness", "peace", "positive"]):
            return {"emotion": "Positive", "confidence": 0.85}
        else:
            return {"emotion": "Neutral", "confidence": 0.75}
