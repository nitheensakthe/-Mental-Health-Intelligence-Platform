import os
import joblib
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

class CGNIVEXPredictor:
    def __init__(self):
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or vectorizer artifact missing. Run training first.")
            
        self.vectorizer = joblib.load(vectorizer_path)
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.model_name = model_data["name"]
        
    def predict_text(self, text: str):
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(vec)[0]
        
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}
            
        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_status": pred_label,
            "probabilities": probabilities
        }

    def predict_batch(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        df = df.copy()
        cleaned_texts = df[text_column].apply(clean_text)
        vecs = self.vectorizer.transform(cleaned_texts)
        df['predicted_status'] = self.model.predict(vecs)
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vecs)
            df['confidence'] = probs.max(axis=1)
            
        return df

if __name__ == "__main__":
    predictor = CGNIVEXPredictor()
    sample_text = "I feel completely overwhelmed by work pressure and can't sleep at all."
    res = predictor.predict_text(sample_text)
    print("Sample Prediction Result:")
    print(res)
