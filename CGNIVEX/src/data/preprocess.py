import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def clean_text(text: str) -> str:
    """Clean text by removing URLs, mentions, non-alphanumeric chars, and lowercasing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove mentions
    text = re.sub(r'#', '', text)                 # remove hashtag symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # keep only letters and space
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra whitespaces
    return text.lower()

def preprocess_and_split(input_csv: str = None):
    """
    Leak-free pipeline:
    1. Loads dataset
    2. Cleans text
    3. Stratified 80/10/10 train/val/test split BEFORE vectorization
    4. Fits TF-IDF vectorizer strictly on train set, transforms val/test
    5. Saves matrices, split dataframes, and vectorizer artifact
    """
    if input_csv is None:
        input_csv = os.path.join(DATA_DIR, "mental_health_dataset.csv")
        
    df = pd.read_csv(input_csv)
    print(f"Loaded dataset from '{input_csv}'. Shape: {df.shape}")
    
    # Cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 3].reset_index(drop=True)
    
    # Train / Val / Test Split (80% Train, 10% Val, 10% Test)
    X = df['cleaned_text']
    y = df['status']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Fit TF-IDF strictly on Train
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save artifacts
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    
    # Save split dataframes
    pd.DataFrame({'text': X_train, 'status': y_train}).to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    pd.DataFrame({'text': X_val, 'status': y_val}).to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    pd.DataFrame({'text': X_test, 'status': y_test}).to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    # Save vectorized feature matrices
    joblib.dump((X_train_vec, y_train), os.path.join(PROCESSED_DIR, "train_vec.pkl"))
    joblib.dump((X_val_vec, y_val), os.path.join(PROCESSED_DIR, "val_vec.pkl"))
    joblib.dump((X_test_vec, y_test), os.path.join(PROCESSED_DIR, "test_vec.pkl"))
    
    print("Preprocessing completed successfully with ZERO data leakage. Artifacts saved.")

if __name__ == "__main__":
    preprocess_and_split()
