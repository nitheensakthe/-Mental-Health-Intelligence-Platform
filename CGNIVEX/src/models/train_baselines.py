import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def train_and_benchmark():
    """Trains multiple classifiers on train set, evaluates on validation set, and saves the best model."""
    print("Loading processed feature matrices...")
    X_train, y_train = joblib.load(os.path.join(PROCESSED_DIR, "train_vec.pkl"))
    X_val, y_val = joblib.load(os.path.join(PROCESSED_DIR, "val_vec.pkl"))
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "MultinomialNB": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_model_name = ""
    best_f1 = -1.0
    
    print("\n=== BENCHMARKING MODELS ===")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average='macro')
        weighted_f1 = f1_score(y_val, preds, average='weighted')
        
        results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        }
        print(f" -> {name} | Val Accuracy: {acc:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model
            best_model_name = name
            
    print(f"\n[BEST MODEL] {best_model_name} with Val Macro F1 of {best_f1:.4f}")
    
    # Save best model artifact
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({"model": best_model, "name": best_model_name}, best_model_path)
    
    # Save benchmark metrics
    metrics_path = os.path.join(MODELS_DIR, "benchmark_results.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved best model to '{best_model_path}' and metrics to '{metrics_path}'.")

if __name__ == "__main__":
    train_and_benchmark()
