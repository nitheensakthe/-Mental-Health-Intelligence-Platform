import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def evaluate_test_set():
    """Evaluates the best model on the holdout test set."""
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    test_vec_path = os.path.join(PROCESSED_DIR, "test_vec.pkl")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model file not found. Train models first.")
        
    model_data = joblib.load(best_model_path)
    model = model_data["model"]
    model_name = model_data["name"]
    
    X_test, y_test = joblib.load(test_vec_path)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    weighted_f1 = f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, output_dict=True)
    conf_mat = confusion_matrix(y_test, preds).tolist()
    labels = sorted(list(set(y_test)))
    
    eval_results = {
        "model_name": model_name,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "labels": labels,
        "confusion_matrix": conf_mat,
        "classification_report": report
    }
    
    output_path = os.path.join(MODELS_DIR, "test_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"\n================ TEST SET EVALUATION ================")
    print(f"Model: {model_name}")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print(f"Full report saved to '{output_path}'.")

if __name__ == "__main__":
    evaluate_test_set()
