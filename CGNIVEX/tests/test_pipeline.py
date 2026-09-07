import os
import sys
import pytest
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_DIR)

from src.data.preprocess import clean_text
from src.models.predict import CGNIVEXPredictor

def test_clean_text():
    raw_text = "Check out http://example.com @user #mentalhealth I'm SO stressed out!!!"
    cleaned = clean_text(raw_text)
    assert "http" not in cleaned
    assert "@user" not in cleaned
    assert "#" not in cleaned
    assert "im so stressed out" in cleaned

def test_predictor_single_text():
    predictor = CGNIVEXPredictor()
    result = predictor.predict_text("I am feeling very anxious about my upcoming exam.")
    assert "predicted_status" in result
    assert result["predicted_status"] in ['Normal', 'Depression', 'Anxiety', 'Stress', 'Suicidal', 'Bi-Polar', 'Personality Disorder']
    assert len(result["probabilities"]) == 7

def test_predictor_batch():
    predictor = CGNIVEXPredictor()
    df = pd.DataFrame({
        "text": [
            "Great workout session today!",
            "Deadlines piling up, so stressed out."
        ]
    })
    res_df = predictor.predict_batch(df)
    assert "predicted_status" in res_df.columns
    assert len(res_df) == 2
