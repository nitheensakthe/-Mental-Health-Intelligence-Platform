import os

# Prevent OpenBLAS / MKL memory allocation errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_POSTS_PATH = os.path.join(DATA_DIR, "sample_posts.csv")

MODELS_SAVED_DIR = os.path.join(BASE_DIR, "models_saved")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "cgnivex.db")

# Ensure required directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_SAVED_DIR, os.path.dirname(DATABASE_PATH)]:
    os.makedirs(path, exist_ok=True)

# System Constants
MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 128
GNN_EMBEDDING_DIM = 64
SEED = 42

# Supported Emotion Categories
EMOTION_LABELS = [
    "Positive",
    "Neutral",
    "Sadness",
    "Anxiety/Stress",
    "Anger",
    "Fear",
    "Joy"
]

# Supported Topic Categories
TOPIC_LABELS = [
    "Academic pressure",
    "Work pressure",
    "Anxiety",
    "Depression",
    "Loneliness",
    "Relationships",
    "Sleep",
    "General wellbeing"
]

# Risk Levels
RISK_LEVELS = ["Low", "Moderate", "High"]

# Research Disclaimer
DISCLAIMER = (
    "DISCLAIMER: CGNIVEX is a research-oriented data analytics and machine learning prototype. "
    "It provides a Mental Health Risk Indicator based on public community discussion trends. "
    "This system is NOT a medical tool and does NOT provide medical or clinical diagnoses."
)
