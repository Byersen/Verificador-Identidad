import numpy as np
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/"))
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_DATA_PATH", "data/embeddings.npy"))
LABELS_FILE = Path(os.getenv("LABELS_DATA_PATH", "data/labels.csv"))
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def load_data():
    """Load embeddings and labels from disk."""
    if not EMBEDDINGS_FILE.exists() or not LABELS_FILE.exists():
        logging.error("Data files not found. Please run 'scripts/embeddings.py' first.")
        return None, None

    X = np.load(EMBEDDINGS_FILE)
    y = np.loadtxt(LABELS_FILE, delimiter=',')
    logging.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y
