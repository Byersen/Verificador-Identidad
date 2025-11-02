import numpy as np
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import joblib

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


def main():
    X, y = load_data()
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    logging.info(f"Data split: {len(y_train)} train, {len(y_val)} validation")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logging.info("Features scaled with StandardScaler.")

    model = LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='liblinear'
    )
    logging.info("Training LogisticRegression...")
    model.fit(X_train_scaled, y_train)
    logging.info("Training completed.")

    try:
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_proba),
            "total_validation_samples": int(len(y_val)),
            "positive_samples_val": int(np.sum(y_val))
        }

        logging.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    except Exception as e:
        logging.warning(f"Could not compute validation metrics: {e}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    logging.info("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
