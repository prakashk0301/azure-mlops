"""Model evaluation and reporting module."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)


def load_model(model_path: str = "data/models/model.pkl"):
    """Load trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate classification report."""
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification Report:\n{report}")
    return report


def save_metrics(metrics: dict, output_path: str = "data/models/metrics.json") -> None:
    """Save metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {output_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = "data/models/confusion_matrix.png") -> None:
    """Plot and save confusion matrix."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str = "data/models/roc_curve.png") -> None:
    """Plot and save ROC curve."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def evaluate_model(
    model_path: str = "data/models/model.pkl",
    data_path: str = "data/processed/preprocessed_data.csv",
    target_col: str = "churn",
    metrics_output: str = "data/models/metrics.json"
) -> dict:
    """Main evaluation pipeline."""
    logger.info("Starting model evaluation pipeline...")
    model = load_model(model_path)
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = calculate_metrics(y, y_pred, y_pred_proba)
    logger.info(f"Evaluation Metrics: {metrics}")
    generate_classification_report(y, y_pred)
    save_metrics(metrics, metrics_output)
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_pred_proba)
    logger.info("Model evaluation pipeline completed")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        metrics = evaluate_model()
        print(f"Evaluation completed. Metrics: {metrics}")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.warning("Please train model first: python src/train.py")