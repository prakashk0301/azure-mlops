"""Model training module."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={test_size}")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_config: dict
) -> RandomForestClassifier:
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    hyperparams = model_config.get("hyperparameters", {})
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def save_model(model: RandomForestClassifier, output_path: str = "data/models/model.pkl") -> None:
    """Save trained model to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {output_path}")


def train_pipeline(
    data_path: str = "data/processed/preprocessed_data.csv",
    config_path: str = "configs/config.yaml",
    model_output_path: str = "data/models/model.pkl",
    target_col: str = "churn"
) -> Tuple[RandomForestClassifier, dict]:
    """Main training pipeline."""
    logger.info("Starting model training pipeline...")

    config = load_config(config_path)
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X,
        y,
        test_size=config.get("data", {}).get("test_size", 0.2),
        random_state=config.get("data", {}).get("random_state", 42)
    )

    model = train_model(X_train, y_train, config.get("model", {}))
    save_model(model, model_output_path)

    train_metrics = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.shape[1],
        "target_distribution": y.value_counts().to_dict()
    }
    logger.info(f"Training metrics: {train_metrics}")
    logger.info("Model training pipeline completed")
    return model, train_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        model, metrics = train_pipeline()
        print(f"Training completed. Metrics: {metrics}")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.warning("Please run preprocessing first: python src/preprocessing.py")