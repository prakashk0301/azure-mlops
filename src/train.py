"""Model training module."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target variable
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict
) -> RandomForestClassifier:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model hyperparameters
        
    Returns:
        Trained model
    """
    logger.info("Training Random Forest model...")
    
    # Extract hyperparameters from config
    hyperparams = model_config.get("hyperparameters", {})
    
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model


def save_model(model, output_path: str = "data/models/model.pkl") -> None:
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
    """
    Main training pipeline.
    
    Args:
        data_path: Path to preprocessed data
        config_path: Path to configuration file
        model_output_path: Path to save model
        target_col: Name of target column
        
    Returns:
        Trained model and training metrics
    """
    logger.info("Starting model training pipeline...")
    
    # Load config
    config = load_config(config_path)
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Prepare features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Train-test split
    training_config = config.get("training", {})
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y,
        test_size=config.get("data", {}).get("test_size", 0.2),
        random_state=config.get("data", {}).get("random_state", 42)
    )
    
    # Train model
    model = train_model(X_train, y_train, config.get("model", {}))
    
    # Save model
    save_model(model, model_output_path)
    
    # Log training info
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
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.warning("Please run preprocessing first: python src/preprocessing.py")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, model.predict(X_test))
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
        return model, auc