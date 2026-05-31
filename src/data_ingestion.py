"""Data ingestion module for loading and validating customer churn data."""

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """Load customer churn dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded data shape: {df.shape}")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate data schema and quality."""
    required_columns = ["customer_id", "churn"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Target 'churn' must be binary (0 or 1)")

    if df.empty:
        raise ValueError("DataFrame is empty")

    logger.info("Data validation passed")
    return True


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw/churn_data.csv") -> None:
    """Save raw data to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to {output_path}")


def ingest_data(input_path: str, output_path: str = "data/raw/churn_data.csv") -> pd.DataFrame:
    """Run data ingestion, validation, and raw save."""
    logger.info("Starting data ingestion...")

    df = load_data(input_path)
    validate_data(df)
    save_raw_data(df, output_path)

    logger.info("Data ingestion completed successfully")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        ingest_data("data/raw/input_data.csv")
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        logger.warning("Please place input data at data/raw/input_data.csv")
