"""Data ingestion module for loading and validating customer churn data."""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(path: str) -> pd.DataFrame:
    """
    Load customer churn dataset from CSV.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with customer data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded data shape: {df.shape}")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data schema and quality.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ["customer_id", "churn"]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check target variable
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Target 'churn' must be binary (0 or 1)")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    logger.info("Data validation passed")
    return True


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw/churn_data.csv") -> None:
    """Save raw data to output directory."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to {output_path}")


def ingest_data(input_path: str, output_path: str = "data/raw/churn_data.csv") -> pd.DataFrame:
    """
    Main data ingestion pipeline.
    
    Args:
        input_path: Path to source data
        output_path: Path to save raw data
        
    Returns:
        Validated DataFrame
    """
    logger.info("Starting data ingestion...")
    
    # Load data
    df = load_data(input_path)
    
    # Validate data
    validate_data(df)
    
    # Save raw data
    save_raw_data(df, output_path)
    
    logger.info("Data ingestion completed successfully")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage - update path as needed
    try:
        ingest_data("data/raw/input_data.csv")
    except FileNotFoundError:
        logger.warning("Input data file not found. Please place data in data/raw/input_data.csv")
    
    print(f"✓ Loaded {len(df)} rows, {df.churn.mean():.2%} churn rate")
    return df