"""Data preprocessing and feature engineering module."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in dataset.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'drop', or 'forward_fill'
        
    Returns:
        DataFrame with missing values handled
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found")
        return df
    
    logger.info(f"Handling missing values with strategy: {strategy}")
    
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "forward_fill":
        df = df.fillna(method="ffill")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical variables
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    
    logger.info(f"Encoding {len(categorical_cols)} categorical columns")
    
    for col in categorical_cols:
        if col != "customer_id":  # Don't encode ID
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new features
    """
    logger.info("Creating engineered features")
    
    # Example features - customize based on your data
    if "tenure" in df.columns and "monthly_charges" in df.columns:
        df["total_charges_per_month"] = df.get("total_charges", df["monthly_charges"] * df["tenure"]) / (df["tenure"] + 1)
        df["high_value_customer"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)
    
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
        df["is_long_tenure"] = (df["tenure"] > 36).astype(int)
    
    return df


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        X: Input features DataFrame
        
    Returns:
        Scaled features DataFrame and the scaler object
    """
    logger.info("Scaling numeric features")
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_scaled = X.copy()
    
    scaler = StandardScaler()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X_scaled, scaler


def preprocess_data(
    input_path: str = "data/raw/churn_data.csv",
    output_path: str = "data/processed/preprocessed_data.csv",
    target_col: str = "churn"
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Main preprocessing pipeline.
    
    Args:
        input_path: Path to raw data
        output_path: Path to save processed data
        target_col: Name of target column
        
    Returns:
        Tuple of (features, target, output_path)
    """
    logger.info("Starting data preprocessing...")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df, strategy="mean")
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Create new features
    df = create_features(df)
    
    # Separate features and target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col, "customer_id"], errors="ignore")
    else:
        X = df.drop(columns=["customer_id"], errors="ignore")
        y = None
    
    # Scale features
    X_scaled, _ = scale_features(X)
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed_df = X_scaled.copy()
    if y is not None:
        processed_df[target_col] = y
    
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Processed data shape: {processed_df.shape}")
    
    return X_scaled, y, output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        X, y, path = preprocess_data()
        print(f"Preprocessing completed. Output: {path}")
    except FileNotFoundError:
        logger.warning("Input data file not found. Run data_ingestion.py first.")
        "charge_per_month_tenure",
        "is_long_tenure",
        "num_services",
        "has_phone_service"
    ]
    
    X = df[feature_cols]
    y = df["churn"]
    
    return X, y, feature_cols