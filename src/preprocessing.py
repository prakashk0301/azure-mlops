"""Data preprocessing and feature engineering module."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handle missing values in dataset."""
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Handling missing values with strategy: {strategy}")

    if strategy == "drop":
        return df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "forward_fill":
        df = df.fillna(method="ffill")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    logger.info(f"Encoding {len(categorical_cols)} categorical columns")

    for col in categorical_cols:
        if col == "customer_id":
            continue
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset."""
    logger.info("Creating engineered features")

    if "tenure" in df.columns and "monthly_charges" in df.columns:
        df["total_charges_per_month"] = (
            df.get("total_charges", df["monthly_charges"] * df["tenure"])
            / (df["tenure"] + 1)
        )
        df["high_value_customer"] = (
            df["monthly_charges"] > df["monthly_charges"].quantile(0.75)
        ).astype(int)

    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
        df["is_long_tenure"] = (df["tenure"] > 36).astype(int)

    return df


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler."""
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
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Main preprocessing pipeline."""
    logger.info("Starting data preprocessing...")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded data shape: {df.shape}")

    df = handle_missing_values(df, strategy="mean")
    df = encode_categorical_features(df)
    df = create_features(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y = df[target_col]
    X = df.drop(columns=[target_col, "customer_id"], errors="ignore")
    X_scaled, _ = scale_features(X)

    processed_df = X_scaled.copy()
    processed_df[target_col] = y
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Processed data shape: {processed_df.shape}")
    return X_scaled, y, output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        _, _, path = preprocess_data()
        print(f"Preprocessing completed. Output: {path}")
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        logger.warning("Please run data_ingestion.py first.")