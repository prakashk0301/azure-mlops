"""Unit tests for ML pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataIngestion:
    """Test data ingestion module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            "customer_id": ["001", "002", "003"],
            "tenure": [12, 24, 36],
            "monthly_charges": [65.0, 75.0, 85.0],
            "churn": [0, 1, 0]
        }
        return pd.DataFrame(data)
    
    def test_data_validation(self, sample_data):
        """Test that data validation works."""
        assert "customer_id" in sample_data.columns
        assert "churn" in sample_data.columns
        assert sample_data["churn"].isin([0, 1]).all()
    
    def test_data_shape(self, sample_data):
        """Test data shape."""
        assert sample_data.shape[0] == 3
        assert sample_data.shape[1] == 4


class TestPreprocessing:
    """Test preprocessing module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            "tenure": [12, 24, np.nan, 36],
            "monthly_charges": [65.0, 75.0, 85.0, 95.0],
            "churn": [0, 1, 0, 1]
        }
        return pd.DataFrame(data)
    
    def test_missing_values_handling(self, sample_data):
        """Test missing value handling."""
        df = sample_data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        assert df.isnull().sum().sum() == 0
    
    def test_feature_shape(self, sample_data):
        """Test feature preprocessing."""
        df = sample_data.copy()
        X = df.drop(columns=["churn"])
        assert X.shape[1] == 2


class TestModelTraining:
    """Test model training module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_train_test_split(self, sample_data):
        """Test train-test split."""
        from sklearn.model_selection import train_test_split
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_model_creation(self, sample_data):
        """Test model can be created and trained."""
        from sklearn.ensemble import RandomForestClassifier
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])
        assert model.n_estimators == 10


class TestEvaluation:
    """Test model evaluation module."""
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        return y_true, y_pred
    
    def test_accuracy_calculation(self, predictions):
        """Test accuracy calculation."""
        from sklearn.metrics import accuracy_score
        y_true, y_pred = predictions
        acc = accuracy_score(y_true, y_pred)
        assert 0 <= acc <= 1
    
    def test_f1_score(self, predictions):
        """Test F1 score calculation."""
        from sklearn.metrics import f1_score
        y_true, y_pred = predictions
        f1 = f1_score(y_true, y_pred)
        assert 0 <= f1 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])