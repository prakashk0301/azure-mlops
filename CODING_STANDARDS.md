# MLOps Project Python Coding Standards

## Code Style

### Formatting
- Use **black** for code formatting: `black src/`
- Line length: 88 characters (black default)
- Use 4 spaces for indentation

### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Type Hints
Always use type hints for functions:

```python
def process_data(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, list]:
    """Process data with specified threshold."""
    pass
```

## Documentation

### Docstrings
Use Google-style docstrings:

```python
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a random forest model.
    
    Args:
        X_train: Training features with shape (n_samples, n_features)
        y_train: Training target with shape (n_samples,)
        
    Returns:
        Trained RandomForestClassifier model
        
    Raises:
        ValueError: If input data is invalid
    """
    pass
```

### Comments
- Use comments for "why", not "what"
- Keep comments short and relevant
- Update comments when code changes

## Testing

### Test Structure
```python
class TestModuleName:
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return data
    
    def test_function_case(self, sample_data):
        """Test description."""
        assert result == expected
```

### Coverage
- Aim for 80%+ code coverage
- Test edge cases and error conditions
- Run tests: `pytest tests/ -v --cov=src`

## Import Organization

```python
# Standard library
import os
import logging
from pathlib import Path

# Third-party
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Local imports
from src.utils import helper_function
```

Use `ruff` to check imports: `ruff check src/`

## Error Handling

```python
def load_data(path: str) -> pd.DataFrame:
    """Load data with proper error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise
    
    return df
```

## Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data with logging."""
    logger.info(f"Processing data with shape: {df.shape}")
    
    try:
        result = df.dropna()
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    
    logger.info("Processing completed successfully")
    return result
```

## Configuration

- Use YAML for configuration files
- Use environment variables for sensitive data
- Document all configuration options

## Git Workflow

1. Create feature branch: `git checkout -b feature/name`
2. Make changes following style guidelines
3. Run tests: `pytest tests/ -v`
4. Format code: `black src/`
5. Lint code: `ruff check src/`
6. Commit with clear message: `git commit -m "feat: description"`
7. Push: `git push origin feature/name`
8. Create pull request

## Pre-commit Checklist

- [ ] Code formatted with black
- [ ] Tests passing
- [ ] Linting passed with ruff
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Comments updated
- [ ] No hardcoded values
- [ ] Proper error handling
