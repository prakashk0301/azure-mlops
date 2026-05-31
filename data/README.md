# Data Directory

## Structure

```
data/
├── raw/              # Original, unmodified data
├── processed/        # Cleaned and processed data
├── features/         # Feature engineered data
└── models/           # Trained model artifacts
```

## Data Sources

- **raw/**: Place your raw customer churn dataset here
  - Expected format: CSV with columns including:
    - Customer demographics (age, tenure, location)
    - Service usage (monthly charges, total charges)
    - Service types (internet, phone, streaming)
    - Churn status (target variable)

## Data Processing Pipeline

1. **Raw Data**: Ingest data from source
2. **Preprocessing**: Handle missing values, scaling, encoding
3. **Feature Engineering**: Create new features for better model performance
4. **Train/Test Split**: Divide data for model training and evaluation

## Data Access

Data files are referenced by the pipeline scripts:
- `src/data_ingestion.py` - Loads raw data
- `src/preprocessing.py` - Processes and transforms data
- `src/train.py` - Uses processed data for training

## Note

Do not commit large data files to Git. Use `.gitignore` to exclude:
- `data/raw/**`
- `data/processed/**`
- `data/features/**`
- `*.csv`
- `*.parquet`
