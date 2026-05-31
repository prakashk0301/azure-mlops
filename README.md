# Churn Prediction MLOps

An enterprise-grade machine learning operations pipeline for customer churn prediction, built on Azure Cloud with CI/CD automation.

## Project Structure

```
churn-mlops/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipelines
├── configs/
│   ├── config.yaml        # Main configuration
│   └── environment.yaml   # Conda environment
├── data/
│   ├── raw/               # Original datasets
│   ├── processed/         # Cleaned data
│   └── features/          # Engineered features
├── src/
│   ├── data_ingestion.py  # Data loading
│   ├── preprocessing.py   # Data processing
│   ├── train.py           # Model training
│   └── evaluate.py        # Model evaluation
├── tests/
│   └── test_pipeline.py   # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f configs/environment.yaml
conda activate mlops-churn

# Or install with pip
pip install -r requirements.txt
```

### 2. Run Pipeline Locally

```bash
# Data ingestion
python src/data_ingestion.py

# Preprocessing
python src/preprocessing.py

# Training
python src/train.py

# Evaluation
python src/evaluate.py
```

### 3. Run Tests

```bash
pytest tests/ -v --tb=short
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model hyperparameters
- Data paths
- Training settings
- Azure resource names
- Feature selection options

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/mlops-pipeline.yml`) automatically:

1. **Tests** - Runs unit tests on every push
2. **Lint** - Code quality checks with ruff
3. **Train & Deploy** - Trains model on main branch merges

### Required Secrets

Add these to GitHub repository settings:
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

## Model Development Workflow

### Experiment Tracking
- Use MLflow to log experiments and metrics
- Models and metrics stored in `data/models/`

### Feature Engineering
- Processed features in `data/features/`
- Preprocessing logic in `src/preprocessing.py`

### Model Evaluation
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC
- Results saved with model artifacts

## Azure Integration

### Setup Azure ML Workspace

```bash
az login
az ml workspace create \
  --name churn-mlops-ws \
  --resource-group churn-mlops-rg
```

### Register Model

```bash
az ml model create \
  --name churn-model \
  --path data/models/
```

## Development Guidelines

### Code Style
- Format: `black src/`
- Lint: `ruff check src/`
- Types: Use type hints where possible

### Testing
- Unit tests in `tests/`
- Test coverage: Minimum 80%
- Run: `pytest tests/ -v --cov=src`

### Git Workflow
1. Create feature branch: `git checkout -b feature/name`
2. Commit changes: `git commit -m "description"`
3. Push and create PR: `git push origin feature/name`
4. Merge after CI passes

## Monitoring & Logging

- MLflow dashboard: `mlflow ui`
- Pipeline logs: Check GitHub Actions
- Azure logs: Azure Portal > ML Workspace > Jobs

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Test Failures
```bash
pytest tests/ -v --tb=long
```

### Azure Authentication
```bash
az login --use-device-code
```

## Performance Optimization

- Use feature selection to reduce dimensionality
- Tune hyperparameters in `configs/config.yaml`
- Monitor model drift in production
- Retrain on schedule (see CI/CD)

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Submit PR with clear description

## License

MIT License

## Support

For issues or questions, create an issue in the repository.
