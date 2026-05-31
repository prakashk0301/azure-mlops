# ML Pipelines Directory

## Overview

This directory contains pipeline orchestration and deployment configurations for the ML models.

## Contents

- **Azure ML Pipelines**: Azure Machine Learning pipeline definitions
- **Workflow Configurations**: Orchestration scripts for model training and evaluation
- **Deployment Scripts**: Scripts for deploying models to production

## CI/CD Integration

The GitHub Actions workflows are located in `.github/workflows/` and trigger automatically:
- On every push to main branch
- On scheduled basis (weekly Monday at 2 AM UTC)
- On pull requests

## Pipeline Stages

1. **Data Ingestion**: Load and validate input data
2. **Preprocessing**: Clean and transform data
3. **Feature Engineering**: Create features for ML models
4. **Model Training**: Train ML model with configured hyperparameters
5. **Model Evaluation**: Evaluate model performance
6. **Model Registration**: Register model to Azure ML Registry
7. **Deployment**: Deploy to production or staging environment

## Usage

See main README.md for pipeline execution instructions.
