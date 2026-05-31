"""Setup configuration for the MLOps project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="churn-mlops",
    version="1.0.0",
    author="ML Team",
    description="Machine Learning pipeline for customer churn prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.0",
        "pydantic>=2.6.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "black>=24.0.0",
            "ruff>=0.3.0",
            "mypy>=1.0.0",
        ],
        "azure": [
            "azure-ai-ml>=1.14.0",
            "azure-identity>=1.15.0",
        ],
    },
)
