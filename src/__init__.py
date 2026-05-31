"""Initialization module for source package."""

__version__ = "1.0.0"
__author__ = "ML Team"

from . import data_ingestion
from . import preprocessing
from . import train
from . import evaluate

__all__ = [
    "data_ingestion",
    "preprocessing",
    "train",
    "evaluate"
]
