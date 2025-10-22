"""
Cybersecurity Web Threat Analysis Package
=========================================

A comprehensive package for analyzing suspicious web traffic patterns
using machine learning techniques.

Modules:
--------
- data_preprocessing: Data cleaning and preparation
- feature_engineering: Feature creation and transformation
- anomaly_detector: Isolation Forest implementation
- classifier: Random Forest classifier
- model_training: Training pipeline
- evaluation: Model evaluation metrics
- visualization: Plotting utilities
- utils: Helper functions

Author: Utkarsh Sharma
License: MIT
"""

__version__ = '1.0.0'
__author__ = 'Utkarsh Sharma'
__email__ = 'nany23111996@gmail.com'

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .anomaly_detector import AnomalyDetector
from .classifier import TrafficClassifier

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'AnomalyDetector',
    'TrafficClassifier'
]
