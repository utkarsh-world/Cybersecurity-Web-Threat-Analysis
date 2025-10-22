"""
Utility Functions Module
========================

This module provides helper functions for the project.

Functions:
----------
- save_dataframe: Save dataframe to CSV
- load_model_config: Load model configuration
- create_directories: Create project directories
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, Any, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save dataframe to CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output file path
    index : bool
        Whether to save index (default: False)
    """
    df.to_csv(filepath, index=index)
    logger.info(f"Dataframe saved to {filepath}")


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from JSON file.

    Parameters:
    -----------
    config_path : str
        Path to configuration file

    Returns:
    --------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_model_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save model configuration to JSON file.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Output file path
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


def create_directories(directories: List[str]) -> None:
    """
    Create project directories if they don't exist.

    Parameters:
    -----------
    directories : list
        List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {len(directories)} directories")


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
    --------
    Path
        Project root path
    """
    return Path(__file__).parent.parent


def format_bytes(bytes_value: float) -> str:
    """
    Format bytes into human-readable format.

    Parameters:
    -----------
    bytes_value : float
        Bytes value

    Returns:
    --------
    str
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def calculate_threat_score(row: pd.Series) -> float:
    """
    Calculate threat score based on traffic characteristics.

    Parameters:
    -----------
    row : pd.Series
        Row of dataframe with traffic data

    Returns:
    --------
    float
        Threat score (0-100)
    """
    score = 0

    # High bytes in
    if row.get('bytes_in', 0) > 1000000:
        score += 30

    # High transfer ratio
    if row.get('transfer_ratio', 0) > 10:
        score += 25

    # High packet size
    if row.get('avg_packet_size', 0) > 10000:
        score += 20

    # Anomaly flag
    if row.get('anomaly_label', 'Normal') == 'Anomaly':
        score += 25

    return min(score, 100)


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
