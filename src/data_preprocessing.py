"""
Data Preprocessing Module
=========================

This module handles data cleaning, validation, and initial preprocessing
for the web threat analysis pipeline.

Classes:
--------
DataPreprocessor: Main class for data preprocessing operations
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for web traffic data.

    Attributes:
    -----------
    data_path : str
        Path to the CSV data file
    df : pd.DataFrame
        Loaded dataframe
    """

    def __init__(self, data_path: str):
        """
        Initialize the DataPreprocessor.

        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        """
        self.data_path = data_path
        self.df = None
        logger.info(f"DataPreprocessor initialized with path: {data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def check_data_quality(self) -> dict:
        """
        Check data quality metrics.

        Returns:
        --------
        dict
            Dictionary containing quality metrics
        """
        if self.df is None:
            self.load_data()

        quality_metrics = {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }

        logger.info("Data quality check completed")
        return quality_metrics

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values, duplicates, and data types.

        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe
        """
        if self.df is None:
            self.load_data()

        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_duplicates = initial_rows - len(self.df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")

        # Convert datetime columns
        datetime_columns = ['creation_time', 'end_time', 'time']
        for col in datetime_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                logger.info(f"Converted {col} to datetime")

        # Standardize text columns
        text_columns = ['src_ip_country_code', 'protocol']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.upper()

        logger.info("Data cleaning completed")
        return self.df

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        if self.df is None:
            self.load_data()

        return self.df.describe()

    def detect_outliers(self, column: str, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numeric column.

        Parameters:
        -----------
        column : str
            Column name to check for outliers
        method : str
            Method to use ('iqr' or 'zscore')

        Returns:
        --------
        pd.Series
            Boolean series indicating outliers
        """
        if self.df is None:
            self.load_data()

        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            outliers = z_scores > 3
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

        logger.info(f"Detected {outliers.sum()} outliers in {column} using {method} method")
        return outliers

    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save cleaned data to CSV file.

        Parameters:
        -----------
        output_path : str
            Path to save the cleaned data
        """
        if self.df is None:
            raise ValueError("No data to save. Run clean_data() first.")

        self.df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('../data/CloudWatch_Traffic_Web_Attack.csv')
    df_clean = preprocessor.clean_data()
    print(f"Cleaned data shape: {df_clean.shape}")
