"""
Feature Engineering Module
==========================

This module handles creation of derived features from raw web traffic data.

Classes:
--------
FeatureEngineer: Main class for feature engineering operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates and transforms features for machine learning models.

    Attributes:
    -----------
    df : pd.DataFrame
        Input dataframe
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with cleaned data
        """
        self.df = df.copy()
        logger.info("FeatureEngineer initialized")

    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create time-based features.

        Returns:
        --------
        pd.DataFrame
            Dataframe with temporal features
        """
        # Session duration
        self.df['session_duration'] = (
            self.df['end_time'] - self.df['creation_time']
        ).dt.total_seconds()

        # Extract hour and day of week
        self.df['hour'] = self.df['creation_time'].dt.hour
        self.df['day_of_week'] = self.df['creation_time'].dt.dayofweek
        self.df['day_name'] = self.df['creation_time'].dt.day_name()
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

        logger.info("Created temporal features")
        return self.df

    def create_traffic_features(self) -> pd.DataFrame:
        """
        Create traffic-related features.

        Returns:
        --------
        pd.DataFrame
            Dataframe with traffic features
        """
        # Total bytes
        self.df['total_bytes'] = self.df['bytes_in'] + self.df['bytes_out']

        # Transfer ratio
        self.df['transfer_ratio'] = self.df['bytes_in'] / self.df['bytes_out']
        self.df['transfer_ratio'] = self.df['transfer_ratio'].replace([np.inf, -np.inf], np.nan)
        self.df['transfer_ratio'] = self.df['transfer_ratio'].fillna(
            self.df['transfer_ratio'].median()
        )

        # Average packet size
        self.df['avg_packet_size'] = (
            self.df['total_bytes'] / self.df['session_duration']
        )
        self.df['avg_packet_size'] = self.df['avg_packet_size'].replace([np.inf, -np.inf], np.nan)
        self.df['avg_packet_size'] = self.df['avg_packet_size'].fillna(
            self.df['avg_packet_size'].median()
        )

        # Bytes per second
        self.df['bytes_in_per_sec'] = self.df['bytes_in'] / self.df['session_duration']
        self.df['bytes_out_per_sec'] = self.df['bytes_out'] / self.df['session_duration']

        logger.info("Created traffic features")
        return self.df

    def create_categorical_features(self) -> pd.DataFrame:
        """
        Create categorical features and encodings.

        Returns:
        --------
        pd.DataFrame
            Dataframe with categorical features
        """
        # Flag for suspicious activity (all are suspicious in this dataset)
        self.df['is_suspicious'] = 1

        # Protocol encoding (if multiple protocols exist)
        if 'protocol' in self.df.columns:
            self.df['protocol_encoded'] = pd.Categorical(self.df['protocol']).codes

        # Country encoding
        if 'src_ip_country_code' in self.df.columns:
            self.df['country_encoded'] = pd.Categorical(
                self.df['src_ip_country_code']
            ).codes

        logger.info("Created categorical features")
        return self.df

    def create_statistical_features(self) -> pd.DataFrame:
        """
        Create statistical features based on aggregations.

        Returns:
        --------
        pd.DataFrame
            Dataframe with statistical features
        """
        # Log transformations for skewed features
        self.df['log_bytes_in'] = np.log1p(self.df['bytes_in'])
        self.df['log_bytes_out'] = np.log1p(self.df['bytes_out'])
        self.df['log_total_bytes'] = np.log1p(self.df['total_bytes'])

        # Squared features for capturing non-linear relationships
        self.df['bytes_in_squared'] = self.df['bytes_in'] ** 2
        self.df['bytes_out_squared'] = self.df['bytes_out'] ** 2

        logger.info("Created statistical features")
        return self.df

    def create_ip_features(self) -> pd.DataFrame:
        """
        Create IP-based features.

        Returns:
        --------
        pd.DataFrame
            Dataframe with IP features
        """
        # Count of sessions per source IP
        ip_counts = self.df['src_ip'].value_counts()
        self.df['src_ip_session_count'] = self.df['src_ip'].map(ip_counts)

        # Flag for high-frequency IPs
        threshold = self.df['src_ip_session_count'].quantile(0.75)
        self.df['is_high_frequency_ip'] = (
            self.df['src_ip_session_count'] > threshold
        ).astype(int)

        logger.info("Created IP features")
        return self.df

    def create_all_features(self) -> pd.DataFrame:
        """
        Create all engineered features.

        Returns:
        --------
        pd.DataFrame
            Dataframe with all engineered features
        """
        self.create_temporal_features()
        self.create_traffic_features()
        self.create_categorical_features()
        self.create_statistical_features()
        self.create_ip_features()

        logger.info(f"Feature engineering completed. Total features: {len(self.df.columns)}")
        return self.df

    def get_feature_list(self, feature_type: str = 'all') -> List[str]:
        """
        Get list of features by type.

        Parameters:
        -----------
        feature_type : str
            Type of features to return ('all', 'numeric', 'categorical')

        Returns:
        --------
        List[str]
            List of feature names
        """
        if feature_type == 'numeric':
            return self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif feature_type == 'categorical':
            return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            return self.df.columns.tolist()

    def save_features(self, output_path: str) -> None:
        """
        Save engineered features to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save the feature dataframe
        """
        self.df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    df = pd.read_csv('../data/CloudWatch_Traffic_Web_Attack.csv')
    engineer = FeatureEngineer(df)
    df_features = engineer.create_all_features()
    print(f"Feature engineering completed. Shape: {df_features.shape}")
