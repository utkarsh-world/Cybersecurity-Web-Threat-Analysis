"""
Anomaly Detection Module
========================

This module implements Isolation Forest for detecting anomalous web traffic patterns.

Classes:
--------
AnomalyDetector: Isolation Forest-based anomaly detector
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in web traffic using Isolation Forest algorithm.

    Attributes:
    -----------
    contamination : float
        Expected proportion of anomalies in the dataset
    model : IsolationForest
        Trained Isolation Forest model
    scaler : StandardScaler
        Feature scaler
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize the AnomalyDetector.

        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies (default: 0.05)
        n_estimators : int
            Number of trees in the forest (default: 100)
        random_state : int
            Random state for reproducibility (default: 42)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None

        logger.info(f"AnomalyDetector initialized with contamination={contamination}")

    def fit(self, X: pd.DataFrame, feature_columns: Optional[list] = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model.

        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        feature_columns : list, optional
            List of feature columns to use

        Returns:
        --------
        self
            Fitted detector
        """
        if feature_columns is None:
            # Default features for anomaly detection
            feature_columns = [
                'bytes_in', 'bytes_out', 'session_duration',
                'avg_packet_size', 'total_bytes'
            ]

        self.feature_names = feature_columns
        X_features = X[feature_columns]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # Fit model
        self.model.fit(X_scaled)

        logger.info(f"Model fitted on {len(X)} samples with {len(feature_columns)} features")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.

        Parameters:
        -----------
        X : pd.DataFrame
            Data to predict

        Returns:
        --------
        np.ndarray
            Predictions (-1 for anomaly, 1 for normal)
        """
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)

        n_anomalies = (predictions == -1).sum()
        logger.info(f"Predicted {n_anomalies} anomalies out of {len(X)} samples")

        return predictions

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters:
        -----------
        X : pd.DataFrame
            Data to score

        Returns:
        --------
        np.ndarray
            Anomaly scores (lower scores = more anomalous)
        """
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        scores = self.model.decision_function(X_scaled)
        return scores

    def get_anomaly_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies.

        Parameters:
        -----------
        X : pd.DataFrame
            Data with predictions

        Returns:
        --------
        pd.DataFrame
            Dataframe with anomaly details
        """
        predictions = self.predict(X)
        scores = self.decision_function(X)

        result = X.copy()
        result['anomaly_score'] = predictions
        result['anomaly_label'] = np.where(predictions == -1, 'Anomaly', 'Normal')
        result['decision_score'] = scores

        return result

    def save_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained model and scaler.

        Parameters:
        -----------
        model_path : str
            Path to save the model
        scaler_path : str
            Path to save the scaler
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path: str, scaler_path: str) -> 'AnomalyDetector':
        """
        Load trained model and scaler.

        Parameters:
        -----------
        model_path : str
            Path to the saved model
        scaler_path : str
            Path to the saved scaler

        Returns:
        --------
        self
            Detector with loaded model
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
        return self


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    df = pd.read_csv('../data/CloudWatch_Traffic_Web_Attack.csv')

    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)
    anomaly_details = detector.get_anomaly_details(df)

    print(f"Anomalies detected: {(anomaly_details['anomaly_label'] == 'Anomaly').sum()}")
