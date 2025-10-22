"""
Traffic Classification Module
=============================

This module implements Random Forest classifier for traffic classification.

Classes:
--------
TrafficClassifier: Random Forest-based traffic classifier
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficClassifier:
    """
    Classifies web traffic as high-risk or normal using Random Forest.

    Attributes:
    -----------
    model : RandomForestClassifier
        Trained classifier
    scaler : StandardScaler
        Feature scaler
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 random_state: int = 42):
        """
        Initialize the TrafficClassifier.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest (default: 100)
        max_depth : int
            Maximum depth of trees (default: 10)
        random_state : int
            Random state for reproducibility (default: 42)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = ['Normal Traffic', 'High Traffic']

        logger.info(f"TrafficClassifier initialized with {n_estimators} estimators")

    def prepare_target(self, X: pd.DataFrame, threshold_percentile: float = 75) -> pd.Series:
        """
        Create target variable based on total_bytes threshold.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        threshold_percentile : float
            Percentile for threshold (default: 75)

        Returns:
        --------
        pd.Series
            Target variable
        """
        threshold = X['total_bytes'].quantile(threshold_percentile / 100)
        y = (X['total_bytes'] > threshold).astype(int)

        logger.info(f"Target created with threshold at {threshold_percentile}th percentile")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return y

    def train(self, X: pd.DataFrame, y: pd.Series,
              feature_columns: Optional[list] = None,
              test_size: float = 0.3) -> Dict:
        """
        Train the classifier.

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable
        feature_columns : list, optional
            List of feature columns to use
        test_size : float
            Test set proportion (default: 0.3)

        Returns:
        --------
        dict
            Training metrics
        """
        if feature_columns is None:
            feature_columns = [
                'bytes_in', 'bytes_out', 'avg_packet_size',
                'transfer_ratio', 'hour'
            ]

        self.feature_names = feature_columns
        X_features = X[feature_columns]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }

        logger.info(f"Training completed. Test accuracy: {metrics['test_accuracy']:.4f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict traffic class.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict

        Returns:
        --------
        np.ndarray
            Predictions
        """
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict

        Returns:
        --------
        np.ndarray
            Class probabilities
        """
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.

        Parameters:
        -----------
        X : pd.DataFrame
            Test features
        y : pd.Series
            True labels

        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(
                y, y_pred, target_names=self.class_names
            )
        }

        return metrics

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

    def load_model(self, model_path: str, scaler_path: str) -> 'TrafficClassifier':
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
            Classifier with loaded model
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
        return self


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    df = pd.read_csv('../data/CloudWatch_Traffic_Web_Attack.csv')

    classifier = TrafficClassifier()
    y = classifier.prepare_target(df)
    metrics = classifier.train(df, y)

    print(f"Model performance: {metrics}")
