from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detector import AnomalyDetector
from src.classifier import TrafficClassifier

# Load & clean data
data_path = 'data/CloudWatch_Traffic_Web_Attack.csv'
preprocessor = DataPreprocessor(data_path)
df_clean = preprocessor.clean_data()

# Engineer features
engineer = FeatureEngineer(df_clean)
df_features = engineer.create_all_features()

# Train Isolation Forest
detector = AnomalyDetector(contamination=0.05)
detector.fit(df_features)
detector.save_model('models/isolation_forest.pkl', 'models/if_scaler.pkl')

# Train Random Forest Classifier
classifier = TrafficClassifier(n_estimators=100, max_depth=10)
y = classifier.prepare_target(df_features)  # Target = high_traffic
classifier.train(df_features, y)
classifier.save_model('models/random_forest.pkl', 'models/rf_scaler.pkl')

print("✓ Both models trained and saved to /models/")