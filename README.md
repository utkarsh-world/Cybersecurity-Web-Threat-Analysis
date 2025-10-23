# 🛡️ Cybersecurity: Suspicious Web Threat Interactions Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

**Advanced ML-powered analysis of suspicious web traffic patterns using AWS CloudWatch data**

[Live Demo](https://your-streamlit-app-url.streamlit.app) • [Documentation](#documentation) • [Report Issue](https://github.com/yourusername/Web-Threat-Analysis-Cybersecurity/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Machine Learning Models](#-machine-learning-models)
- [Results](#-results)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This project provides a comprehensive analysis of **282 suspicious web traffic records** collected through AWS CloudWatch, aimed at detecting potential cyber attacks and security threats. Using advanced machine learning techniques including **Isolation Forest** and **Random Forest classifiers**, the system achieves **97.65% accuracy** in identifying malicious traffic patterns.

### 🎯 Project Objectives

- ✅ Detect anomalous traffic patterns using unsupervised learning
- ✅ Classify high-risk traffic sessions with supervised learning
- ✅ Identify geographic and temporal attack patterns
- ✅ Provide actionable security recommendations
- ✅ Deploy interactive visualization dashboard

### 📊 Dataset Information

- **Source**: AWS CloudWatch VPC Flow Logs
- **Records**: 282 suspicious traffic sessions
- **Time Period**: April 25-26, 2024 (11-hour window)
- **Features**: 16 original + 8 engineered features
- **Protocol**: 100% HTTPS (Port 443)
- **Detection**: WAF rule-based flagging

---

## ✨ Key Features

### 🤖 Machine Learning Capabilities

| Feature | Description | Performance |
|---------|-------------|-------------|
| **Anomaly Detection** | Isolation Forest algorithm | 5.32% anomaly rate |
| **Traffic Classification** | Random Forest classifier | 97.65% accuracy |
| **Feature Engineering** | 8 derived features | 49.69% feature importance |
| **Real-time Scoring** | Prediction pipeline | < 100ms latency |

### 📈 Advanced Analytics

- **Geographic Analysis**: Traffic distribution across 7 countries
- **Temporal Analysis**: Hourly attack pattern identification
- **Correlation Analysis**: Feature relationship mapping
- **Outlier Detection**: IQR-based statistical methods
- **Visual Analytics**: 9 interactive charts and visualizations

### 🖥️ Interactive Dashboard

- Real-time data exploration
- Dynamic filtering and drill-down
- Model prediction interface
- Security alerts and recommendations
- Export capabilities (PDF, CSV)

---

## 🎬 Demo

### Dashboard Preview

```
🏠 Home Page                  📊 Data Analysis              🤖 ML Models
├── Project Overview         ├── Traffic Distribution      ├── Anomaly Detection
├── Key Metrics             ├── Geographic Patterns       ├── Classification
└── Quick Insights          └── Temporal Trends           └── Feature Importance
```

### Sample Visualizations

<div align="center">

| Distribution Analysis | Geographic Mapping | Anomaly Detection |
|:---:|:---:|:---:|
| ![](outputs/charts/distribution.png) | ![](outputs/charts/geography.png) | ![](outputs/charts/anomalies.png) |

</div>

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Web-Threat-Analysis-Cybersecurity.git
cd Web-Threat-Analysis-Cybersecurity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py
```

### Docker Installation (Optional)

```bash
# Build Docker image
docker build -t web-threat-analysis .

# Run container
docker run -p 8501:8501 web-threat-analysis
```

---

## 💻 Usage

### 1. Data Analysis Pipeline

```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

# Load and preprocess data
preprocessor = DataPreprocessor('data/CloudWatch_Traffic_Web_Attack.csv')
df_clean = preprocessor.clean_data()

# Engineer features
engineer = FeatureEngineer(df_clean)
df_features = engineer.create_features()

# Train models
trainer = ModelTrainer(df_features)
models = trainer.train_all_models()
```

### 2. Anomaly Detection

```python
from src.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(contamination=0.05)
detector.fit(df_features)
anomalies = detector.predict(df_features)

# Get anomaly details
anomaly_df = detector.get_anomaly_details()
print(f"Detected {len(anomaly_df)} anomalies")
```

### 3. Traffic Classification

```python
from src.classifier import TrafficClassifier

classifier = TrafficClassifier(n_estimators=100, max_depth=10)
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)

# Evaluate performance
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### 4. Streamlit Dashboard

```bash
# Launch interactive dashboard
streamlit run streamlit_app/app.py

# Access at: http://localhost:8501
```

---

## 📁 Project Structure

```
Web-Threat-Analysis-Cybersecurity/
│
├── 📊 data/
│   ├── CloudWatch_Traffic_Web_Attack.csv    # Original dataset
│   ├── cleaned_data.csv                      # Processed data
│   └── README.md                             # Data documentation
│
├── 📝 src/
│   ├── __init__.py
│   ├── data_preprocessing.py                 # Data cleaning module
│   ├── feature_engineering.py                # Feature creation
│   ├── anomaly_detector.py                   # Isolation Forest implementation
│   ├── classifier.py                         # Random Forest classifier
│   ├── model_training.py                     # Training pipeline
│   ├── evaluation.py                         # Model evaluation
│   ├── visualization.py                      # Plotting utilities
│   └── utils.py                              # Helper functions
│
├── 🤖 models/
│   ├── isolation_forest.pkl                  # Trained anomaly detector
│   ├── random_forest.pkl                     # Trained classifier
│   ├── scaler.pkl                            # Feature scaler
│   └── model_config.json                     # Model parameters
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb            # EDA notebook
│   ├── 02_feature_engineering.ipynb         # Feature analysis
│   ├── 03_model_training.ipynb              # Model development
│   └── 04_results_analysis.ipynb            # Results visualization
│
├── 📤 outputs/
│   ├── charts/                              # Generated visualizations
│   ├── reports/                             # Analysis reports
│   └── predictions/                         # Model predictions
│
├── 🖥️ streamlit_app/
│   ├── app.py                               # Main Streamlit app
│   ├── pages/                               # Multi-page app
│   │   ├── 1_Data_Analysis.py
│   │   ├── 2_ML_Models.py
│   │   └── 3_Predictions.py
│   ├── assets/                              # Static assets
│   └── config.toml                          # App configuration
│
├── 🧪 tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📋 requirements.txt                       # Python dependencies
├── 🐳 Dockerfile                            # Docker configuration
├── ⚙️ setup.py                              # Package setup
├── 📖 README.md                             # This file
├── 📄 LICENSE                               # MIT License
└── .gitignore                               # Git ignore rules
```

---

## 🤖 Machine Learning Models

### 1. Isolation Forest (Anomaly Detection)

**Purpose**: Identify unusual traffic patterns without labeled data

**Configuration**:
```python
{
    "contamination": 0.05,      # Expected anomaly rate
    "n_estimators": 100,        # Number of trees
    "max_samples": "auto",      # Samples per tree
    "random_state": 42          # Reproducibility
}
```

**Performance**:
- ✅ Detected 15 anomalies (5.32% of traffic)
- ✅ Perfect identification of critical IP (155.91.45.242)
- ✅ All top 5 threats correctly flagged

### 2. Random Forest Classifier (Traffic Classification)

**Purpose**: Classify traffic sessions as high-risk or normal

**Configuration**:
```python
{
    "n_estimators": 100,        # Number of trees
    "max_depth": 10,            # Tree depth
    "min_samples_split": 2,     # Min samples for split
    "criterion": "gini",        # Split quality measure
    "random_state": 42          # Reproducibility
}
```

**Performance**:
- ✅ Accuracy: **97.65%**
- ✅ Precision: **95-98%**
- ✅ Recall: **95-98%**
- ✅ F1-Score: **95-98%**
- ✅ Only 2 misclassifications in 85 test samples

### Feature Importance

| Rank | Feature | Importance | Significance |
|------|---------|------------|--------------|
| 1 | avg_packet_size | 49.69% | Most critical |
| 2 | bytes_in | 25.86% | High impact |
| 3 | bytes_out | 18.72% | Moderate impact |
| 4 | transfer_ratio | 5.66% | Low impact |
| 5 | hour | 0.08% | Minimal impact |

---

## 📊 Results

### Critical Findings

#### 🚨 Threat Level: CRITICAL

1. **Single IP Dominance**
   - IP: `155.91.45.242` (United States)
   - Traffic: **92.4%** of total (334.26 MB)
   - Sessions: 13 anomalous sessions
   - **Action**: IMMEDIATE BLOCKING REQUIRED

2. **Data Exfiltration Indicators**
   - Incoming/Outgoing Ratio: **14.2:1**
   - Pattern: Large-scale data download
   - All connections: HTTP 200 (successful)

3. **Automated Attack Signatures**
   - Session Duration: Exactly **600 seconds** (all 282 sessions)
   - Variation: **Zero** (indicates scripting)
   - Campaign: Coordinated attack

4. **Geographic Concentration**
   - United States: **40.1%** (113 sessions)
   - Canada: **25.5%** (72 sessions)
   - Combined: **65.6%** of all traffic

5. **Targeted Attack**
   - Destination: `10.138.69.97` (single target)
   - Pattern: Focused exploitation
   - Risk: High priority for forensics

### Performance Metrics

```
┌─────────────────────────────────────────────────────┐
│           MODEL PERFORMANCE SUMMARY                  │
├─────────────────────────────────────────────────────┤
│ Isolation Forest:                                    │
│   • Anomalies Detected: 15 (5.32%)                  │
│   • Contamination Rate: 5%                           │
│   • Critical IP Identified: ✓                        │
├─────────────────────────────────────────────────────┤
│ Random Forest Classifier:                            │
│   • Accuracy: 97.65%                                 │
│   • Precision: 0.95-0.98                            │
│   • Recall: 0.95-0.98                               │
│   • F1-Score: 0.95-0.98                             │
│   • Confusion Matrix: [[63,1], [1,20]]              │
└─────────────────────────────────────────────────────┘
```

---

## 🎨 Streamlit Dashboard

### Features

#### 🏠 Home Page
- Project overview and objectives
- Key metrics dashboard
- Quick security insights
- Recent threat alerts

#### 📊 Data Analysis
- Interactive data exploration
- Dynamic filtering options
- Geographic heatmaps
- Temporal trend analysis
- Traffic distribution plots

#### 🤖 ML Models
- Model performance metrics
- Feature importance visualization
- Confusion matrix analysis
- ROC curves (if applicable)
- Model comparison

#### 🎯 Predictions
- Real-time threat prediction
- Upload custom data
- Batch prediction mode
- Downloadable results
- Security recommendations

#### 📈 Visualizations
- 9 interactive charts
- Customizable views
- Export to PNG/PDF
- Responsive design

### Deployment

#### Local Deployment
```bash
streamlit run streamlit_app/app.py
```

#### Streamlit Cloud
1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Select repository and branch
4. Deploy!

#### Heroku Deployment
```bash
heroku create your-app-name
git push heroku main
heroku open
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code coverage
pytest --cov=src tests/

# Run linter
flake8 src/ tests/

# Format code
black src/ tests/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

---

## 📝 Documentation

### API Documentation

Generate API documentation using Sphinx:

```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

### Notebooks

Explore Jupyter notebooks for detailed analysis:
- `01_data_exploration.ipynb`: Initial data analysis
- `02_feature_engineering.ipynb`: Feature creation process
- `03_model_training.ipynb`: Model development
- `04_results_analysis.ipynb`: Results and insights

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add: Your feature description"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Add comments for complex logic
- Ensure all tests pass

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Utkarsh Sharma**

- 📧 Email: nany23111996@gmail.com
- 💼 LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/utkarsh-sharma-a5a17936b)
- 🐙 GitHub: [@yourusername](https://github.com/utkarsh-world)
- 🌐 Portfolio: [Your Website] soon to come

---

## 🙏 Acknowledgments

- AWS CloudWatch for traffic data
- scikit-learn for ML algorithms
- Streamlit for dashboard framework
- The cybersecurity community for insights

---

## 📚 References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. ICDM.
2. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
3. AWS CloudWatch Documentation
4. MITRE ATT&CK Framework

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/Web-Threat-Analysis-Cybersecurity?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/Web-Threat-Analysis-Cybersecurity?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/Web-Threat-Analysis-Cybersecurity?style=social)

---

<div align="center">

**Made with ❤️ and ☕ by Utkarsh Sharma**

⭐ Star this repo if you find it helpful!

</div>
