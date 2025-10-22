"""
Cybersecurity Web Threat Analysis Dashboard
==========================================

Main Streamlit application for interactive threat analysis.

Author: Utkarsh Sharma
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector
from classifier import TrafficClassifier
import utils

# Page configuration
st.set_page_config(
    page_title="Web Threat Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
custom_css = """
<style>
.main {
    padding: 0rem 1rem;
}
.stAlert {
    padding: 1rem;
    border-radius: 0.5rem;
}
h1 {
    color: #1f77b4;
    padding-bottom: 1rem;
}
h2 {
    color: #2c3e50;
    padding-top: 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset."""
    try:
        data_path = Path(__file__).parent.parent / 'data' / 'CloudWatch_Traffic_Web_Attack.csv'
        preprocessor = DataPreprocessor(str(data_path))
        df = preprocessor.clean_data()

        engineer = FeatureEngineer(df)
        df = engineer.create_all_features()

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def main():
    """Main application function."""

    # Sidebar
    with st.sidebar:
        st.title("🛡️ Threat Analysis")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Data Analysis", "🤖 ML Models", "🎯 Predictions"]
        )

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This dashboard provides comprehensive analysis of suspicious "
            "web traffic patterns using machine learning."
        )

        st.markdown("### Contact")
        st.markdown("**Utkarsh Sharma**")
        st.markdown("📧 nany23111996@gmail.com")

    # Load data
    df = load_data()

    if df is None:
        st.error("Failed to load data. Please check the data file.")
        return

    # Main content based on selected page
    if page == "🏠 Home":
        show_home(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 ML Models":
        show_ml_models(df)
    elif page == "🎯 Predictions":
        show_predictions(df)


def show_home(df):
    """Display home page."""
    st.title("🛡️ Cybersecurity: Web Threat Analysis Dashboard")
    st.markdown("### Advanced ML-powered Analysis of Suspicious Web Traffic")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Unique IPs", f"{df['src_ip'].nunique()}")

    with col3:
        st.metric("Countries", f"{df['src_ip_country_code'].nunique()}")

    with col4:
        st.metric("Total Traffic", "362 MB")

    st.markdown("---")

    # Overview
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This project analyzes **282 suspicious web traffic records** from AWS CloudWatch 
    to detect potential cyber attacks and security threats.

    **Key Features:**
    - 🤖 Isolation Forest for anomaly detection
    - 🌲 Random Forest for traffic classification
    - 📊 Interactive visualizations
    - 🎯 Real-time threat prediction
    """)


def show_data_analysis(df):
    """Display data analysis page."""
    st.title("📊 Data Analysis")

    st.markdown("### Dataset Overview")
    st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Visualizations
    st.markdown("### Geographic Distribution")
    country_counts = df['src_ip_country_code'].value_counts()
    fig = px.bar(x=country_counts.values, y=country_counts.index, orientation='h')
    st.plotly_chart(fig, use_container_width=True)


def show_ml_models(df):
    """Display ML models page."""
    st.title("🤖 Machine Learning Models")

    model_type = st.selectbox("Select Model", 
                              ["Isolation Forest", "Random Forest"])

    if model_type == "Isolation Forest":
        st.markdown("### Isolation Forest - Anomaly Detection")

        contamination = st.slider("Contamination Rate", 0.01, 0.20, 0.05)

        if st.button("Train Model", type="primary"):
            with st.spinner("Training..."):
                detector = AnomalyDetector(contamination=contamination)
                detector.fit(df)
                result_df = detector.get_anomaly_details(df)

                n_anomalies = (result_df['anomaly_label'] == 'Anomaly').sum()
                st.success(f"Detected {n_anomalies} anomalies!")

                st.dataframe(result_df.head())
    else:
        st.markdown("### Random Forest - Classification")
        st.info("Model training interface")


def show_predictions(df):
    """Display predictions page."""
    st.title("🎯 Threat Predictions")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        pred_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(pred_df)} records")
        st.dataframe(pred_df.head())


if __name__ == "__main__":
    main()
