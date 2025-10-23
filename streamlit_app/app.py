import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Cybersecurity Web Threat Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- SIDEBAR ----
with st.sidebar:
    st.title("Web Threat Analysis Dashboard")
    st.write("""
    **Objective:**  
    Analyze and identify suspicious web traffic using advanced machine learning.

    **Project Highlights:**  
    - Automatic anomaly detection (Isolation Forest)
    - Intelligent traffic classification (Random Forest)
    - Interactive graphs and analysis
    - Predict on custom or live data
    - Complete dataset explorer and business insights
    """)

    st.info("""
    Use the sections below to:
    - Explore imported network records
    - Visualize threat patterns and attack trends
    - Train, review, and evaluate model output
    - Predict on new traffic data
    """)

    st.write("""
    **Author:** Utkarsh Sharma  
    **Contact:** utkarshaily2004@gmail.com  
    **GitHub:** https://github.com/utkarsh-world  
    """)

    st.markdown("---")
    st.caption("This dashboard is part of a Cybersecurity ML portfolio focused on data-driven threat defense and attack mitigation best practices.")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "CloudWatch_Traffic_Web_Attack.csv"
    return pd.read_csv(data_path)

df = load_data()

# ---- NAVIGATION ----
PAGES = {
    "Home": "home",
    "Dataset Explorer": "explorer",
    "ML Models": "models",
    "Predict/Upload": "predict",
    "Visual Analysis": "visuals",
    "Project Documentation": "docs",
    "Contact": "contact"
}

selected_page = st.selectbox(
    "Go to Section",
    list(PAGES.keys()),
    key="mainnav"
)

st.markdown("---")

# ---- PAGE LOGIC ----

def show_home():
    st.header("Welcome to the Cybersecurity Web Threat Analysis App")
    st.write("""
    This project helps identify and classify suspicious web traffic sessions using a real AWS CloudWatch dataset.  
    It combines exploratory data analysis, feature engineering, two types of machine-learning models, and business-friendly recommendations.
    """)
    st.subheader("Project Objective and Value")
    st.write("""
    - Stop critical threats and malicious infrastructure reconnaissance  
    - Detect attacks before they cause harm  
    - Give security teams rich visual context for decisions  
    - Empower business leaders with clear summaries and technical proof
    """)
    st.subheader("Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Unique IPs", df['src_ip'].nunique())
    col3.metric("Source Countries", df['src_ip_country_code'].nunique())
    col4.metric("Total Data (MB)", f"{df['bytes_in'].sum() / 1024 / 1024:.2f}")

    st.markdown("---")

def show_explorer():
    st.header("Dataset Explorer")
    st.write("Browse, filter, and understand the structure of the suspicious web traffic records used in modeling.")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("Data Shape:", df.shape)
    if st.checkbox("Show summary statistics"):
        st.dataframe(df.describe())

def show_models():
    st.header("Machine Learning Models")
    st.write("""
    **Isolation Forest**: Unsupervised anomaly detection, finds outliers in multivariate space  
    **Random Forest Classifier**: Supervised model to classify suspicious traffic intensity
    """)
    st.write("Train and review model performance here. For demonstration, please use the included model scripts to retrain – or invoke models from your 'models' directory directly.")
    st.warning("This demo does not retrain models in-browser to keep run times fast. Use `train_models.py` for comprehensive retraining.")

def show_predict():
    st.header("Traffic Session Prediction / Upload Data")
    st.write("Upload new CSV files or run predictions on the current loaded dataset (if models are trained and available).")
    st.info("Model must be trained and model `.pkl` files available in `/models`.")
    uploaded = st.file_uploader("Upload CSV For Prediction", type="csv")
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        st.dataframe(df_upload.head())
        st.write("You can now use prediction modules/scripts on the uploaded data.")

def show_visuals():
    st.header("Visual Analysis: Threat Patterns & Trends")
    st.subheader("Traffic by Country")
    country_counts = df['src_ip_country_code'].value_counts()
    fig = px.bar(
        x=country_counts.index, y=country_counts.values, 
        labels={"x":"Country", "y":"Sessions"},
        title="Traffic Origin by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Temporal Analysis")
    temp = df.groupby(df['creation_time'].str[:13]).size()
    fig2 = px.line(x=temp.index, y=temp.values, labels={"x":"Hour", "y":"Sessions"}, title="Sessions over Time by Hour")
    st.plotly_chart(fig2, use_container_width=True)

def show_docs():
    st.header("Project Documentation")
    st.write("""
    - **Business Problem:** Detect high-risk web sessions for proactive incident response  
    - **Data Source:** AWS CloudWatch VPC Flow Logs, 282 suspicious sessions  
    - **Key Features:** Advanced EDA, Feature Engineering, ML with Isolation Forest and Random Forest  
    - **Security Recommendations:** Block critical IPs, monitor for outliers, automate future alerting  
    """)
    st.write("**Repository:** [GitHub - Web Threat Analysis](https://github.com/YOUR_USERNAME/Web-Threat-Analysis-Cybersecurity)")
    st.write("**Author:** Utkarsh Sharma")

def show_contact():
    st.header("Contact & Support")
    st.write("""
    - For feedback, bug reports, or collaboration requests, please email **utkarshaily2004@gmail.com**
    - Project author LinkedIn: (www.linkedin.com/in/utkarsh-sharma-a5a17936b)
    - GitHub: (https://github.com/utkarsh-world)
    """)
    st.markdown("Thank you for exploring the Cybersecurity Threat Dashboard. Stay secure!")

# ---- RENDER PAGE ----
if selected_page == "Home":
    show_home()
elif selected_page == "Dataset Explorer":
    show_explorer()
elif selected_page == "ML Models":
    show_models()
elif selected_page == "Predict/Upload":
    show_predict()
elif selected_page == "Visual Analysis":
    show_visuals()
elif selected_page == "Project Documentation":
    show_docs()
elif selected_page == "Contact":
    show_contact()
