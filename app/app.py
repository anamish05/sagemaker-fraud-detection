import streamlit as st
import pandas as pd
import joblib
import io
import numpy as np
from preprocessing_utils import ProPreprocessor, add_features
import requests
from pathlib import Path


# --- HELPER: Parse Text Report to Table ---
def parse_classification_report(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # We skip the header/footer and focus on the classes
        data = []
        for line in lines[2:4]:  # This usually captures Class 0 and Class 1
            parts = line.split()
            data.append({
                "Class": parts[0],
                "Precision": parts[1],
                "Recall": parts[2],
                "F1-Score": parts[3],
                "Support": parts[4]
            })
        return pd.DataFrame(data)
    except:
        return None

BASE_DIR = Path(__file__).resolve().parent

# --- PAGE CONFIG ---
st.set_page_config(page_title="FraudEngine Classificator", layout="wide")

# --- HEADER ---
st.title("🛡️ Financial Fraud Detection Dashboard")
st.markdown("---")

# --- SIDEBAR: MODEL EVIDENCE ---
st.sidebar.header("📊 Model Performance")

# Show the Confusion Matrix
# if st.sidebar.checkbox("Show Confusion Matrix"):
#     st.sidebar.image("confusion_matrix.png", use_container_width=True)

if st.sidebar.checkbox("Show Roc Curve"):
    st.sidebar.image(BASE_DIR /"roc_curve.png", use_container_width=True)

# Show the Classification Report text
if st.sidebar.checkbox("Show Classification Report"):
    st.sidebar.markdown("**Detailed Metrics**")
    report_df = parse_classification_report(BASE_DIR /"classification_report.txt")
    if report_df is not None:
        st.sidebar.table(report_df)
    else:
        st.sidebar.warning("Could not parse report file.")

# Show the ROC AUC Score
try:
    with open(BASE_DIR /"metrics.txt", "r") as f:
        metrics = f.read()
    st.sidebar.metric("ROC AUC Score", metrics.split(":")[-1].strip())
except:
    pass

# --- MAIN: PREDICTION LOGIC ---

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    # This is the new file you downloaded from S3
    preprocessor = ProPreprocessor("preprocessor_meta.joblib")
    return model, preprocessor

model, preprocessor = load_artifacts()



st.subheader("1. Data Ingestion")

# Create two columns for the two ways to get data
col_upload, col_demo = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload your own CSV", type="csv")

with col_demo:
    st.write("--- or ---")
    run_demo = st.button("🎲 Generate Random Demo Batch (4k rows)")

# Logic to determine which data to use
raw_data = None

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
elif run_demo:

    demo_url = "https://raw.githubusercontent.com/anamish05/sagemaker-fraud-detection/refs/heads/main/data/raw_to_test.csv"
    with st.spinner("Fetching data from GitHub..."):
        try:
            raw_data = pd.read_csv(
                demo_url, 
                sep=',', encoding='utf-8', on_bad_lines='skip',
                storage_options={'User-Agent': 'Mozilla/5.0'}
            ).sample(n=4000).reset_index(drop=True)
            st.success("Successfully loaded 4,000 random transactions!")
        except Exception as e:
            st.error(f"Error loading demo: {e}")
            response = requests.get(demo_url)
            st.write(f"URL Status Code: {response.status_code}") 
            st.code(response.text[:100])

# --- PROCESSING & PREDICTION (Only runs if raw_data exists) ---
if raw_data is not None:
    st.write("Preview of Data:", raw_data.head(5))
    
    # We trigger the analysis automatically for the demo, 
    # or via button for the upload
    if run_demo or st.button("🚀 Analyze Uploaded Data"):
        with st.spinner("Applying SageMaker Preprocessing rules..."):
            # 1. Preprocess
            cleaned_data = preprocessor.transform(raw_data)
            featured_data = add_features(cleaned_data)

            X = featured_data.drop(columns=['Class'], errors='ignore')
        
            # Predict
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]
            
            raw_data['Prediction'] = preds
            raw_data['Risk_Score'] = probs
            
            # Display Results
            st.subheader("2. Analysis Results")
            frauds = raw_data[raw_data['Prediction'] == 1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scanned", len(raw_data))
            col2.metric("Fraud Detected", len(frauds), delta_color="inverse")
            # Get the ROC AUC from your metrics file
            col3.metric("Model Confidence (AUC)", metrics.split(":")[-1].strip())
    
            # Tabs for Data vs. Model Evidence
            tab1, tab2 = st.tabs(["🔍 Flagged Transactions", "📊 Model Evidence"])
    
            with tab1:
                if len(frauds) > 0:
                    st.error("🚨 High-Risk Transactions Found")
                    st.dataframe(frauds.sort_values(by='Risk_Score', ascending=False))
                else:
                    st.success("✅ No fraudulent patterns detected.")
    
            with tab2:
                st.write("The Confusion Matrix below shows how the model performed on unseen test data.")
                
                # DISPLAY CONFUSION MATRIX HERE
                st.image(BASE_DIR /"confusion_matrix.png", caption="Confusion Matrix (Test Set)", use_container_width=True)
                
                # Display the Classification Report Table
                st.write("### Detailed Class Metrics")
                report_df = parse_classification_report("classification_report.txt")
                if report_df is not None:
                    st.table(report_df)
