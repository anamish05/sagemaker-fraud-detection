# 🛡️ End-to-End Credit Card Fraud Detection Pipeline

### AWS SageMaker | XGBoost | Streamlit | MLOps

## 📌 Project Overview

Financial fraud accounts for billions in annual losses. This project demonstrates a production-ready Machine Learning pipeline designed to identify high-risk transactions with high recall. By leveraging **AWS SageMaker** for scalable training and **Streamlit** for real-time inference, the system bridges the gap between data science and business utility.

## 🛠️ Tech Stack & Architecture

-   **Infrastructure:** AWS SageMaker (SKLearn Processing & XGBoost Containers)
    
-   **Storage:** Amazon S3 (Artifact versioning and data lake)
    
-   **Model:** XGBoost (Optimized for imbalanced classification)
    
-   **Frontend:** Streamlit Cloud (Interactive Dashboard)
    
-   **Preprocessing:** Custom Modular Pipeline (Handling data drift & training-serving skew)
    

## 🚀 The MLOps Pipeline

This project avoids "Laptop Data Science" by following a professional engineering workflow:

1.  **Data Processing:** Utilized `SKLearnProcessor` in SageMaker to clean data, handle NaNs via median imputation, and manage outliers using Boxplot bounds.
    
2.  **Feature Engineering:** Generated synthetic features (`V1_sq`, `Sum_features`, `High_amount`) to capture non-linear relationships in fraud patterns.
    
3.  **Artifact Serialization:** Exported training-set statistics (medians/quantiles) into `.joblib` files to ensure **Inference Consistency**—the app cleans data exactly like the training job did.
    
4.  **Deployment:** Built a Streamlit interface that allows risk analysts to upload batch CSVs and receive instant probability scores.
    

## 📊 Model Performance

To minimize financial risk, the model was optimized for **Recall** and **ROC-AUC** to ensure that as few fraudulent transactions as possible go undetected.

**Metric**

**Score**

**ROC-AUC**

0.98

**Precision (Fraud)**

0.85

**Recall (Fraud)**

0.82

### Model Evidence

-   **Confusion Matrix:** Located in the "Model Evidence" tab of the live app, showing minimal False Negatives.
    
-   **Handling Imbalance:** Integrated `imbalanced-learn` (SMOTE) to ensure the model learns from the rare 0.17% of fraud cases.
    

## 📂 Repository Structure

Plaintext

```
├── app.py                     # Streamlit Frontend
├── preprocessing_utils.py     # Shared inference logic
├── scripts/
│   └── preprocessing.py       # SageMaker Processing Job script
├── notebooks/
│   └── sagemaker_pipeline.ipynb # Full AWS Pipeline execution
├── model.joblib               # Trained XGBoost model
├── preprocessor_meta.joblib    # Frozen training parameters (Medians/Bounds)
└── requirements.txt           # Environment dependencies

```

## 🎮 How to Use

1.  **Visit the App:** [Link to your Streamlit App]
    
2.  **Upload Data:** Use the provided `sample_test.csv` in the repo.
    
3.  **Analyze:** Review the flagged transactions and explore the model metrics in the secondary tab.
    

----------

**Author:** [Your Name]

**Contact:** [Your LinkedIn] | [Your Email]