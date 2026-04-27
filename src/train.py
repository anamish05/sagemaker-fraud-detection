import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SageMaker paths to train and test datasets, model and outputs
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    args = parser.parse_args()

    #  Load Data
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.test, 'test.csv'))

    # Split features and target
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    # Train model. Pipeline normalization+XGBoost with optimized parameters. Search of optimal hyperparameters performed in params_search.ipynb

    pipe_xgb = ImbPipeline([
        ("scaler", MinMaxScaler()),
        ("clf", xgb.XGBClassifier(
            random_state=42,
            subsample=0.8,
            scale_pos_weight=650,
            n_estimators=500,
            min_child_weight=0.5,
            max_depth=12,
            max_delta_step=1,
            learning_rate=0.05,
            colsample_bytree=0.8))
        ])

    pipe_xgb.fit(X_train, y_train)

    # Evaluation
    y_test_pred_proba = pipe_xgb.predict_proba(X_test)[::, 1]
    y_pred = pipe_xgb.predict(X_test)
    
    # Save ROC curve
    plot = RocCurveDisplay.from_estimator(pipe_xgb, X_test, y_test)
    plt.title("Model ROC Curve")
    plt.savefig(os.path.join(args.output_data_dir, 'roc_curve.png'))

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(args.output_data_dir, 'classification_report.txt'), 'a') as f:
        f.write(report)

    # Generate Confusion matrix
    cm = ConfusionMatrixDisplay.from_estimator(pipe_xgb, X_test, y_test)
    plt.savefig(os.path.join(args.output_data_dir, 'confusion_matrix.png'))

    # Save the ROC AUC Score
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    with open(os.path.join(args.output_data_dir, 'metrics.txt'), 'a') as f:
        f.write(f"ROC_AUC_Score: {auc_score}\n")
    
    #  Save Model
    joblib.dump(pipe_xgb, os.path.join(args.model_dir, "model.joblib"))
    print("Training and Evaluation complete.")

