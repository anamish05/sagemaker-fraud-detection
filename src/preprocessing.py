import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #  Paths where SageMaker will mount S3 data
    input_data_path = "/opt/ml/processing/input/creditcard.csv"
    output_dir = "/opt/ml/processing/output"
    
    #  Read the raw data
    df = pd.read_csv(input_data_path)
    
    # Clean & Feature Engineer 
    class DataPreprocessing():
        
      def __init__(self):
        self.nans = None
        self.medians_nan = None
        self.median_amount  = None

      def fit(self, X):
        self.nans = X.columns[X.isna().any()].tolist()
        self.medians_nan = X[self.nans].median()
        self.median_amount = X["Amount"].median()

      def transform(self, X):
        X = X.copy()
        # handle missing data
        if self.nans:
          X[self.nans] = X[self.nans].fillna(self.medians_nan)

        return X

    # Functions to apply bounds for outliers hadling
    def boxplot_bounds(series, whis=1.5):
      s = series.dropna()
      q1 = s.quantile(0.25)
      q3 = s.quantile(0.75)
      iqr = q3 - q1
      low = q1 - whis * iqr
      high = q3 + whis * iqr
      return low, high
    
    def apply_bounds(df, f, low, high, fill):
      df = df.copy()
      mask = (df[f] < low) | (df[f] > high)
      df.loc[mask, f] = fill
      return df

    # Reducing memory usage
    def reduce_mem_usage(df):

        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
        for col in df.columns:
            col_type = df[col].dtype
    
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')
    
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
        return df

    # Reducing memory usage
    df = reduce_mem_usage(df)

    # Split data to train and test, applying stratify
    TARGET_NAME = 'Class'
    train, valid = train_test_split(
        df,
        test_size=0.33,
        random_state=42,
        shuffle=True,
        stratify=df[TARGET_NAME]
        )

    # Handling outliers in Amount column, replacing them by median. Substantiation provided in preliminary_analysis.ipynb.
    train.loc[(train['Amount']<0.1) | (train['Amount']>500), 'Amount'] = train['Amount'].median()

    # Handling other columns (they are unlabeled due to financial data undisclosure and applied PCA)
    NO_TARGET_FEATURES_TRAIN = train.drop(columns=[TARGET_NAME, 'Amount'])
    preprocessor = DataPreprocessing()
    preprocessor.fit(train)
    train = preprocessor.transform(train)
    valid = preprocessor.transform(valid)
    # Handling outliers in other columns
    NO_TARGET_NAMES = NO_TARGET_FEATURES_TRAIN.columns.tolist()
    bounds = {}
    fills = {}
    
    for f in NO_TARGET_NAMES:
      low, high = boxplot_bounds(train[f], whis=1.5)
      bounds[f] = (low, high)
      fills[f] = train[f].median()
    
    for f in NO_TARGET_NAMES:
      low, high = bounds[f]
      fill = fills[f]
      train = apply_bounds(train, f, low, high, fill)
      valid = apply_bounds(valid, f, low, high, fill)
        
    # Adding additional features to catch silent FNs. The columns selected based on statistical analysis provided in preliminary_analysis.ipynb.
    def add_features(df):
      df['High_amount']=(df['Amount']>200).astype(int)
      df['Sum_features']=df[['V2', 'V4', 'V6', 'V1']].abs().sum(axis=1)
      df['V4_Amount'] = df['V4'] * df['Amount']
      df['V2_Amount'] = df['V2'] * df['Amount']
      df['V1_sq'] = df['V1']**2
      df['V6_sq'] = df['V6']**2
      return df
        
    X_train_f = add_features(train)
    X_valid_f = add_features(valid)

    # Saving the cleaning parameters to apply on raw test data in Streamlit.
    preprocessor_meta = {
        'medians': preprocessor.medians_nan.to_dict(),
        'median_amount': float(preprocessor.median_amount),
        'bounds': bounds,  #  contains (low, high) tuples
        'fills': fills
    }

    
    # Saving to the output directory
    # SageMaker automatically uploads everything in this folder back to S3
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    
    X_train_f.to_csv(f"{output_dir}/train/train.csv", index=False)
    X_valid_f.to_csv(f"{output_dir}/test/test.csv", index=False)
    joblib.dump(preprocessor_meta, f"{output_dir}/preprocessor_meta.joblib")
    
    print("Successfully processed data and saved to output!")
