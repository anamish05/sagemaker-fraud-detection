import numpy as np
import pandas as pd
import joblib

class ProPreprocessor:
    def __init__(self, meta_path):
        # Load the frozen rules from SageMaker
        meta = joblib.load(meta_path)
        self.medians = meta['medians']
        self.median_amount = meta['median_amount']
        self.bounds = meta['bounds']
        self.fills = meta['fills']

    def transform(self, df):
        df = df.copy()
        
        # 1. Handle Amount Outliers (Using TRAIN median)
        df.loc[(df['Amount'] < 0.1) | (df['Amount'] > 500), 'Amount'] = self.median_amount
        
        # 2. Handle NaNs (Using TRAIN medians)
        for col, val in self.medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        # 3. Handle Feature Outliers (Using TRAIN bounds/fills)
        for col, (low, high) in self.bounds.items():
            if col in df.columns:
                fill_val = self.fills[col]
                mask = (df[col] < low) | (df[col] > high)
                df.loc[mask, col] = fill_val
        return df

def add_features(df):
    df = df.copy()
    df['High_amount'] = (df['Amount'] > 200).astype(int)
    df['Sum_features'] = df[['V2', 'V4', 'V6', 'V1']].abs().sum(axis=1)
    df['V4_Amount'] = df['V4'] * df['Amount']
    df['V2_Amount'] = df['V2'] * df['Amount']
    df['V1_sq'] = df['V1']**2
    df['V6_sq'] = df['V6']**2
    return df
