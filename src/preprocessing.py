# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(path)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant model features."""
    model_features = [
        # Metadata
        'asset_type', 
        # Usage Metrics
        'odometer', 'utilization_7day_avg', 'days_since_service',
        # Historical Reliability
        'historical_failure_count',
        # Telemetry Trends
        'vibration_index', 'vibration_7day_std', 
        'temp_delta', 'oil_press_std_7d', 
        'load_7day_std', 'stress_7day_avg'
    ]
    return df[model_features + ['date', 'target']].copy(), model_features

def temporal_split(df: pd.DataFrame, model_features: list, cutoff_day=240, end_day=270):
    """Split data into train/test sets using Temporal Split."""
    train_df = df[df['date'] < cutoff_day].copy()
    test_df = df[(df['date'] >= cutoff_day) & (df['date'] <= end_day)].copy()

    # One-hot encode categorical variables
    X_train = pd.get_dummies(train_df[model_features], columns=['asset_type'], drop_first=True)
    X_test = pd.get_dummies(test_df[model_features], columns=['asset_type'], drop_first=True)

    # Align train and test columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols].fillna(X_train[numeric_cols].median()))
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols].fillna(X_train[numeric_cols].median()))

    y_train = train_df['target']
    y_test = test_df['target']

    return X_train, X_test, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='../data/processed'):
    """Save preprocessed CSVs for modeling."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train_temporal.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test_temporal.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train_temporal.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test_temporal.csv', index=False)
    print("✅ Processed data saved successfully.")
