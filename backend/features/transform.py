import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from backend.logger import logger

SILVER_PATH = "data/silver/flights_clean.csv"
SCALER_PATH = "models/scaler.joblib"

LOG_COLS = ['distance', 'air_time']
SIGNED_LOG_COLS = ['dep_delay', 'arr_delay']      # log1p can't be used directly
LOW_CARD_COLS = ['carrier', 'origin']             # one-hot
HIGH_CARD_COLS = ['flight', 'tailnum', 'dest']    # over max_categorical_cardinality (50)

def encode_time(df):
    "Gives hour, day of week, and month from the time_hour column."
    df = df.copy()
    t = pd.to_datetime(df['time_hour'])
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["month"] = t.dt.month

    for col, values, period in [("hour", df["hour"], 24),
                                ("day_of_week", df["day_of_week"], 7),
                                ("month", df["month"]-1, 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * values / period)

    logger.info("Time Features encoded")
    return df

def log_transform(df, cols = LOG_COLS, signed_cols = SIGNED_LOG_COLS):
    """Log transformation for LOG_COLS and SIGNED_LOG_COLS"""
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    for col in signed_cols:
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    logger.info("Log transformed")
    return df

def standardize(df, cols, scaler = None, path = SCALER_PATH):
    """Standardize the columns in df."""
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler().fit(df[cols])
        os.makedirs(os.path.dirname(path), exist_ok = True)
        joblib.dump(scaler, path)
        logger.info(f"Scaler saved: {path}")
    df[cols] = scaler.transform(df[cols])
    return df, scaler

def encode_categoricals(df, low_cols = LOW_CARD_COLS, high_cols = HIGH_CARD_COLS, freq_maps = None):
    """Encode categorical columns in df."""
    df = df.copy()
    if freq_maps is None:
        freq_maps = {col: df[col].value_counts(normalize=True) for col in high_cols}

    for col in high_cols:
        df[f"{col}_freq"] = df[col].map(freq_maps[col]).astype(float).fillna(0)
    df = df.drop(columns=high_cols)

    df = pd.get_dummies(df, columns=low_cols, dtype=int)
    logger.info(f"One-hot encoded: {low_cols}, frequency encoded: {high_cols}")
    return df, freq_maps

if __name__ == "__main__":
    df = pd.read_csv(SILVER_PATH)
    logger.info(f"Silver data imported: {df.shape}")
    df = encode_time(df)
    df = log_transform(df)
    df, _ = encode_categoricals(df)
    logger.info(f"Transformed shape: {df.shape}")
