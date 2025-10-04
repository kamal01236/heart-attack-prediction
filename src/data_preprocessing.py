import numpy as np
import pandas as pd


def add_glucose_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "glucose" in df.columns:
        df["glucose_missing"] = df["glucose"].isnull().astype(int)
    return df


def winsorize_columns(df: pd.DataFrame, cols=None, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        return df
    for col in cols:
        if col in df.columns:
            low = df[col].quantile(lower_q)
            high = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run lightweight, non-learning preprocessing used by training and inference.
    This intentionally keeps only deterministic transforms (missing indicators, winsorization).
    """
    df = add_glucose_missing(df)
    winsor_cols = ["tot cholesterol", "Systolic BP", "Diastolic BP", "BMI", "glucose"]
    df = winsorize_columns(df, winsor_cols)
    return df
