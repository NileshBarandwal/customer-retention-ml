"""
Data ingestion, cleaning, and feature engineering pipeline.

Handles loading raw customer data, preprocessing (missing values, encoding,
scaling), and producing train/validation/test splits ready for model training.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_PATH = Path("data/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed")


def download_data(url: str = DATA_URL, path: Path = RAW_DATA_PATH) -> Path:
    """Download the Telco Churn CSV from *url* and save to *path* if absent.

    Args:
        url:  Direct download URL for the raw CSV.
        path: Local destination path.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"[download_data] Dataset already present at '{path}'. Skipping download.")
    else:
        print(f"[download_data] Downloading dataset from:\n  {url}")
        urllib.request.urlretrieve(url, path)
        print(f"[download_data] Saved to '{path}'.")

    return path


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame exactly as it appears on disk.

    Raises:
        FileNotFoundError: If *path* does not exist. Run download_data() first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Call download_data() first."
        )
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {len(df):,} rows × {df.shape[1]} columns from '{path}'.")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print a structured summary of the DataFrame for quick EDA.

    Prints:
        - Shape (rows, columns)
        - Column dtypes
        - Missing-value counts per column (only columns with gaps)
        - Class distribution of the 'Churn' target column

    Args:
        df: Raw or partially processed DataFrame containing a 'Churn' column.
    """
    print("=" * 60)
    print("BASIC DATASET INFO")
    print("=" * 60)

    print(f"\nShape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n--- Column dtypes ---")
    print(df.dtypes.to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\n--- Missing values : none ---")
    else:
        print("\n--- Missing values ---")
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"count": missing, "pct%": pct}).to_string())

    if "Churn" in df.columns:
        print("\n--- Churn class distribution ---")
        counts = df["Churn"].value_counts()
        pct = (df["Churn"].value_counts(normalize=True) * 100).round(2)
        print(pd.DataFrame({"count": counts, "pct%": pct}).to_string())

    print("=" * 60)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw DataFrame and return a model-ready version.

    Steps applied:
        1. Convert 'TotalCharges' to numeric (contains whitespace strings).
        2. Impute resulting NaN 'TotalCharges' with the column median.
        3. Drop the 'customerID' column (non-informative identifier).
        4. Encode the 'Churn' target column as 0 (No) / 1 (Yes).

    Args:
        df: Raw DataFrame as returned by load_data().

    Returns:
        Cleaned DataFrame with numeric target and no customerID column.
    """
    df = df.copy()

    # TotalCharges contains " " strings for new customers with zero tenure
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_fixed = df["TotalCharges"].isna().sum()
    if n_fixed:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        print(
            f"[clean_data] Imputed {n_fixed} NaN TotalCharges values with "
            f"median ({median_val:.2f})."
        )

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        print("[clean_data] Dropped 'customerID' column.")

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        print("[clean_data] Encoded 'Churn': Yes→1, No→0.")

    print(f"[clean_data] Done. Shape after cleaning: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Label-encode all remaining categorical columns and split X / y.

    Each object-dtype column is encoded in-place with sklearn LabelEncoder.
    The 'Churn' column is extracted as the target and removed from features.

    Args:
        df: Cleaned DataFrame (output of clean_data()).

    Returns:
        Tuple (X, y):
            X: DataFrame of encoded features.
            y: Binary Series (0/1) for the Churn target.
    """
    df = df.copy()

    y = df.pop("Churn").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(
        f"[feature_engineering] Label-encoded {len(cat_cols)} categorical columns: "
        f"{cat_cols}"
    )
    print(f"[feature_engineering] Feature matrix: {df.shape}, Target: {y.shape}")

    return df, y


def save_processed_data(
    X: pd.DataFrame,
    y: pd.Series,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Persist the processed feature matrix and target vector to disk.

    Saves:
        <path>/X.csv  — feature matrix
        <path>/y.csv  — target vector

    Args:
        X:    Processed feature DataFrame.
        y:    Target Series.
        path: Directory where files will be written.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    X.to_csv(path / "X.csv", index=False)
    y.to_csv(path / "y.csv", index=False, header=True)

    print(f"[save_processed_data] Saved X → '{path / 'X.csv'}' ({X.shape})")
    print(f"[save_processed_data] Saved y → '{path / 'y.csv'}' ({y.shape})")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline() -> tuple[pd.DataFrame, pd.Series]:
    """Execute the full data pipeline end-to-end and return (X, y).

    Downloads the raw CSV if absent, cleans it, engineers features, saves
    processed outputs, and prints a final summary.
    """
    download_data()
    raw_df = load_data()
    basic_info(raw_df)
    cleaned_df = clean_data(raw_df)
    X, y = feature_engineering(cleaned_df)
    save_processed_data(X, y)

    churn_rate = y.mean() * 100
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total records : {len(X):,}")
    print(f"  Churn rate    : {churn_rate:.2f}%")
    print(f"  Features      : {X.shape[1]}")
    print("=" * 60)

    return X, y


if __name__ == "__main__":
    run_pipeline()
