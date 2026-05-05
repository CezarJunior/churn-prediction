"""
Download the IBM Telco Customer Churn dataset and save it locally.

Usage:
    python data/download_data.py
"""

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from src.config import DATA_URL, RAW_DATA_PATH


def download_dataset(url: str = DATA_URL, dest: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Download the Telco Customer Churn CSV from the given URL and persist it.

    Parameters
    ----------
    url : str
        Remote URL of the dataset.
    dest : Path
        Local path where the file will be saved.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame for quick inspection.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from:\n  {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest.write_bytes(response.content)
    print(f"Saved to: {dest}")

    df = pd.read_csv(dest)
    _print_basic_info(df)
    return df


def _print_basic_info(df: pd.DataFrame) -> None:
    """Print a concise summary of the downloaded dataset."""
    print("\n" + "=" * 50)
    print("Dataset Overview")
    print("=" * 50)
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing values   : None")
    else:
        print(f"  Missing values   :\n{missing.to_string()}")

    if "Churn" in df.columns:
        churn_rate = df["Churn"].map({"Yes": 1, "No": 0}).mean()
        print(f"  Churn rate       : {churn_rate:.1%}")

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("=" * 50 + "\n")


if __name__ == "__main__":
    download_dataset()
