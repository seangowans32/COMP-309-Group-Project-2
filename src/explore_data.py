"""COMP309 Group Project 2 — Step 1 starter for Fatima.

This is a basic exploration helper so Sean's cleaning has a clear upstream
step. It is NOT Fatima's finished exploration report, dictionary, or charts.
It does NOT modify the raw data.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/auto_theft.csv"

def explore():
    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    print("\n--- RAW DATA ---")
    print("Rows:", len(df), "Columns:", len(df.columns))
    print("Columns and dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\n--- DATA QUALITY ---")
    print("Null values by column:")
    print(df.isna().sum().to_string())
    print("Unique event IDs:", df["EVENT_UNIQUE_ID"].nunique())
    print("Repeated event ID rows:", df["EVENT_UNIQUE_ID"].duplicated().sum())
    print("Non-numeric neighbourhood codes:", pd.to_numeric(df["HOOD_158"], errors="coerce").isna().sum())
    print("\n--- THEFT ACTIVITY ---")
    print("Occurrence years:")
    print(df["OCC_YEAR"].value_counts(dropna=False).sort_index().to_string())
    print("Premises types:")
    print(df["PREMISES_TYPE"].value_counts(dropna=False).head(12).to_string())
    print("Ten most common neighbourhoods (raw counts; BEFORE deduplication):")
    print(df["NEIGHBOURHOOD_158"].value_counts(dropna=False).head(10).to_string())
    print("\nExploration starter only. Fatima owns the full analysis and charts.")

if __name__ == "__main__":
    explore()
