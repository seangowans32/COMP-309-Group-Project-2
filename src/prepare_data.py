"""Prepare the COMP309 Toronto auto-theft group project's monthly panel.

Source: Toronto Police auto theft open-data CSV (read-only input).
One EVENT_UNIQUE_ID counts once. Include complete months Jan 2014–Sep 2026.
A hotspot is any neighbourhood whose count meets or exceeds the *within-month*
75th-percentile count. Tied counts are kept together, not arbitrarily split.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/auto_theft.csv"
OUT = ROOT / "data"
START = pd.Timestamp("2014-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-10-01")
EXPECTED_NEIGHBOURHOODS = 158


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW, encoding="utf-8-sig", low_memory=False)
    required = {
        "EVENT_UNIQUE_ID", "OCC_DATE", "OCC_YEAR", "OCC_MONTH",
        "HOOD_158", "NEIGHBOURHOOD_158", "PREMISES_TYPE", "OCC_HOUR",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Required source columns missing: {sorted(missing)}")
    log = {"raw_rows": int(len(raw)), "raw_columns": int(len(raw.columns))}
    # Important for reproducibility: the occasional ID conflict is audited.
    raw["OCC_DATE"] = pd.to_datetime(raw["OCC_DATE"], errors="coerce")
    raw["_hood"] = pd.to_numeric(raw["HOOD_158"], errors="coerce")
    conflicts = raw.groupby("EVENT_UNIQUE_ID", dropna=False).agg(
        date_variants=("OCC_DATE", "nunique"),
        hood_variants=("_hood", "nunique"),
    )
    conflict_ids = conflicts[
        (conflicts["date_variants"] > 1) | (conflicts["hood_variants"] > 1)
    ]
    log["conflicting_event_ids"] = int(len(conflict_ids))
    if len(conflict_ids):
        raise ValueError("Some duplicate event IDs have conflicting dates/neighbourhoods; review before de-duplicating.")

    # Drop duplicates before filtering, matching the group plan.
    df = raw.drop_duplicates("EVENT_UNIQUE_ID", keep="first").copy()
    log["duplicate_rows_removed"] = int(len(raw) - len(df))
    log["unique_events"] = int(len(df))
    invalid_date = df["OCC_DATE"].isna()
    log["unparseable_occurrence_dates"] = int(invalid_date.sum())
    df = df.loc[~invalid_date].copy()

    invalid_hood = df["_hood"].isna()
    log["unknown_neighbourhood_events_removed"] = int(invalid_hood.sum())
    df = df.loc[~invalid_hood].copy()
    invalid_period = ~df["OCC_DATE"].between(
        START, END_EXCLUSIVE, inclusive="left"
    )
    log["outside_complete_data_period_removed"] = int(invalid_period.sum())
    df = df.loc[~invalid_period].copy()

    df["HOOD_158"] = df["_hood"].astype(int)
    df["NEIGHBOURHOOD_158"] = df["NEIGHBOURHOOD_158"].astype(str).str.strip()
    df["MONTH_START"] = df["OCC_DATE"].dt.to_period("M").dt.to_timestamp()
    log["cleaned_events"] = int(len(df))
    log["neighbourhoods"] = int(df["HOOD_158"].nunique())
    if log["neighbourhoods"] != EXPECTED_NEIGHBOURHOODS:
        raise ValueError(f"Expected 158 neighbourhoods, got {log['neighbourhoods']}")
    if df["EVENT_UNIQUE_ID"].duplicated().any():
        raise AssertionError("Duplicate event IDs remain")

    # OCC_MONTH is a month *name*, e.g. January (not a month number).
    # Verify rather than silently mixing two definitions of occurrence date.
    expected_names = df["OCC_DATE"].dt.month_name().str.lower()
    month_labels = df["OCC_MONTH"].astype(str).str.strip().str.lower()
    months_comparable = df["OCC_MONTH"].notna()
    year_values = pd.to_numeric(df["OCC_YEAR"], errors="coerce")
    years_comparable = year_values.notna()
    log["occ_month_disagreements"] = int(
        ((month_labels != expected_names) & months_comparable).sum()
    )
    log["occ_year_disagreements"] = int(
        ((year_values != df["OCC_DATE"].dt.year) & years_comparable).sum()
    )

    names = df.groupby("HOOD_158")["NEIGHBOURHOOD_158"].agg(
        lambda series: series.mode().iloc[0]
    )
    names.index.name = "HOOD_158"
    names.rename("neighbourhood").to_csv(OUT / "neighbourhoods.csv")

    clean = df.drop(columns=["_hood"]).sort_values(
        ["MONTH_START", "HOOD_158", "EVENT_UNIQUE_ID"]
    )
    clean.to_csv(OUT / "cleaned.csv", index=False)

    counts = (clean.groupby(["MONTH_START", "HOOD_158"])
              .size().rename("theft_count"))
    months = pd.date_range(START, END_EXCLUSIVE - pd.DateOffset(months=1), freq="MS")
    idx = pd.MultiIndex.from_product(
        [months, sorted(names.index)],
        names=["MONTH_START", "HOOD_158"],
    )
    panel = counts.reindex(idx, fill_value=0).reset_index()
    panel["neighbourhood"] = panel["HOOD_158"].map(names)
    panel["cutoff_75pct"] = panel.groupby("MONTH_START")["theft_count"].transform(
        lambda x: x.quantile(0.75)
    )
    panel["hotspot"] = (panel["theft_count"] >= panel["cutoff_75pct"]).astype(int)
    # A higher theft count has a smaller rank. Equal counts share the same rank.
    panel["monthly_count_rank"] = panel.groupby("MONTH_START")["theft_count"].rank(
        method="min", ascending=False
    ).astype(int)
    panel = panel.sort_values(["HOOD_158", "MONTH_START"]).reset_index(drop=True)
    panel.to_csv(OUT / "panel.csv", index=False)

    log.update(
        panel_rows=int(len(panel)),
        months=int(panel["MONTH_START"].nunique()),
        zero_count_cells=int((panel["theft_count"] == 0).sum()),
        zero_count_percent=round(float((panel["theft_count"] == 0).mean() * 100), 3),
        hotspot_count=int(panel["hotspot"].sum()),
        hotspot_percent=round(float(panel["hotspot"].mean() * 100), 3),
        hotspot_min_per_month=int(panel.groupby("MONTH_START")["hotspot"].sum().min()),
        hotspot_max_per_month=int(panel.groupby("MONTH_START")["hotspot"].sum().max()),
        first_month=str(months.min().date()),
        last_month=str(months.max().date()),
    )
    print(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    prepare()
