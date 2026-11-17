"""Leakage-safe month-t features for forecast month t+1.

Rows are indexed by feature_month t; labels refer to the next month t+1.
The newest observed month has *no* outcome, but its predictors remain
available in features_lookup.csv so the API can forecast October 2026.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data"
START_FEATURE_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2021-12-01")
VAL_END = pd.Timestamp("2023-12-01")
TEST_END = pd.Timestamp("2026-09-01")

COUNT_FEATURES = ["last1", "mean3", "mean12", "same_month_ly", "trend"]
RELATIVE_FEATURES = [
    "last1_pct_rank", "mean3_pct_rank", "mean12_pct_rank", "same_month_ly_pct_rank",
]
ADDITIONAL_FEATURES = [
    "house_share12", "outside_share12", "night_share12",
    "month_sin", "month_cos", "city_total3",
    "previous_hotspot", "hotspot_rate12", "last_count_rank",
]
ALL_FEATURES = COUNT_FEATURES + RELATIVE_FEATURES + ADDITIONAL_FEATURES
LABEL = "hotspot_next_month"
IDENTITY = ["HOOD_158", "neighbourhood", "feature_month", "forecast_month"]


def build() -> dict:
    panel = pd.read_csv(PROC / "panel.csv", parse_dates=["MONTH_START"])
    events = pd.read_csv(
        PROC / "cleaned.csv",
        usecols=["MONTH_START", "HOOD_158", "PREMISES_TYPE", "OCC_HOUR"],
        parse_dates=["MONTH_START"],
        low_memory=False,
    )
    panel = panel.sort_values(["HOOD_158", "MONTH_START"]).reset_index(drop=True)
    # Build monthly house/outside/night counts across *all* neighbourhood-month cells.
    premises = events["PREMISES_TYPE"].fillna("").str.strip().str.lower()
    hour = pd.to_numeric(events["OCC_HOUR"], errors="coerce")
    events["house"] = premises.eq("house").astype(int)
    events["outside"] = premises.eq("outside").astype(int)
    events["night"] = (hour.ge(22) | hour.le(5)).astype(int)
    breakdown = (events.groupby(["HOOD_158", "MONTH_START"])
                 [["house", "outside", "night"]].sum().reset_index())
    df = panel.merge(breakdown, how="left", on=["HOOD_158", "MONTH_START"],
                     validate="one_to_one")
    for col in ["house", "outside", "night"]:
        df[col] = df[col].fillna(0).astype(int)

    g = df.groupby("HOOD_158", sort=False)
    df["last1"] = df["theft_count"]
    df["mean3"] = g["theft_count"].transform(
        lambda x: x.rolling(3, min_periods=3).mean()
    )
    df["mean12"] = g["theft_count"].transform(
        lambda x: x.rolling(12, min_periods=12).mean()
    )
    # For month t, target is t+1. The same calendar month last year is t-11.
    df["same_month_ly"] = g["theft_count"].shift(11)
    df["trend"] = df["mean3"] - df["mean12"]
    df["previous_hotspot"] = df["hotspot"]
    df["hotspot_rate12"] = g["hotspot"].transform(
        lambda x: x.rolling(12, min_periods=12).mean()
    )
    df["last_count_rank"] = df["monthly_count_rank"]

    for col in ["house", "outside", "night", "theft_count"]:
        df[f"{col}_sum12"] = g[col].transform(
            lambda x: x.rolling(12, min_periods=12).sum()
        )
    volume = df["theft_count_sum12"].replace({0: np.nan})
    for target, source in [
        ("house_share12", "house"), ("outside_share12", "outside"),
        ("night_share12", "night"),
    ]:
        # Define 0 share if the 12-month theft denominator is 0.
        df[target] = (df[f"{source}_sum12"] / volume).fillna(0)
    df = df.drop(columns=[f"{c}_sum12" for c in [
        "house", "outside", "night", "theft_count"
    ]])

    for source in ["last1", "mean3", "mean12", "same_month_ly"]:
        df[source + "_pct_rank"] = df.groupby("MONTH_START")[source].rank(
            pct=True, ascending=True, method="average"
        )

    df["feature_month"] = df["MONTH_START"]
    df["forecast_month"] = df["feature_month"] + pd.offsets.MonthBegin(1)
    target_calendar = df["forecast_month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * target_calendar / 12)
    df["month_cos"] = np.cos(2 * np.pi * target_calendar / 12)
    city_monthly = df.groupby("MONTH_START")["theft_count"].sum().sort_index()
    city_total3 = city_monthly.rolling(3, min_periods=3).mean()
    df["city_total3"] = df["feature_month"].map(city_total3)

    # Known labels from t+1, *never* input to ML: group shift(-1) on each
    # neighbourhood; the cutoff is computed across all 158 at t+1.
    df["next_month_theft_count"] = g["theft_count"].shift(-1)
    df[LABEL] = g["hotspot"].shift(-1)
    df["next_month_cutoff"] = g["cutoff_75pct"].shift(-1)

    df = df.loc[df["feature_month"] >= START_FEATURE_MONTH].copy()
    if df[ALL_FEATURES].isna().any().any():
        na = df[ALL_FEATURES].isna().sum()
        raise ValueError(f"Missing historical features:\n{na[na>0]}")
    expected_months = pd.date_range(df["feature_month"].min(), df["feature_month"].max(), freq="MS")
    if len(df) != 158 * len(expected_months):
        raise AssertionError("Some neighbourhood-month feature rows are missing")

    # The last feature month remains available for a genuine future forecast.
    df["forecast_month"] = df["forecast_month"].dt.strftime("%Y-%m")
    df["feature_month"] = df["feature_month"].dt.strftime("%Y-%m")
    data_cols = IDENTITY + ALL_FEATURES + [
        LABEL, "next_month_theft_count", "next_month_cutoff",
    ]
    dataset = df[data_cols].sort_values(["forecast_month", "HOOD_158"]).reset_index(drop=True)
    lookup = dataset.drop(columns=[LABEL, "next_month_theft_count", "next_month_cutoff"])
    lookup.to_csv(PROC / "features_lookup.csv", index=False)

    labelled = dataset.dropna(subset=[LABEL]).copy()
    labelled[LABEL] = labelled[LABEL].astype(int)
    # Date boundaries are defined by the *month predicted*, not feature month.
    train = labelled.loc[labelled["forecast_month"] <= "2021-12"]
    val = labelled.loc[
        (labelled["forecast_month"] >= "2022-01") &
        (labelled["forecast_month"] <= "2023-12")
    ]
    test = labelled.loc[
        (labelled["forecast_month"] >= "2024-01") &
        (labelled["forecast_month"] <= "2026-09")
    ]
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        frame.to_csv(PROC / f"{name}.csv", index=False)

    schema = dict(
        target=LABEL,
        identifier_columns=IDENTITY,
        numeric_features=ALL_FEATURES,
        train_target_months=["2015-02", "2021-12"],
        validation_target_months=["2022-01", "2023-12"],
        test_target_months=["2024-01", "2026-09"],
        available_forecast_months=["2015-02", "2026-10"],
        class_definition="Next-month reported theft count >= the 75th percentile across 158 Toronto neighbourhoods; ties included",
        last_forecast_has_no_label=True,
        class_1_meaning="Hotspot relative to other Toronto neighbourhoods, NOT probability of a particular car being stolen",
        notes="All feature values are known by end of feature_month. The latest forecast month (2026-10) is unlabelled.",
    )
    stats = dict(
        all_feature_rows=len(dataset),
        labelled_rows=len(labelled),
        unlabelled_rows=int(dataset[LABEL].isna().sum()),
        train_rows=len(train), validation_rows=len(val), test_rows=len(test),
        hotspot_pct_train=round(float(train[LABEL].mean() * 100), 3),
        hotspot_pct_val=round(float(val[LABEL].mean() * 100), 3),
        hotspot_pct_test=round(float(test[LABEL].mean() * 100), 3),
        feature_count=len(ALL_FEATURES),
        first_target_month=labelled["forecast_month"].min(),
        last_target_month=labelled["forecast_month"].max(),
        next_unlabelled_month=dataset.loc[dataset[LABEL].isna(), "forecast_month"].unique().tolist(),
    )
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    build()
