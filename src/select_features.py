"""COMP309 section 4.2 — compare feature sets without developing a final model.

Two feature selection methods run on the *training* data only:
(1) SelectKBest(mutual_information); (2) random-forest importance.
Then use simple Logistic Regression / Decision Tree reference classifiers
to compare selected subsets on the chronological validation set.

These are *feature-selection checks*, not Emine's final model evaluation.
Do not use test.csv to choose features, thresholds or model parameters.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data"
TARGET = "hotspot_next_month"


def compare():
    train = pd.read_csv(PROC / "train.csv")
    val = pd.read_csv(PROC / "validation.csv")
    from build_features import ALL_FEATURES
    numeric = ALL_FEATURES

    # Keep all 158 neighbourhoods and keep full chronological months in splits.
    assert train["forecast_month"].max() < val["forecast_month"].min()
    assert train[TARGET].notna().all() and val[TARGET].notna().all()
    assert not train[numeric].isna().any().any()
    assert not val[numeric].isna().any().any()

    selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(
        X, y, random_state=42
    ), k=10).fit(train[numeric], train[TARGET])
    mi = selector.scores_

    importance_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=35,
        random_state=42, n_jobs=2
    )
    importance_model.fit(train[numeric], train[TARGET])
    rankings = pd.DataFrame({
        "feature": numeric,
        "mutual_information": mi,
        "random_forest_importance": importance_model.feature_importances_,
    })
    rankings["mi_rank"] = rankings["mutual_information"].rank(
        ascending=False, method="min"
    ).astype(int)
    rankings["forest_rank"] = rankings["random_forest_importance"].rank(
        ascending=False, method="min"
    ).astype(int)
    rankings = rankings.sort_values(["mi_rank","feature"])

    count_cols = ["last1", "mean3", "mean12", "same_month_ly", "trend"]
    relative_cols = [
        "last1_pct_rank", "mean3_pct_rank", "mean12_pct_rank", "same_month_ly_pct_rank"
    ]
    sets = {
        "counts_only": count_cols,
        "counts_plus_relative": count_cols + relative_cols,
        "all_features": numeric,
        "mutual_information_top10": rankings.nsmallest(10, "mi_rank")["feature"].tolist(),
        "forest_importance_top10": rankings.nsmallest(10, "forest_rank")["feature"].tolist(),
    }
    runs = []
    for group_name, columns in sets.items():
        for model_name, clf in [
            ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
            ("Decision Tree", DecisionTreeClassifier(
                max_depth=7, min_samples_leaf=55, random_state=42
            )),
        ]:
            preprocess = ColumnTransformer([
                ("numeric", StandardScaler(), columns),
                ("neighbourhood", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ), ["HOOD_158"]),
            ])
            fitted = Pipeline([
                ("preprocessor", preprocess), ("classifier", clf),
            ])
            input_columns = columns + ["HOOD_158"]
            fitted.fit(train[input_columns], train[TARGET])
            proba = fitted.predict_proba(val[input_columns])[:, 1]
            pred = (proba >= 0.5).astype(int)
            runs.append({
                "feature_set": group_name, "model": model_name,
                "number_of_numeric_features": len(columns),
                "validation_f1": float(f1_score(val[TARGET], pred)),
                "validation_auc": float(roc_auc_score(val[TARGET], proba)),
                "features": ", ".join(columns),
            })
    result = pd.DataFrame(runs).sort_values(["feature_set", "model"])
    print("\n--- TRAINING-ONLY FEATURE SELECTION ---")
    print(rankings.to_string(index=False))
    print("\n--- CHRONOLOGICAL VALIDATION FEATURE-SET COMPARISON ---")
    print(result.drop(columns=["features"]).round(4).to_string(index=False))
    return rankings, result



def compare_start_years():
    """Validation-only 2015-versus-2018 comparison for Sean/Emine."""
    from build_features import ALL_FEATURES
    train = pd.read_csv(PROC / "train.csv")
    val = pd.read_csv(PROC / "validation.csv")
    cols = ALL_FEATURES + ["HOOD_158"]
    results = []
    for first_year in (2015, 2018):
        period = train.loc[train["forecast_month"] >= f"{first_year}-01"].copy()
        n = period[TARGET].value_counts().min()
        balanced = period.groupby(TARGET, group_keys=False).sample(n=n, random_state=42)
        model = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("numeric", StandardScaler(), ALL_FEATURES),
                ("neighbourhood", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["HOOD_158"]),
            ])),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        model.fit(balanced[cols], balanced[TARGET])
        scores = model.predict_proba(val[cols])[:, 1]
        predictions = (scores >= 0.5).astype(int)
        results.append({
            "start_year": first_year,
            "training_rows": len(period),
            "validation_f1": round(float(f1_score(val[TARGET], predictions)), 4),
            "validation_auc": round(float(roc_auc_score(val[TARGET], scores)), 4),
        })
    print("\n--- TRAINING FROM 2015 VS 2018 ---")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    compare()
    compare_start_years()
