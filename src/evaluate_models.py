"""Section 6: train, tune, and evaluate hotspot classifiers.

The chronological splits in train.csv, validation.csv, and test.csv are
Sean's. This script does not rebuild them and does not look at the test
set until the model, the feature set, and the probability threshold are
already chosen.

Selection rule, fixed before any test score is computed:

1. Fit every configuration on the training months only.
2. Keep configurations whose validation F1 is within 0.01 of the best
   validation F1. The threshold behind that F1 is the one that maximises
   F1 on the validation set.
3. From that shortlist, pick the highest validation F1 on the 73
   neighbourhoods that swing. Those are the neighbourhoods whose hotspot
   rate, over forecast months February 2015 to September 2026, is strictly
   between 20% and 80%.
4. Break remaining ties by validation ROC AUC, then fewer features, then
   a model that does not use neighbourhood identity.

The same rule is applied inside each algorithm so the ROC chart can show
one validation-chosen model per algorithm. The deployed model is the
winner of the global shortlist.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import joblib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "hotspot_next_month"
COUNT_FEATURES = ["last1", "mean3", "mean12", "same_month_ly", "trend"]
RELATIVE_FEATURES = [
    "last1_pct_rank",
    "mean3_pct_rank",
    "mean12_pct_rank",
    "same_month_ly_pct_rank",
]
EXTRA_FEATURES = [
    "house_share12",
    "outside_share12",
    "night_share12",
    "month_sin",
    "month_cos",
    "city_total3",
    "previous_hotspot",
    "hotspot_rate12",
    "last_count_rank",
]
ALL_FEATURES = COUNT_FEATURES + RELATIVE_FEATURES + EXTRA_FEATURES
LEAKAGE_COLUMNS = {"hotspot_next_month", "next_month_theft_count", "next_month_cutoff"}
RAW_URL = "https://raw.githubusercontent.com/seangowans32/COMP-309-Group-Project-2/main/data/{name}"

# Shortlist width is part of the locked rule above.
F1_BAND = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Directory with train.csv, validation.csv, and test.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Where to write figures/ and results/.",
    )
    return parser.parse_args()


def ensure_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = data_dir / name
        if path.exists() and path.stat().st_size > 0:
            continue
        url = RAW_URL.format(name=name)
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    for name, frame, rows in (
        ("train", train, 13114),
        ("validation", val, 3792),
        ("test", test, 5214),
    ):
        if len(frame) != rows:
            raise AssertionError(f"{name} has {len(frame)} rows, expected {rows}")
        if frame[TARGET].isna().any():
            raise AssertionError(f"{name} has missing labels")
        if frame[ALL_FEATURES].isna().any().any():
            raise AssertionError(f"{name} has missing features")
    if train["forecast_month"].max() >= val["forecast_month"].min():
        raise AssertionError("Training months are not strictly before validation")
    if val["forecast_month"].max() >= test["forecast_month"].min():
        raise AssertionError("Validation months are not strictly before the test")
    leaked = LEAKAGE_COLUMNS.intersection(ALL_FEATURES)
    if leaked:
        raise AssertionError(f"Label columns listed as features: {sorted(leaked)}")
    return train, val, test


def swing_neighbourhoods(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.Index:
    """Match section 3: 11 chronic, 74 rare, 73 in between.

    The rate uses every labelled forecast month (February 2015 to September
    2026), which is the period Fatima's 11 / 74 / 73 split comes from.
    """
    labelled = pd.concat([train, val, test], ignore_index=True)
    rates = labelled.groupby("HOOD_158")[TARGET].mean()
    chronic = int((rates >= 0.80).sum())
    rare = int((rates <= 0.20).sum())
    swing = rates[(rates > 0.20) & (rates < 0.80)]
    if chronic != 11 or rare != 74 or len(swing) != 73:
        raise AssertionError(
            f"Expected 11 / 74 / 73 neighbourhood groups, got {chronic} / {rare} / {len(swing)}"
        )
    return swing.index


def mutual_info_top10(train: pd.DataFrame) -> list[str]:
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(
        train[ALL_FEATURES], train[TARGET], random_state=RANDOM_STATE
    )
    order = pd.Series(scores, index=ALL_FEATURES).sort_values(ascending=False)
    print("\nMutual information rank (training rows only)", flush=True)
    for rank, (feature, score) in enumerate(order.items(), start=1):
        print(f"  {rank:2d}. {feature:<24} {score:.4f}", flush=True)
    return order.head(10).index.tolist()


def feature_sets(mi_top10: list[str]) -> dict[str, list[str]]:
    return {
        "counts_plus_ranks": COUNT_FEATURES + RELATIVE_FEATURES,
        "all_18": list(ALL_FEATURES),
        "mutual_info_top10": list(mi_top10),
    }


def configurations(sets: dict[str, list[str]]) -> list[dict]:
    """Small grid. Every value here is chosen on validation, never on test."""
    configs: list[dict] = []
    for set_name, columns in sets.items():
        for use_neighbourhood in (False, True):
            for C in (0.1, 1.0, 10.0):
                configs.append(
                    {
                        "algorithm": "Logistic Regression",
                        "feature_set": set_name,
                        "features": columns,
                        "use_neighbourhood": use_neighbourhood,
                        "params": {"C": C, "class_weight": "balanced"},
                        "params_tried": "C in {0.1, 1, 10}; class_weight balanced",
                    }
                )
            for max_depth, min_leaf in ((8, 20), (16, 10), (None, 30)):
                configs.append(
                    {
                        "algorithm": "Random Forest",
                        "feature_set": set_name,
                        "features": columns,
                        "use_neighbourhood": use_neighbourhood,
                        "params": {
                            "n_estimators": 200,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "max_features": "sqrt",
                            "class_weight": "balanced_subsample",
                        },
                        "params_tried": (
                            "n_estimators 200; max_depth in {8, 16, unlimited}; "
                            "min_samples_leaf in {10, 20, 30}; class_weight balanced_subsample"
                        ),
                    }
                )
            for learning_rate, max_depth, min_leaf in (
                (0.05, 3, 40),
                (0.1, 3, 20),
                (0.05, 6, 20),
            ):
                configs.append(
                    {
                        "algorithm": "Gradient boosting",
                        "feature_set": set_name,
                        "features": columns,
                        "use_neighbourhood": use_neighbourhood,
                        "params": {
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "max_iter": 250,
                            "l2_regularization": 0.1,
                            "class_weight": "balanced",
                        },
                        "params_tried": (
                            "learning_rate in {0.05, 0.1}; max_depth in {3, 6}; "
                            "min_samples_leaf in {20, 40}; max_iter 250; "
                            "l2_regularization 0.1; class_weight balanced"
                        ),
                    }
                )
    return configs


def build_estimator(config: dict):
    algorithm = config["algorithm"]
    params = config["params"]
    if algorithm == "Logistic Regression":
        model = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif algorithm == "Gradient boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_iter=params["max_iter"],
            l2_regularization=params["l2_regularization"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(algorithm)
    transformers = [("numeric", StandardScaler(), config["features"])]
    if config["use_neighbourhood"]:
        transformers.append(
            (
                "neighbourhood",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["HOOD_158"],
            )
        )
    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            ("classifier", model),
        ]
    )
    columns = list(config["features"])
    if config["use_neighbourhood"]:
        columns.append("HOOD_158")
    return pipeline, columns


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (proba >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(threshold)
    return best_t, best_f1


def safe_auc(y_true, scores) -> float:
    scores = np.asarray(scores, dtype=float)
    if np.unique(scores).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def classification_metrics(y_true, predictions, scores) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": safe_auc(y_true, scores),
    }


def subset_metrics(frame: pd.DataFrame, mask: np.ndarray, predictions, scores) -> dict[str, float]:
    metrics = classification_metrics(frame.loc[mask, TARGET], predictions[mask], scores[mask])
    metrics["rows"] = int(mask.sum())
    metrics["hotspot_share"] = float(frame.loc[mask, TARGET].mean())
    return metrics


def baseline_bundle(frame: pd.DataFrame, swing_ids: pd.Index) -> dict:
    y = frame[TARGET]
    swing = frame["HOOD_158"].isin(swing_ids).to_numpy()
    always_pred = np.zeros(len(frame), dtype=int)
    always_score = np.zeros(len(frame), dtype=float)
    persist_pred = frame["previous_hotspot"].astype(int).to_numpy()
    rank_score = frame["mean12_pct_rank"].to_numpy(dtype=float)
    rank_pred = (rank_score >= 0.75).astype(int)
    out = {}
    for name, pred, score in (
        ("always_no", always_pred, always_score),
        ("persistence", persist_pred, persist_pred.astype(float)),
        ("mean12_top25", rank_pred, rank_score),
    ):
        out[name] = {
            "overall": classification_metrics(y, pred, score),
            "swing": subset_metrics(frame, swing, pred, score),
        }
    return out


def selection_key(row: dict) -> tuple:
    return (
        row["swing_val_f1"],
        row["val_auc"],
        -row["n_features"],
        0 if not row["use_neighbourhood"] else -1,
    )


def choose(rows: list[dict]) -> dict:
    best_f1 = max(row["val_f1"] for row in rows)
    shortlist = [row for row in rows if row["val_f1"] >= best_f1 - F1_BAND]
    return max(shortlist, key=selection_key)


def format_params(config: dict, threshold: float) -> str:
    parts = [f"{key}={value}" for key, value in config["params"].items()]
    hood = "with neighbourhood id" if config["use_neighbourhood"] else "no neighbourhood id"
    parts.append(f"features={config['feature_set']}")
    parts.append(hood)
    parts.append(f"threshold={threshold:.2f}")
    return "; ".join(parts)


def fit_config(config: dict, train: pd.DataFrame, val: pd.DataFrame, swing_ids: pd.Index) -> dict:
    pipeline, columns = build_estimator(config)
    pipeline.fit(train[columns], train[TARGET])
    proba = pipeline.predict_proba(val[columns])[:, 1]
    threshold, val_f1 = best_threshold(val[TARGET], proba)
    predictions = (proba >= threshold).astype(int)
    swing = val["HOOD_158"].isin(swing_ids).to_numpy()
    swing_f1 = float(f1_score(val.loc[swing, TARGET], predictions[swing], zero_division=0))
    return {
        "algorithm": config["algorithm"],
        "feature_set": config["feature_set"],
        "features": list(config["features"]),
        "use_neighbourhood": config["use_neighbourhood"],
        "params": config["params"],
        "params_tried": config["params_tried"],
        "n_features": len(config["features"]),
        "threshold": threshold,
        "val_f1": val_f1,
        "val_auc": safe_auc(val[TARGET], proba),
        "swing_val_f1": swing_f1,
        "val_metrics": classification_metrics(val[TARGET], predictions, proba),
    }


def refit_and_score(
    chosen: dict,
    train: pd.DataFrame,
    frame: pd.DataFrame,
    swing_ids: pd.Index,
) -> tuple[Pipeline, np.ndarray, np.ndarray, dict]:
    config = {
        "algorithm": chosen["algorithm"],
        "feature_set": chosen["feature_set"],
        "features": chosen["features"],
        "use_neighbourhood": chosen["use_neighbourhood"],
        "params": chosen["params"],
    }
    pipeline, columns = build_estimator(config)
    pipeline.fit(train[columns], train[TARGET])
    proba = pipeline.predict_proba(frame[columns])[:, 1]
    predictions = (proba >= chosen["threshold"]).astype(int)
    swing = frame["HOOD_158"].isin(swing_ids).to_numpy()
    scored = {
        "overall": classification_metrics(frame[TARGET], predictions, proba),
        "swing": subset_metrics(frame, swing, predictions, proba),
        "confusion": [int(v) for v in confusion_matrix(frame[TARGET], predictions).ravel()],
    }
    return pipeline, predictions, proba, scored


def rounded(metrics: dict) -> dict:
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in metrics.items()}


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Not a hotspot", "Hotspot"],
        yticklabels=["Not a hotspot", "Hotspot"],
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa0a6", linewidth=1, label="Chance (AUC 0.50)")
    for label, y_true, scores in curves:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = safe_auc(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{label} (AUC {auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test set ROC curves, January 2024 to September 2026")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coefficient_table(pipeline: Pipeline, chosen: dict, val: pd.DataFrame) -> list[dict]:
    """Shuffle each input on the validation set and record the drop in ROC AUC.

    This describes the chosen model. It is not used to change the model,
    and it does not touch the test set.
    """
    columns = list(chosen["features"])
    if chosen["use_neighbourhood"]:
        columns.append("HOOD_158")
    base = pipeline.predict_proba(val[columns])[:, 1]
    base_auc = safe_auc(val[TARGET], base)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for feature in columns:
        shuffled = val[columns].copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        dropped = base_auc - safe_auc(val[TARGET], pipeline.predict_proba(shuffled)[:, 1])
        rows.append(
            {
                "feature": feature,
                "auc_drop": round(float(dropped), 4),
                "kind": "validation permutation drop in ROC AUC",
            }
        )
    rows.sort(key=lambda row: row["auc_drop"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    data_dir = args.data.resolve() if args.data else outdir / "data"
    ensure_data(data_dir)
    figures = outdir / "figures"
    results_dir = outdir / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(data_dir)
    swing_ids = swing_neighbourhoods(train, val, test)
    print(
        f"Rows train/val/test {len(train)}/{len(val)}/{len(test)}; swing neighbourhoods {len(swing_ids)}",
        flush=True,
    )
    mi_top10 = mutual_info_top10(train)
    sets = feature_sets(mi_top10)
    configs = configurations(sets)
    print(f"Fitting {len(configs)} configurations on training data only", flush=True)

    rows = []
    started = time.time()
    for index, config in enumerate(configs, start=1):
        t0 = time.time()
        row = fit_config(config, train, val, swing_ids)
        rows.append(row)
        print(
            f"[{index:02d}/{len(configs)}] {row['algorithm']:<22} {row['feature_set']:<20} "
            f"hood={int(row['use_neighbourhood'])} F1={row['val_f1']:.4f} "
            f"AUC={row['val_auc']:.4f} swingF1={row['swing_val_f1']:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    print(f"Search finished in {time.time() - started:.0f}s", flush=True)

    chosen = choose(rows)
    per_algorithm = []
    for algorithm in ("Logistic Regression", "Random Forest", "Gradient boosting"):
        per_algorithm.append(choose([row for row in rows if row["algorithm"] == algorithm]))

    print("\nChosen on validation only:", chosen["algorithm"], format_params(
        {"params": chosen["params"], "feature_set": chosen["feature_set"], "use_neighbourhood": chosen["use_neighbourhood"]},
        chosen["threshold"],
    ), flush=True)

    baselines = {
        "validation": baseline_bundle(val, swing_ids),
        "test": baseline_bundle(test, swing_ids),
    }

    def same_config(left: dict, right: dict) -> bool:
        return (
            left["algorithm"] == right["algorithm"]
            and left["feature_set"] == right["feature_set"]
            and left["use_neighbourhood"] == right["use_neighbourhood"]
            and left["params"] == right["params"]
            and left["threshold"] == right["threshold"]
        )

    # One validation-chosen model per algorithm, plus the global winner when
    # the within-algorithm rule picks a different configuration.
    to_score = list(per_algorithm)
    if not any(same_config(candidate, chosen) for candidate in to_score):
        to_score.append(chosen)

    test_models = []
    curves = []
    # Persistence and the 12-month rank are scores, so they belong on the ROC chart.
    curves.append(("Persistence rule", test[TARGET].to_numpy(), test["previous_hotspot"].to_numpy(dtype=float)))
    curves.append(("12-month rank", test[TARGET].to_numpy(), test["mean12_pct_rank"].to_numpy(dtype=float)))

    for candidate in to_score:
        pipeline, predictions, proba, scored = refit_and_score(candidate, train, test, swing_ids)
        is_chosen = same_config(candidate, chosen)
        label = candidate["algorithm"] + (" (chosen)" if is_chosen else "")
        curves.append((label, test[TARGET].to_numpy(), proba))
        scored_public = {
            "overall": rounded(scored["overall"]),
            "swing": rounded(scored["swing"]),
            "confusion": {
                "tn": scored["confusion"][0],
                "fp": scored["confusion"][1],
                "fn": scored["confusion"][2],
                "tp": scored["confusion"][3],
            },
        }
        entry = {
            "algorithm": candidate["algorithm"],
            "feature_set": candidate["feature_set"],
            "features": candidate["features"],
            "use_neighbourhood": candidate["use_neighbourhood"],
            "params": candidate["params"],
            "params_tried": candidate["params_tried"],
            "threshold": candidate["threshold"],
            "validation": rounded(candidate["val_metrics"]),
            "validation_swing_f1": round(candidate["swing_val_f1"], 4),
            "chosen": is_chosen,
            "test": scored_public,
        }
        if is_chosen:
            entry["importances"] = coefficient_table(pipeline, candidate, val)
            model_dir = outdir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_dir / "hotspot_model.joblib")
            (model_dir / "hotspot_model.json").write_text(
                json.dumps(
                    {
                        "algorithm": candidate["algorithm"],
                        "feature_set": candidate["feature_set"],
                        "features": candidate["features"],
                        "use_neighbourhood": candidate["use_neighbourhood"],
                        "params": candidate["params"],
                        "threshold": round(candidate["threshold"], 2),
                        "label": TARGET,
                        "decision": "Predict hotspot when the model score is at or above the threshold.",
                        "trained_on": "train.csv only (forecast months February 2015 to December 2021)",
                    },
                    indent=2,
                )
            )
            cm = scored["confusion"]
            plot_confusion(
                np.array([[cm[0], cm[1]], [cm[2], cm[3]]]),
                f"{candidate['algorithm']} on the test set\nthreshold {candidate['threshold']:.2f}",
                figures / "figure_6_1_confusion_matrix.png",
            )
        test_models.append(entry)

    plot_roc(curves, figures / "figure_6_2_roc_curves.png")

    def public_row(row: dict) -> dict:
        return {
            "algorithm": row["algorithm"],
            "feature_set": row["feature_set"],
            "use_neighbourhood": row["use_neighbourhood"],
            "params": row["params"],
            "threshold": round(row["threshold"], 2),
            "n_features": row["n_features"],
            "val_f1": round(row["val_f1"], 4),
            "val_auc": round(row["val_auc"], 4),
            "swing_val_f1": round(row["swing_val_f1"], 4),
            "val_metrics": rounded(row["val_metrics"]),
        }

    leaderboard = sorted(rows, key=lambda row: (row["val_f1"], row["val_auc"]), reverse=True)
    payload = {
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "splits": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_hotspot_share": round(float(train[TARGET].mean()), 4),
            "validation_hotspot_share": round(float(val[TARGET].mean()), 4),
            "test_hotspot_share": round(float(test[TARGET].mean()), 4),
            "train_months": [train["forecast_month"].min(), train["forecast_month"].max()],
            "validation_months": [val["forecast_month"].min(), val["forecast_month"].max()],
            "test_months": [test["forecast_month"].min(), test["forecast_month"].max()],
        },
        "swing_definition": (
            "Forecast months February 2015 to September 2026. "
            "A neighbourhood swings when its hotspot share is greater than 0.20 and less than 0.80. "
            "11 neighbourhoods are at or above 0.80 and 74 are at or below 0.20."
        ),
        "swing_hood_ids": [int(value) for value in swing_ids.tolist()],
        "mutual_info_top10": mi_top10,
        "feature_sets": {name: columns for name, columns in sets.items()},
        "selection_rule": (
            "Shortlist configs within 0.01 validation F1 of the best. "
            "Pick the highest swing-neighbourhood validation F1. "
            "Tie-break with validation ROC AUC, then fewer features, then no neighbourhood id."
        ),
        "baselines": baselines,
        "leaderboard": [public_row(row) for row in leaderboard],
        "per_algorithm": test_models,
        "chosen_algorithm": chosen["algorithm"],
    }
    # Baseline metrics were raw floats. Round a copy for the file.
    payload["baselines"] = json.loads(json.dumps(baselines), parse_float=lambda s: round(float(s), 4))
    (results_dir / "section6_results.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {results_dir / 'section6_results.json'}", flush=True)
    chosen_entries = [entry for entry in test_models if entry["chosen"]]
    if len(chosen_entries) != 1:
        raise AssertionError(f"Expected one chosen model, found {len(chosen_entries)}")


if __name__ == "__main__":
    main()
