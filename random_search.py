import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from automl import compute_dataset_profile_xy

# ===================== Helpers from project 1 =====================

class TargetLabelEncoder:
    def __init__(self):
        self._le = LabelEncoder()
        self.classes_ = None

    def fit(self, y):
        y = pd.Series(y) if not isinstance(y, (pd.Series, pd.Index)) else y
        self._le.fit(y.astype(str) if y.dtype == "object" else y)
        self.classes_ = list(self._le.classes_)
        return self

    def transform(self, y):
        y = pd.Series(y) if not isinstance(y, (pd.Series, pd.Index)) else y
        return self._le.transform(y.astype(str) if y.dtype == "object" else y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        return self._le.inverse_transform(np.asarray(y_encoded))


def split_xy(df: pd.DataFrame, target: str):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def to_serializable(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def read_dataset(csv_path: Path, na_values: Optional[str]) -> pd.DataFrame:
    read_kwargs: Dict[str, Any] = {}
    if na_values:
        read_kwargs["na_values"] = [v for v in na_values.split(",") if v]
    return pd.read_csv(csv_path, **read_kwargs)

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.int_cols_ = X.select_dtypes(include=[np.integer]).columns.tolist()
        self.float_cols_ = X.select_dtypes(include=[np.floating]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        self.medians_ = X[self.int_cols_].median(numeric_only=True)
        self.means_ = X[self.float_cols_].mean(numeric_only=True)
        return self

    def transform(self, X):
        X = X.copy()
        if self.int_cols_:
            X[self.int_cols_] = X[self.int_cols_].fillna(self.medians_)
        if self.float_cols_:
            X[self.float_cols_] = X[self.float_cols_].fillna(self.means_)
        if self.cat_cols_:
            X[self.cat_cols_] = X[self.cat_cols_].fillna("(NA)").astype("string")
        return X


def build_two_stage_preprocessor():
    stage1 = MissingValueHandler()
    enc_scale = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                make_column_selector(dtype_include=[np.number]),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                make_column_selector(
                    dtype_include=["object", "string", "category"]
                ),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        [("stage1_missing", stage1), ("stage2_encode_scale", enc_scale)]
    )


# ===================== Model factory & grids =====================

def build_model(model_type: str, random_state: int) -> Any:
    if model_type == "sklearn.linear_model.LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs")

    if model_type == "sklearn.svm.LinearSVC":
        return LinearSVC(random_state=random_state, max_iter=5000)

    if model_type == "sklearn.linear_model.SGDClassifier":
        return SGDClassifier(random_state=random_state, max_iter=2000, tol=1e-3)

    if model_type == "sklearn.ensemble.RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "sklearn.ensemble.ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")

def load_param_grid(model_type: str, grid_path: Path) -> Dict[str, Any]:
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as f:
        grids_all = json.load(f)

    if model_type not in grids_all:
        raise KeyError(f"model_type={model_type} not present in {grid_path}")

    grid_raw = grids_all[model_type]
    grid: Dict[str, Any] = {}
    for param_name, values in grid_raw.items():
        # JSON null -> Python None is already handled by json.load,
        # but we keep the normalization explicit.
        grid[param_name] = [(None if v is None else v) for v in values]
    return grid


def count_grid_combinations(param_grid: Dict[str, Any]) -> int:
    if not param_grid:
        return 1
    lengths = []
    for _, vals in param_grid.items():
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            lengths.append(1)
        else:
            lengths.append(len(vals))
    return int(math.prod(lengths))

# ===================== Dataset profiling =====================

# def compute_dataset_profile(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
#     X, y = split_xy(df, target_col)
#     n_samples, n_features = X.shape

#     num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_cols = X.select_dtypes(
#         include=["object", "string", "category"]
#     ).columns.tolist()

#     missing_per_col = X.isna().mean()
#     missing_overall = float(missing_per_col.mean()) if len(missing_per_col) else 0.0
#     missing_max = float(missing_per_col.max()) if len(missing_per_col) else 0.0

#     profile: Dict[str, Any] = {
#         "n_samples": int(n_samples),
#         "n_features": int(n_features),
#         "n_num_features": int(len(num_cols)),
#         "n_cat_features": int(len(cat_cols)),
#         "missing_fraction_mean": missing_overall,
#         "missing_fraction_max": missing_max,
#     }

#     # class distribution
#     vc = y.value_counts(dropna=False)
#     profile["class_counts"] = {
#         str(k): int(v) for k, v in vc.to_dict().items()
#     }
#     profile["class_proportions"] = {
#         str(k): float(v) / float(len(y)) for k, v in vc.to_dict().items()
#     }

#     # cardinality of categorical features
#     if cat_cols:
#         card = [X[c].nunique(dropna=True) for c in cat_cols]
#         profile["cat_cardinality_min"] = int(np.min(card))
#         profile["cat_cardinality_median"] = float(np.median(card))
#         profile["cat_cardinality_max"] = int(np.max(card))
#     else:
#         profile["cat_cardinality_min"] = 0
#         profile["cat_cardinality_median"] = 0.0
#         profile["cat_cardinality_max"] = 0

#     # prosta korelacja cech numerycznych z targetem (zakodowanym liczbowo)
#     if num_cols:
#         le = LabelEncoder().fit(y.astype(str))
#         y_enc = pd.Series(le.transform(y.astype(str)), index=y.index)
#         corr = X[num_cols].corrwith(y_enc)
#         corr = corr.dropna()
#         if len(corr):
#             profile["num_abs_corr_mean"] = float(corr.abs().mean())
#             profile["num_abs_corr_max"] = float(corr.abs().max())
#         else:
#             profile["num_abs_corr_mean"] = 0.0
#             profile["num_abs_corr_max"] = 0.0
#     else:
#         profile["num_abs_corr_mean"] = 0.0
#         profile["num_abs_corr_max"] = 0.0

#     return profile


def stratified_sample_indices(y: pd.Series, max_rows: int, random_state: int) -> np.ndarray:
    n = len(y)
    if n <= max_rows:
        return np.arange(n, dtype=int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_rows,
        random_state=random_state,
    )
    idx_train, _ = next(splitter.split(np.zeros((n, 1)), y))
    return np.asarray(idx_train, dtype=int)


# ===================== random search =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-friendly RandomizedSearchCV screening for a single (model_type, dataset)."
    )
    parser.add_argument("--model_type", required=True, help="Model type (logreg, linear_svc, sgd, rf, extratrees)")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (used in metadata/logging)")
    parser.add_argument("--target_col", required=True, help="Target column name")
    parser.add_argument("--output_path", required=True, help="Output directory (summary.json + cv_results.csv)")
    parser.add_argument("--dataset_path", default=None, help="Optional path to CSV. Default: ./data/{dataset_name}.csv")
    parser.add_argument("--na_values", default=None, help="Optional NA values (e.g. 'na,NA,?')")
    parser.add_argument("--grid_path", default="./config/model_type_grid.json", help="Path to model grid JSON")

    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)

    # fixed-by-spec knobs (still exposed for convenience)
    parser.add_argument("--cv_folds", type=int, default=3, help="Fixed to 3 by spec (will be enforced).")
    parser.add_argument("--cv_max_rows", type=int, default=9000, help="Max rows used for CV search sampling.")
    parser.add_argument("--n_iter_cap", type=int, default=20, help="Max random search iterations cap (default 20).")

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage: load dataset
    dataset_path = Path(args.dataset_path) if args.dataset_path else (Path("./data") / f"{args.dataset_name}.csv")
    df = read_dataset(dataset_path, args.na_values)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {dataset_path}")

    dataset_profile = compute_dataset_profile_xy(df.drop(columns=[args.target_col]), df[args.target_col])
    X_full, y_full = split_xy(df, args.target_col)

    # Stage: sampling for CV search (<= 9000 rows)
    cv_folds = 3  # enforced
    max_rows = int(args.cv_max_rows)
    sample_idx = stratified_sample_indices(y_full, max_rows=max_rows, random_state=args.random_state)

    X = X_full.iloc[sample_idx].reset_index(drop=True)
    y = y_full.iloc[sample_idx].reset_index(drop=True)

    # Stage: label encoding
    label_enc = TargetLabelEncoder().fit(y)
    y_enc = label_enc.transform(y)

    # Stage: pipeline
    pipe = Pipeline(
        [
            ("prep", build_two_stage_preprocessor()),
            ("model", build_model(args.model_type, random_state=args.random_state)),
        ]
    )

    # Stage: grid + n_iter rule
    grid_path = Path(args.grid_path)
    param_distributions = load_param_grid(args.model_type, grid_path)

    n_combinations = count_grid_combinations(param_distributions)
    n_iter = min(int(n_combinations), int(args.n_iter_cap))

    # Stage: scoring + CV
    scoring = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "accuracy": make_scorer(accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="balanced_accuracy",
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=args.verbose,
        random_state=args.random_state,
        return_train_score=False,
    )

    # Stage: fit search

    print("Start")
    search.fit(X, y_enc)
    print("Finished")
    # Stage: save cv results
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    # Stage: derive top/worst configs
    cv_sorted = cv_df.sort_values("rank_test_balanced_accuracy")
    top_k = 5

    top_configs = []
    for _, row in cv_sorted.head(top_k).iterrows():
        top_configs.append(
            {
                "rank": int(row["rank_test_balanced_accuracy"]),
                "mean_balanced_accuracy": float(row["mean_test_balanced_accuracy"]),
                "std_balanced_accuracy": float(row["std_test_balanced_accuracy"]),
                "params": row["params"],
            }
        )

    worst_row = cv_sorted.sort_values("rank_test_balanced_accuracy", ascending=False).iloc[0]
    worst_config = {
        "rank": int(worst_row["rank_test_balanced_accuracy"]),
        "mean_balanced_accuracy": float(worst_row["mean_test_balanced_accuracy"]),
        "std_balanced_accuracy": float(worst_row["std_test_balanced_accuracy"]),
        "params": worst_row["params"],
    }

    # Stage: metrics on the same sample used for search/refit (consistent & fast)
    best_est = search.best_estimator_
    y_pred_enc = best_est.predict(X)
    y_pred = label_enc.inverse_transform(y_pred_enc)

    metrics_train = {
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, y_pred_enc)),
        "accuracy": float(accuracy_score(y_enc, y_pred_enc)),
        "f1_macro": float(f1_score(y_enc, y_pred_enc, average="macro")),
    }

    cls_report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=label_enc.classes_)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path),
        "target_col": args.target_col,
        "model_type": args.model_type,
        "label_classes": label_enc.classes_,
        "dataset_profile": dataset_profile,
        "search_sample": {
            "n_rows_used": int(len(X)),
            "cv_max_rows": int(max_rows),
            "sampling": "stratified",
        },
        "search_settings": {
            "grid_path": str(grid_path),
            "n_combinations_in_grid": int(n_combinations),
            "n_iter_used": int(n_iter),
            "cv_folds": int(cv_folds),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "cv_best_index": int(search.best_index_),
        "cv_best_params": search.best_params_,
        "cv_best_balanced_accuracy": float(search.best_score_),
        "top_configs": top_configs,
        "worst_config": worst_config,
        "metrics_train": metrics_train,
        "classification_report_train": cls_report,
        "confusion_matrix_train": cm.tolist(),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)

    print(
        f"Saved results to {out_dir} (cv_results.csv, summary.json). "
        f"best_balanced_accuracy_cv={summary['cv_best_balanced_accuracy']:.4f} "
        f"(n_iter={n_iter}, cv={cv_folds}, rows={len(X)})"
    )


if __name__ == "__main__":
    main()
