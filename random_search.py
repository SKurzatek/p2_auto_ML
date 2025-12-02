#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# opcjonalne biblioteki GPU
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# ===================== Helpers from poprzedniego projektu =====================

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

def build_model(model_type: str, use_gpu: bool, random_state: int = 42):
    """Tworzy obiekt klasyfikatora dla danego model_type."""

    if model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            random_state=random_state,
        )

    if model_type == "xgb":
        if XGBClassifier is None:
            raise RuntimeError(
                "XGBoost (xgboost) nie jest zainstalowany, a model_type=xgb."
            )

        base_params = dict(
            eval_metric="logloss",
            n_estimators=300,
            random_state=random_state,
        )

        if use_gpu:
            # XGBoost 3.x: GPU = tree_method="hist" + device="cuda"
            return XGBClassifier(
                tree_method="hist",
                device="cuda",
                **base_params,
            )
        else:
            return XGBClassifier(
                tree_method="hist",
                n_jobs=-1,
                **base_params,
            )

    if model_type == "catboost_gpu":
        if CatBoostClassifier is None:
            raise RuntimeError(
                "CatBoost nie jest zainstalowany, a model_type=catboost_gpu."
            )
        # zawsze GPU - to jest sens tej wersji
        return CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            task_type="GPU",
            eval_metric="Logloss",
            verbose=False,
            random_seed=random_state,
        )

    if model_type == "svc_rbf":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
        )

    if model_type == "knn":
        return KNeighborsClassifier()

    raise ValueError(f"Nieznany model_type: {model_type}")


def load_param_grid(
    model_type: str, grid_path: Path = Path("./config/model_type_grid.json")
) -> Dict[str, Any]:
    """Wczytuje siatkę hiperparametrów z JSON-a dla danego model_type."""
    if not grid_path.exists():
        raise FileNotFoundError(f"Brak pliku z gridami: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as f:
        grids_all = json.load(f)

    if model_type not in grids_all:
        raise KeyError(
            f"Brak wpisu z gridem dla model_type={model_type} w {grid_path}"
        )

    grid_raw = grids_all[model_type]
    grid = {}
    for param_name, values in grid_raw.items():
        # zamień null -> None
        new_vals = [
            (None if v is None else v) for v in values
        ]
        grid[param_name] = new_vals

    return grid


# ===================== Dataset profiling (metadane) =====================

def compute_dataset_profile(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    X, y = split_xy(df, target_col)
    n_samples, n_features = X.shape

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    missing_per_col = X.isna().mean()
    missing_overall = float(missing_per_col.mean()) if len(missing_per_col) else 0.0
    missing_max = float(missing_per_col.max()) if len(missing_per_col) else 0.0

    profile: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),
        "missing_fraction_mean": missing_overall,
        "missing_fraction_max": missing_max,
    }

    # class distribution
    vc = y.value_counts(dropna=False)
    profile["class_counts"] = {
        str(k): int(v) for k, v in vc.to_dict().items()
    }
    profile["class_proportions"] = {
        str(k): float(v) / float(len(y)) for k, v in vc.to_dict().items()
    }

    # cardinality of categorical features
    if cat_cols:
        card = [X[c].nunique(dropna=True) for c in cat_cols]
        profile["cat_cardinality_min"] = int(np.min(card))
        profile["cat_cardinality_median"] = float(np.median(card))
        profile["cat_cardinality_max"] = int(np.max(card))
    else:
        profile["cat_cardinality_min"] = 0
        profile["cat_cardinality_median"] = 0.0
        profile["cat_cardinality_max"] = 0

    # prosta korelacja cech numerycznych z targetem (zakodowanym liczbowo)
    if num_cols:
        le = LabelEncoder().fit(y.astype(str))
        y_enc = pd.Series(le.transform(y.astype(str)), index=y.index)
        corr = X[num_cols].corrwith(y_enc)
        corr = corr.dropna()
        if len(corr):
            profile["num_abs_corr_mean"] = float(corr.abs().mean())
            profile["num_abs_corr_max"] = float(corr.abs().max())
        else:
            profile["num_abs_corr_mean"] = 0.0
            profile["num_abs_corr_max"] = 0.0
    else:
        profile["num_abs_corr_mean"] = 0.0
        profile["num_abs_corr_max"] = 0.0

    return profile


# ===================== JSON serialization helper =====================

def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


# ===================== Główna procedura random search =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RandomizedSearch dla jednego modelu i datasetu."
    )
    parser.add_argument("--model_type", required=True, help="Typ modelu (np. logreg, rf, xgb, catboost_gpu)")
    parser.add_argument("--dataset_name", required=True, help="Nazwa zbioru (używana w meta)")
    parser.add_argument("--target_col", required=True, help="Nazwa kolumny celu")
    parser.add_argument(
        "--output_path",
        required=True,
        help="Ścieżka katalogu, gdzie zapisane będą wyniki (summary + cv_results).",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Opcjonalna pełna ścieżka do CSV; jeśli brak, użyje ./data/{dataset_name}.csv",
    )
    parser.add_argument(
        "--na_values",
        default=None,
        help="Opcjonalnie: lista wartości NA (np. 'na,NA,?').",
    )
    parser.add_argument("--grid_path", default="./config/model_type_grid.json")
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Użyj GPU dla modeli, które to wspierają (np. XGBoost).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.random_state)

    grid_path = Path(args.grid_path)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset path
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = Path("./data") / f"{args.dataset_name}.csv"

    # wczytanie danych
    read_kwargs = {}
    if args.na_values:
        read_kwargs["na_values"] = [v for v in args.na_values.split(",") if v]

    df = pd.read_csv(dataset_path, **read_kwargs)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Kolumna target {args.target_col} nie istnieje w zbiorze {dataset_path}"
        )

    dataset_profile = compute_dataset_profile(df, args.target_col)

    X, y = split_xy(df, args.target_col)

    # label encoding
    label_enc = TargetLabelEncoder().fit(y)
    y_enc = label_enc.transform(y)

    # pipeline
    preprocessor = build_two_stage_preprocessor()
    clf = build_model(args.model_type, use_gpu=args.use_gpu, random_state=args.random_state)
    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", clf),
        ]
    )

    # param grid
    param_distributions = load_param_grid(args.model_type, grid_path)

    # scoring: multi-metric, refit na balanced_accuracy
    scoring = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
    }

    cv = StratifiedKFold(
        n_splits=args.cv, shuffle=True, random_state=args.random_state
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scoring,
        refit="balanced_accuracy",
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=args.verbose,
        random_state=args.random_state,
        return_train_score=False,
    )

    search.fit(X, y_enc)

    # pełne wyniki CV -> csv
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    # top K i najgorsza konfiguracja
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
    worst_row = cv_sorted.sort_values(
        "rank_test_balanced_accuracy", ascending=False
    ).iloc[0]
    worst_config = {
        "rank": int(worst_row["rank_test_balanced_accuracy"]),
        "mean_balanced_accuracy": float(worst_row["mean_test_balanced_accuracy"]),
        "std_balanced_accuracy": float(worst_row["std_test_balanced_accuracy"]),
        "params": worst_row["params"],
    }

    # metryki na całym train dla best_estimator_
    best_est = search.best_estimator_
    y_pred_train_enc = best_est.predict(X)
    y_pred_train = label_enc.inverse_transform(y_pred_train_enc)

    metrics_train = {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_enc, y_pred_train_enc)
        ),
        "accuracy": float(accuracy_score(y_enc, y_pred_train_enc)),
        "f1": float(f1_score(y_enc, y_pred_train_enc)),
    }

    # classification_report i confusion_matrix (na oryginalnych etykietach)
    cls_report = classification_report(
        y,
        y_pred_train,
        output_dict=True,
    )
    cm = confusion_matrix(
        y, y_pred_train, labels=label_enc.classes_
    )

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path),
        "target_col": args.target_col,
        "model_type": args.model_type,
        "use_gpu": args.use_gpu,
        "search_settings": {
            "n_iter": args.n_iter,
            "cv": args.cv,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
            "grid_path": str(grid_path),
        },
        "label_classes": label_enc.classes_,
        "dataset_profile": dataset_profile,
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
        f"Zapisano wyniki do {out_dir} (cv_results.csv, summary.json). "
        f"Najlepszy wynik balanced_accuracy_cv={summary['cv_best_balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
