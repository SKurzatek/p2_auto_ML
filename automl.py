#!/usr/bin/env python3
# automl.py

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from sklearn.decomposition import PCA

# ===================== Stage: preprocessing =====================

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.int_cols_ = X.select_dtypes(include=[np.integer]).columns.tolist()
        self.float_cols_ = X.select_dtypes(include=[np.floating]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
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


def build_two_stage_preprocessor() -> Pipeline:
    stage1 = MissingValueHandler()
    stage2 = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include=[np.number])),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                make_column_selector(dtype_include=["object", "string", "category"]),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("stage1_missing", stage1), ("stage2_encode_scale", stage2)])


class EnsureDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X


# ===================== Stage: dataset profiling =====================

def compute_dataset_profile_xy(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    n_samples, n_features = X.shape
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

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

    vc = y.value_counts(dropna=False)
    class_counts = {str(k): int(v) for k, v in vc.to_dict().items()}
    class_props = {str(k): float(v) / float(len(y)) for k, v in vc.to_dict().items()}
    profile["class_counts"] = class_counts
    profile["class_proportions"] = class_props
    profile["n_classes"] = int(len(class_counts))

    props = list(class_props.values())
    profile["imbalance_ratio"] = float(max(props) / min(props)) if len(props) >= 2 and min(props) > 0 else float("inf")
    prep = build_two_stage_preprocessor()
    try:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        X_proc = prep.fit_transform(X, y_encoded)
        if sparse.issparse(X_proc):
            X_proc = X_proc.toarray()
            
        scorer = make_scorer(balanced_accuracy_score)

        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        profile["landmark_decision_stump"] = float(np.mean(cross_val_score(stump, X_proc, y_encoded, cv=3, scoring=scorer)))

        nb = GaussianNB()
        profile["landmark_naive_bayes"] = float(np.mean(cross_val_score(nb, X_proc, y_encoded, cv=3, scoring=scorer)))

        knn = KNeighborsClassifier(n_neighbors=1)

        idx = np.random.choice(len(X_proc), min(len(X_proc), 2000), replace=False)
        profile["landmark_1nn"] = float(np.mean(cross_val_score(knn, X_proc[idx], y_encoded[idx], cv=3, scoring=scorer)))

        profile["log_feature_sample_ratio"] = float(np.log10(n_features / n_samples))

        profile["class_entropy"] = float(entropy(list(class_props.values())))

        mi = mutual_info_classif(X_proc[idx], y_encoded[idx], discrete_features=False) 
        profile["mean_mutual_information"] = float(np.mean(mi))

        pca = PCA(n_components=0.95)
        pca.fit(X_proc[idx])
        profile["pca_95_components_ratio"] = float(pca.n_components_ / n_features)
    except Exception as e:
        profile["landmark_decision_stump"] = 0.5
        profile["landmark_naive_bayes"] = 0.5
        profile["landmark_1nn"] = 0.5

    return profile


def profile_to_vector(p: Dict[str, Any]) -> np.ndarray:
    def g(key: str, default: float = 0.0) -> float:
        v = p.get(key, default)
        try:
            if v is None:
                return float(default)
            return float(v)
        except Exception:
            return float(default)
    return np.array(
        [
            math.log10(max(g("n_samples"), 1.0)),
            math.log10(max(g("n_features"), 1.0)),
            math.log10(max(g("n_cat_features"), 1.0)),
            g("missing_fraction_mean"),
            g("landmark_decision_stump"),
            g("landmark_naive_bayes"),
            g("landmark_1nn"),
            g("mean_mutual_information"),
            g("class_entropy"),
            g("pca_95_components_ratio")
        ],
        dtype=float,
    )


# ===================== Stage: portfolio model factory =====================

def build_base_estimator(model_type: str, random_state: int) -> Any:
    if model_type == "sklearn.linear_model.LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs")
    if model_type == "sklearn.svm.LinearSVC":
        return LinearSVC(random_state=random_state, max_iter=5000)
    if model_type == "sklearn.linear_model.SGDClassifier":
        return SGDClassifier(random_state=random_state, max_iter=2000, tol=1e-3)
    if model_type == "sklearn.ensemble.RandomForestClassifier":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if model_type == "sklearn.ensemble.ExtraTreesClassifier":
        return ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model_type: {model_type}")


def needs_dense(model_type: str) -> bool:
    return model_type in ("sklearn.ensemble.RandomForestClassifier", "sklearn.ensemble.ExtraTreesClassifier")


def normalize_params_for_pipeline(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k.startswith("model__") or k.startswith("prep__"):
            out[k] = v
        else:
            out[f"model__{k}"] = v
    return out


def build_pipeline(model_type: str, params: Dict[str, Any], random_state: int) -> Pipeline:
    steps = [("prep", build_two_stage_preprocessor())]
    if needs_dense(model_type):
        steps.append(("to_dense", EnsureDense()))
    steps.append(("model", build_base_estimator(model_type, random_state=random_state)))
    pipe = Pipeline(steps)
    pipe.set_params(**normalize_params_for_pipeline(params))
    return pipe


# ===================== Stage: portfolio loading =====================

@dataclass
class PortfolioModel:
    id: str
    model_type: str
    params: Dict[str, Any]


def load_models_config(models_config: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]) -> List[PortfolioModel]:
    if isinstance(models_config, (str, Path)):
        path = Path(models_config)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = models_config

    if isinstance(obj, dict) and "models" in obj:
        out: List[PortfolioModel] = []
        for m in obj["models"]:
            out.append(
                PortfolioModel(
                    id=str(m.get("id", "")),
                    model_type=str(m["class"]),
                    params=dict(m.get("params") or {}),
                )
            )
        return out

    if isinstance(obj, list):
        out = []
        for i, m in enumerate(obj, start=1):
            mid = str(m.get("id") or m.get("name") or f"m{i:04d}")
            if "model_type" in m:
                out.append(PortfolioModel(id=mid, model_type=str(m["model_type"]), params=dict(m.get("params") or {})))
                continue
            cls = str(m.get("class", ""))
            out.append(PortfolioModel(id=mid, model_type = cls, params=dict(m.get("params") or {})))
        return out

    raise ValueError("Unsupported models_config format")


def load_dataset_profiles(report_or_profiles_path: Optional[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    if report_or_profiles_path is None:
        return {}
    path = Path(report_or_profiles_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "dataset_profiles" in obj and isinstance(obj["dataset_profiles"], dict):
        return obj["dataset_profiles"]
    if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        return obj  # dataset_name -> profile
    return {}


def load_model_evidence(report_path: Optional[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    if report_path is None:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {}
    items = obj.get("models_rich")
    if not isinstance(items, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for m in items:
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            out[mid] = m
    return out


# ===================== Stage: MiniAutoML =====================

class MiniAutoML:
    def __init__(
        self,
        models_config: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
        report_or_profiles_path: Optional[Union[str, Path]] = None,
        report_path_for_evidence: Optional[Union[str, Path]] = None,
        random_state: int = 42,
        top_k_models: int = 5,
        shortlist_size: int = 12,
        test_size: float = 0.2,
        cv_folds_stack: int = 3,
    ):
        self.random_state = int(random_state)
        self.top_k_models = int(top_k_models)
        self.shortlist_size = int(shortlist_size)
        self.test_size = float(test_size)
        self.cv_folds_stack = int(cv_folds_stack)

        self.portfolio: List[PortfolioModel] = load_models_config(models_config)
        if len(self.portfolio) > 50:
            raise ValueError(f"Portfolio has {len(self.portfolio)} models; must be <= 50.")

        self.known_dataset_profiles = load_dataset_profiles(report_or_profiles_path)
        self.model_evidence = load_model_evidence(report_path_for_evidence)

        self.label_encoder_: Optional[LabelEncoder] = None
        self.new_dataset_profile_: Optional[Dict[str, Any]] = None

        self.selected_models_: List[Tuple[PortfolioModel, Pipeline]] = []
        self.best_model_idx_: int = 0

        self.ensemble_mode_: Optional[str] = None  # "best" | "avg" | "stacker_logreg_c1" | "stacker_logreg_c01" | "stacker_rf_d3" | "stacker_mlp_10"
        self.stacker_: Optional[MLPClassifier] = None

        self.validation_score_best_: Optional[float] = None
        self.validation_score_avg_: Optional[float] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _predict_proba_1d(self, model: Pipeline, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            if p.shape[1] != 2:
                raise ValueError("MiniAutoML currently supports binary classification only for stacking.")
            return p[:, 1].astype(float)

        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            s = np.asarray(s)
            if s.ndim != 1:
                raise ValueError("MiniAutoML currently supports binary classification only for stacking.")
            return self._sigmoid(s).astype(float)

        yhat = model.predict(X)
        return np.asarray(yhat).astype(float)

    def _profile_similarity_weights(self, new_profile: Dict[str, Any]) -> Dict[str, float]:
        if not self.known_dataset_profiles:
            return {}

        new_v = profile_to_vector(new_profile)
        names = list(self.known_dataset_profiles.keys())
        mat = np.vstack([profile_to_vector(self.known_dataset_profiles[n]) for n in names])

        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        sd[sd == 0] = 1.0

        new_z = (new_v - mu) / sd
        mat_z = (mat - mu) / sd

        d = np.sqrt(((mat_z - new_z) ** 2).sum(axis=1))
        w = 1.0 / (d + 1e-6)
        w = w / w.sum()
        return {names[i]: float(w[i]) for i in range(len(names))}

    def _meta_rank_models(self, new_profile: Dict[str, Any]) -> List[PortfolioModel]:
        weights = self._profile_similarity_weights(new_profile)
        if not weights or not self.model_evidence:
            return list(self.portfolio)

        scored: List[Tuple[float, PortfolioModel]] = []
        for pm in self.portfolio:
            info = self.model_evidence.get(pm.id)
            if not info:
                scored.append((0.0, pm))
                continue

            evidence = info.get("evidence") or []
            num = 0.0
            den = 0.0
            for e in evidence:
                ds = e.get("dataset_name")
                if ds in weights:
                    s = e.get("mean_cv_balanced_accuracy") or e.get("cv_best_balanced_accuracy")
                    try:
                        s = float(s)
                    except Exception:
                        continue
                    num += weights[ds] * s
                    den += weights[ds]
            meta = (num / den) if den > 0 else float(info.get("aggregate", {}).get("mean_cv_balanced_accuracy", 0.0))
            scored.append((meta, pm))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [pm for _, pm in scored]

    def _fit_stackers_and_pick(
        self,
        P_oof: np.ndarray,
        y_tr: np.ndarray,
        P_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Optional[str], Optional[BaseEstimator], Dict[str, float]]:
        """
        Trenuje zestaw różnorodnych meta-modeli i wybiera ten, 
        który najlepiej radzi sobie na danych walidacyjnych.
        """
        # Słownik kandydatów na meta-model (stacker)
        candidates = {
            "stacker_logreg_c1": LogisticRegression(C=1.0, random_state=self.random_state),
            "stacker_logreg_c01": LogisticRegression(C=0.1, random_state=self.random_state),
            "stacker_rf_d3": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=self.random_state),
            "stacker_mlp_10": MLPClassifier(
                hidden_layer_sizes=(10,), max_iter=400, random_state=self.random_state, 
                early_stopping=True, validation_fraction=0.2
            )
        }

        best_name: Optional[str] = None
        best_model: Optional[BaseEstimator] = None
        scores: Dict[str, float] = {}

        for name, model in candidates.items():
            try:
                # Trenowanie na predykcjach OOF (Out-of-Fold)
                model.fit(P_oof, y_tr)
                
                # Predykcja na zbiorze walidacyjnym
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(P_val)[:, 1]
                else:
                    p = model.predict(P_val)
                
                pred = (p >= 0.5).astype(int)
                s = float(balanced_accuracy_score(y_val, pred))
                scores[name] = s

                # Wybór najlepszego na podstawie zbalansowanej dokładności
                if best_name is None or s > scores[best_name]:
                    best_name = name
                    best_model = model
            except Exception:
                scores[name] = 0.0

        return best_name, best_model, scores

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]):
        X = pd.DataFrame(X_train).copy() if not isinstance(X_train, pd.DataFrame) else X_train.copy()
        y = pd.Series(y_train).copy() if not isinstance(y_train, pd.Series) else y_train.copy()

        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str) if y.dtype == "object" else y)
        self.label_encoder_ = le

        if len(np.unique(y_enc)) != 2:
            raise ValueError("MiniAutoML currently supports binary classification only.")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y_enc,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_enc,
        )
        y_tr = np.asarray(y_tr)
        y_val = np.asarray(y_val)

        self.new_dataset_profile_ = compute_dataset_profile_xy(X_tr, pd.Series(le.inverse_transform(y_tr)))
        ranked = self._meta_rank_models(self.new_dataset_profile_)

        shortlist = ranked[: max(self.shortlist_size, self.top_k_models)]
        scored_val: List[Tuple[float, PortfolioModel]] = []

        # Fit/eval shortlist models on train->val to find best 5
        fitted_shortlist: Dict[str, Pipeline] = {}
        for pm in shortlist:
            pipe = build_pipeline(pm.model_type, pm.params, random_state=self.random_state)
            pipe.fit(X_tr, y_tr)
            fitted_shortlist[pm.id] = pipe
            p = self._predict_proba_1d(pipe, X_val)
            pred = (p >= 0.5).astype(int)
            score = float(balanced_accuracy_score(y_val, pred))
            scored_val.append((score, pm))

        scored_val.sort(key=lambda t: t[0], reverse=True)
        top_defs = [pm for _, pm in scored_val[: self.top_k_models]]

        # Refit top models on full training split (consistent base set)
        self.selected_models_ = []
        for pm in top_defs:
            m = build_pipeline(pm.model_type, pm.params, random_state=self.random_state)
            m.fit(X_tr, y_tr)
            self.selected_models_.append((pm, m))

        # Validation matrices for base/avg
        P_val = np.column_stack([self._predict_proba_1d(m, X_val) for _, m in self.selected_models_])

        # Best single among top-k (on validation)
        indiv_scores = []
        for j in range(P_val.shape[1]):
            predj = (P_val[:, j] >= 0.5).astype(int)
            indiv_scores.append(float(balanced_accuracy_score(y_val, predj)))
        self.best_model_idx_ = int(np.argmax(indiv_scores))
        self.validation_score_best_ = float(indiv_scores[self.best_model_idx_])

        # Average ensemble on validation
        p_avg = P_val.mean(axis=1)
        pred_avg = (p_avg >= 0.5).astype(int)
        self.validation_score_avg_ = float(balanced_accuracy_score(y_val, pred_avg))

        # OOF predictions for stacking
        skf = StratifiedKFold(n_splits=self.cv_folds_stack, shuffle=True, random_state=self.random_state)
        P_oof = np.zeros((len(X_tr), len(self.selected_models_)), dtype=float)

        for i, (pm, _) in enumerate(self.selected_models_):
            for train_idx, hold_idx in skf.split(X_tr, y_tr):
                X_fold_tr = X_tr.iloc[train_idx]
                y_fold_tr = y_tr[train_idx]
                X_fold_hold = X_tr.iloc[hold_idx]

                base = build_pipeline(pm.model_type, pm.params, random_state=self.random_state)
                base.fit(X_fold_tr, y_fold_tr)
                P_oof[hold_idx, i] = self._predict_proba_1d(base, X_fold_hold)

        # Train diverse stackers, score each on validation, keep best stacker
        best_stacker_name, best_stacker, stacker_scores = self._fit_stackers_and_pick(P_oof, y_tr, P_val, y_val)
        
        # Zapisujemy wyniki walidacyjne dla nowych kandydatów
        self.validation_score_stacker_logreg_c1_ = stacker_scores.get("stacker_logreg_c1")
        self.validation_score_stacker_logreg_c01_ = stacker_scores.get("stacker_logreg_c01")
        self.validation_score_stacker_rf_d3_ = stacker_scores.get("stacker_rf_d3")
        self.validation_score_stacker_mlp_10_ = stacker_scores.get("stacker_mlp_10")

        # Choose final mode among: best, avg, best_stacker
        mode_scores: Dict[str, float] = {
            "best": float(self.validation_score_best_),
            "avg": float(self.validation_score_avg_),
        }
        if best_stacker_name is not None and best_stacker is not None:
            mode_scores[best_stacker_name] = float(stacker_scores[best_stacker_name])

        self.ensemble_mode_ = max(mode_scores, key=mode_scores.get)

        # Sprawdzenie czy wygrał jeden z nowych stackerów
        if self.ensemble_mode_ in ("stacker_logreg_c1", "stacker_logreg_c01", "stacker_rf_d3", "stacker_mlp_10"):
            self.stacker_ = best_stacker
        else:
            self.stacker_ = None

        return self

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.selected_models_:
            raise RuntimeError("MiniAutoML is not fitted yet. Call fit() first.")
        if self.ensemble_mode_ is None:
            raise RuntimeError("MiniAutoML has no ensemble_mode_. Did fit() complete?")

        X = pd.DataFrame(X_test).copy() if not isinstance(X_test, pd.DataFrame) else X_test.copy()
        P = np.column_stack([self._predict_proba_1d(m, X) for _, m in self.selected_models_])

        if self.ensemble_mode_ in ("stacker_logreg_c1", "stacker_logreg_c01", "stacker_rf_d3", "stacker_mlp_10") and self.stacker_ is not None and P.shape[1] >= 2:
            p1 = self.stacker_.predict_proba(P)[:, 1]
        elif self.ensemble_mode_ == "avg":
            p1 = P.mean(axis=1)
        else:  # "best"
            p1 = P[:, self.best_model_idx_]

        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X_test)[:, 1]
        y_pred_enc = (proba >= 0.5).astype(int)
        if self.label_encoder_ is None:
            return y_pred_enc
        return self.label_encoder_.inverse_transform(y_pred_enc)


# ===================== Stage: CLI runner =====================

def _cli():
    p = argparse.ArgumentParser(description="Run MiniAutoML on one CSV (binary classification).")
    p.add_argument("--csv_path", required=True, help="Path to CSV file")
    p.add_argument("--target_col", required=True, help="Target column name in CSV")
    p.add_argument("--models", default="portfolio/models.json", help="Path to portfolio models.json")
    p.add_argument("--report", default="portfolio/portfolio_report.json", help="Path to portfolio_report.json (profiles+evidence)")
    p.add_argument("--top_k_models", type=int, default=5)
    p.add_argument("--shortlist_size", type=int, default=12)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--cv_folds_stack", type=int, default=3)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found in {args.csv_path}")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    automl = MiniAutoML(
        models_config=args.models,
        report_or_profiles_path=args.report,
        report_path_for_evidence=args.report,
        random_state=args.random_state,
        top_k_models=args.top_k_models,
        shortlist_size=args.shortlist_size,
        test_size=args.test_size,
        cv_folds_stack=args.cv_folds_stack,
    )

    t0 = time.perf_counter()
    automl.fit(X, y)
    t1 = time.perf_counter()

    print("\n=== MiniAutoML results ===")
    print("dataset:", args.csv_path)
    print("fit_time_sec:", round(t1 - t0, 3))
    print("ensemble_mode:", automl.ensemble_mode_)
    print("val_best:", automl.validation_score_best_)
    print("val_avg:", automl.validation_score_avg_)
    print("val_stacker_logreg_c1:", automl.validation_score_stacker_logreg_c1_)
    print("val_stacker_logreg_c01:", automl.validation_score_stacker_logreg_c01_)
    print("val_stacker_rf_d3:", automl.validation_score_stacker_rf_d3_)
    print("val_stacker_mlp_10:", automl.validation_score_stacker_mlp_10_)

    print("\n=== chosen models (top 5) ===")
    for i, (pm, _) in enumerate(automl.selected_models_, start=1):
        print(f"{i}. id={pm.id} type={pm.model_type} params={pm.params}")

    print("\n=== new dataset profile (train split) ===")
    print(automl.new_dataset_profile_)


if __name__ == "__main__":
    _cli()
