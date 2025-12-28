from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ast

# ===================== Stage: containers =====================

@dataclass
class Candidate:
    dataset_name: str
    model_type: str
    run_item_dir: str
    params: Dict[str, Any]
    mean_bacc: float
    std_bacc: Optional[float]
    rank_bacc: Optional[int]
    n_iter_used: Optional[int]
    cv_folds: Optional[int]
    rows_used: Optional[int]
    grid_path: Optional[str]
    dataset_profile: Optional[Dict[str, Any]]


# ===================== Stage: helpers =====================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(x)
    except Exception:
        return None


def _signature(model_type: str, params: Dict[str, Any]) -> str:
    items = sorted(params.items(), key=lambda kv: kv[0])
    payload = json.dumps({"model_type": model_type, "params": items}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_run_items(run_dir: Path) -> List[Path]:
    return sorted({p.parent for p in run_dir.rglob("summary.json")})


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_search_meta(summary: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    ss = summary.get("search_settings", {}) or {}
    sp = summary.get("search_sample", {}) or {}
    return (
        _safe_int(ss.get("n_iter_used")),
        _safe_int(ss.get("cv_folds")),
        _safe_int(sp.get("n_rows_used")),
        None if ss.get("grid_path") is None else str(ss.get("grid_path")),
    )


def _params_from_cv_row(row: pd.Series) -> Dict[str, Any]:
    p = row.get("params", {})
    if isinstance(p, dict):
        return dict(p)
    if isinstance(p, str) and p.strip():
        try:
            parsed = ast.literal_eval(p)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# ===================== Stage: diversity bucketing =====================

def _bucket_linear_C(params: Dict[str, Any]) -> str:
    C = params.get("model__C", None)
    try:
        if C is None:
            return "C:None"
        C = float(C)
        if C < 0.1:
            return "C:small"
        if C <= 1.0:
            return "C:medium"
        return "C:large"
    except Exception:
        return "C:other"


def _bucket_class_weight(params: Dict[str, Any]) -> str:
    cw = params.get("model__class_weight", None)
    return "cw:balanced" if cw == "balanced" else "cw:none"


def _bucket_sgd_alpha(params: Dict[str, Any]) -> str:
    a = params.get("model__alpha", None)
    try:
        if a is None:
            return "alpha:None"
        a = float(a)
        if a <= 1e-5:
            return "alpha:tiny"
        if a <= 1e-4:
            return "alpha:small"
        if a <= 1e-3:
            return "alpha:medium"
        return "alpha:large"
    except Exception:
        return "alpha:other"


def _bucket_tree_depth(params: Dict[str, Any]) -> str:
    d = params.get("model__max_depth", None)
    if d is None:
        return "depth:none"
    try:
        d = int(d)
        if d <= 10:
            return "depth:shallow"
        if d <= 20:
            return "depth:mid"
        return "depth:deep"
    except Exception:
        return "depth:other"


def _bucket_tree_leaf(params: Dict[str, Any]) -> str:
    leaf = params.get("model__min_samples_leaf", None)
    try:
        leaf = int(leaf) if leaf is not None else 1
        return "leaf:1" if leaf <= 1 else "leaf:>1"
    except Exception:
        return "leaf:other"


def _bucket_tree_features(params: Dict[str, Any]) -> str:
    mf = params.get("model__max_features", None)
    if mf is None:
        return "mf:none"
    return f"mf:{mf}"


def bucket_key(model_type: str, params: Dict[str, Any]) -> str:
    if model_type in ("logreg", "linear_svc"):
        return "|".join([_bucket_class_weight(params), _bucket_linear_C(params)])
    if model_type == "sgd":
        loss = params.get("model__loss", "loss:na")
        pen = params.get("model__penalty", "pen:na")
        l1r = params.get("model__l1_ratio", None)
        l1r_b = "l1r:na"
        if pen == "elasticnet":
            try:
                l1r_f = float(l1r) if l1r is not None else 0.0
                l1r_b = "l1r:low" if l1r_f < 0.3 else ("l1r:mid" if l1r_f < 0.7 else "l1r:high")
            except Exception:
                l1r_b = "l1r:other"
        return "|".join([f"loss:{loss}", f"pen:{pen}", l1r_b, _bucket_sgd_alpha(params)])
    if model_type in ("rf", "extratrees"):
        return "|".join([_bucket_tree_depth(params), _bucket_tree_leaf(params), _bucket_tree_features(params)])
    return "bucket:default"


def diversify_candidates(
    cands: List[Candidate],
    per_pair_cap: int,
) -> List[Candidate]:
    """
    Pick candidates in a way that favors distinct "buckets" first, then fills remaining
    slots by quality.
    """
    if not cands:
        return []

    # Sort by quality first
    cands_sorted = sorted(cands, key=lambda c: c.mean_bacc, reverse=True)

    picked: List[Candidate] = []
    seen_buckets: set[str] = set()

    # First pass: unique buckets
    for c in cands_sorted:
        if len(picked) >= per_pair_cap:
            break
        b = bucket_key(c.model_type, c.params)
        if b in seen_buckets:
            continue
        picked.append(c)
        seen_buckets.add(b)

    # Second pass: fill remaining by quality
    if len(picked) < per_pair_cap:
        picked_sigs = {(_signature(c.model_type, c.params)) for c in picked}
        for c in cands_sorted:
            if len(picked) >= per_pair_cap:
                break
            sig = _signature(c.model_type, c.params)
            if sig in picked_sigs:
                continue
            picked.append(c)
            picked_sigs.add(sig)

    return picked


# ===================== Stage: candidate extraction =====================

def extract_candidates_from_run_item(
    run_item_dir: Path,
    delta: float,
    per_pair_cap: int,
    min_abs_keep: int,
) -> List[Candidate]:
    summary_path = run_item_dir / "summary.json"
    cv_path = run_item_dir / "cv_results.csv"
    if not (summary_path.exists() and cv_path.exists()):
        return []

    summary = load_json(summary_path)
    dataset_name = str(summary.get("dataset_name"))
    model_type = str(summary.get("model_type"))
    dataset_profile = None if summary.get("dataset_profile") is None else dict(summary.get("dataset_profile"))
    n_iter_used, cv_folds, rows_used, grid_path = _get_search_meta(summary)

    cv = pd.read_csv(cv_path)

    mean_col = "mean_test_balanced_accuracy"
    std_col = "std_test_balanced_accuracy"
    rank_col = "rank_test_balanced_accuracy"

    if mean_col not in cv.columns or "params" not in cv.columns:
        return []

    # Best mean for this (dataset, model_type)
    best_mean = float(cv[mean_col].max())
    threshold = best_mean - float(delta)

    pool = cv[cv[mean_col] >= threshold].copy()
    pool = pool.sort_values([mean_col], ascending=[False])

    # Ensure we keep at least some candidates even if delta is very tight
    if len(pool) < min_abs_keep:
        pool = cv.sort_values([mean_col], ascending=[False]).head(min_abs_keep).copy()

    cands: List[Candidate] = []
    for _, row in pool.iterrows():
        params = _params_from_cv_row(row)
        if not params:
            continue
        cands.append(
            Candidate(
                dataset_name=dataset_name,
                model_type=model_type,
                run_item_dir=str(run_item_dir),
                params=params,
                mean_bacc=float(row[mean_col]),
                std_bacc=_safe_float(row.get(std_col)),
                rank_bacc=_safe_int(row.get(rank_col)),
                n_iter_used=n_iter_used,
                cv_folds=cv_folds,
                rows_used=rows_used,
                grid_path=grid_path,
                dataset_profile=dataset_profile,
            )
        )

    return diversify_candidates(cands, per_pair_cap=per_pair_cap)


# ===================== Stage: portfolio build =====================

def build_portfolio(
    candidates: List[Candidate],
    max_models: int,
    min_per_model_type: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - models_json (project requirement)
      - report_json (rich portfolio report)
    """
    # Group by signature, aggregate evidence
    grouped: Dict[str, Dict[str, Any]] = {}
    dataset_profiles: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []

    for c in candidates:
        if c.dataset_name and c.dataset_profile:
            dataset_profiles[c.dataset_name] = c.dataset_profile

        sig = _signature(c.model_type, c.params)
        if sig not in grouped:
            grouped[sig] = {
                "signature": sig,
                "model_type": c.model_type,
                "params": c.params,
                "evidence": [],
            }
        grouped[sig]["evidence"].append(
            {
                "dataset_name": c.dataset_name,
                "mean_cv_balanced_accuracy": c.mean_bacc,
                "std_cv_balanced_accuracy": c.std_bacc,
                "rank_cv_balanced_accuracy": c.rank_bacc,
                "cv_folds": c.cv_folds,
                "n_iter_used": c.n_iter_used,
                "rows_used": c.rows_used,
                "run_item_dir": c.run_item_dir,
                "grid_path": c.grid_path,
            }
        )

        trace.append(
            {
                "dataset_name": c.dataset_name,
                "model_type": c.model_type,
                "mean_cv_balanced_accuracy": c.mean_bacc,
                "signature": sig,
                "run_item_dir": c.run_item_dir,
            }
        )

    # Build rich model entries with aggregates
    rich_models: List[Dict[str, Any]] = []
    for sig, entry in grouped.items():
        scores = [e["mean_cv_balanced_accuracy"] for e in entry["evidence"] if isinstance(e["mean_cv_balanced_accuracy"], (int, float))]
        if not scores:
            continue
        rich_models.append(
            {
                "id": None,
                "signature": sig,
                "model_type": entry["model_type"],
                "params": entry["params"],
                "aggregate": {
                    "datasets_supported": len(entry["evidence"]),
                    "mean_cv_balanced_accuracy": float(sum(scores) / len(scores)),
                    "max_cv_balanced_accuracy": float(max(scores)),
                    "min_cv_balanced_accuracy": float(min(scores)),
                },
                "evidence": entry["evidence"],
            }
        )

    # Rank by generality + quality
    rich_models.sort(
        key=lambda m: (
            m["aggregate"]["datasets_supported"],
            m["aggregate"]["mean_cv_balanced_accuracy"],
            m["aggregate"]["max_cv_balanced_accuracy"],
        ),
        reverse=True,
    )

    # Diversity-first selection:
    # 1) Ensure at least min_per_model_type models per model_type (if available)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for m in rich_models:
        by_type.setdefault(m["model_type"], []).append(m)

    selected: List[Dict[str, Any]] = []
    selected_sigs: set[str] = set()

    # Pass A: minimum per model type
    for mt, lst in by_type.items():
        take = min(min_per_model_type, len(lst))
        for m in lst[:take]:
            if len(selected) >= max_models:
                break
            if m["signature"] in selected_sigs:
                continue
            selected.append(m)
            selected_sigs.add(m["signature"])

    # Pass B: round-robin over remaining models (high diversity)
    if len(selected) < max_models:
        # Build iterators per type, skipping already selected
        idx: Dict[str, int] = {mt: 0 for mt in by_type}
        while len(selected) < max_models:
            progressed = False
            for mt, lst in by_type.items():
                # advance to next non-selected
                while idx[mt] < len(lst) and lst[idx[mt]]["signature"] in selected_sigs:
                    idx[mt] += 1
                if idx[mt] >= len(lst):
                    continue
                m = lst[idx[mt]]
                idx[mt] += 1
                if m["signature"] in selected_sigs:
                    continue
                selected.append(m)
                selected_sigs.add(m["signature"])
                progressed = True
                if len(selected) >= max_models:
                    break
            if not progressed:
                break

    # Assign IDs and build models.json list (project requirement)
    models_list: List[Dict[str, Any]] = []
    for i, m in enumerate(selected, start=1):
        mid = f"m{i:04d}"
        m["id"] = mid
        models_list.append(
            {
                "id": mid,
                "model_type": m["model_type"],
                "params": m["params"],
            }
        )

    models_json = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_models": len(models_list),
        "models": models_list,
    }

    report_json = {
        "build_info": {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "selection_strategy": "within_delta_per_pair + per_pair_diversify + global_round_robin_diversity",
            "n_candidates_input": len(candidates),
            "n_unique_models": len(rich_models),
            "n_selected_models": len(selected),
        },
        "preprocessing": {
            "name": "two_stage_missing_ohe_scale_v1",
            "note": "Assumes the same preprocessing pipeline as random_search.py.",
        },
        "dataset_profiles": dataset_profiles,
        "models_rich": selected,
        "trace": trace,
    }

    return models_json, report_json


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ===================== Stage: main =====================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a diverse models.json portfolio from a screening run dir.")
    p.add_argument("--run_dir", required=True, help="Path to ./results/run_YYYYMMDD_HHMMSS directory")
    p.add_argument("--out_dir", default=None, help="Output dir (default: <run_dir>/portfolio)")
    p.add_argument("--max_models", type=int, default=50, help="Portfolio size cap (<=50)")
    p.add_argument("--delta", type=float, default=0.02, help="Include configs within (best - delta) per (dataset, model_type)")
    p.add_argument("--per_pair_cap", type=int, default=6, help="Max selected configs per (dataset, model_type) after diversity bucketing")
    p.add_argument("--min_abs_keep", type=int, default=3, help="Keep at least this many per pair even if delta is tight")
    p.add_argument("--min_per_model_type", type=int, default=6, help="Try to include at least this many per model_type (if available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found or not a directory: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "portfolio")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_items = discover_run_items(run_dir)
    if not run_items:
        raise RuntimeError(f"No summary.json found under: {run_dir}")

    all_candidates: List[Candidate] = []

    for item in run_items:
        # Extract a diverse subset per pair using delta + bucketing
        cands = extract_candidates_from_run_item(
            item,
            delta=float(args.delta),
            per_pair_cap=int(args.per_pair_cap),
            min_abs_keep=int(args.min_abs_keep),
        )
        all_candidates.extend(cands)

    if not all_candidates:
        raise RuntimeError(f"No candidates found under {run_dir}. Check that cv_results.csv exists and has expected columns.")

    models_json, report_json = build_portfolio(
        candidates=all_candidates,
        max_models=int(args.max_models),
        min_per_model_type=int(args.min_per_model_type),
    )

    # Add source metadata to models.json (still project-compliant)
    models_json["source_run_dir"] = str(run_dir)

    write_json(out_dir / "models.json", models_json)
    write_json(out_dir / "portfolio_report.json", report_json)

    print(f"[OK] Wrote: {out_dir / 'models.json'} (n_models={models_json['n_models']})")
    print(f"[OK] Wrote: {out_dir / 'portfolio_report.json'}")


if __name__ == "__main__":
    main()
