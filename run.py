import sys
import subprocess
from pathlib import Path
from datetime import datetime

# CPU-only models compatible with config/model_type_grid.json
MODEL_TYPES = [
    "sklearn.linear_model.LogisticRegression",
    "sklearn.svm.LinearSVC",
    "sklearn.linear_model.SGDClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble.ExtraTreesClassifier",
]

# Datasets present in ./data
DATASETS = [
    {"name": "adult", "target": "income", "na_values": None},
    {"name": "aps_failure", "target": "class", "na_values": "na"},
    {"name": "bank_marketing", "target": "y", "na_values": None},
    {"name": "breast_cancer", "target": "target", "na_values": None},
    {"name": "credit_default", "target": "default", "na_values": None},
    {"name": "santander_small", "target": "target", "na_values": None},
]

BASE_RESULTS_DIR = Path("./results")

# Random-search settings enforced by random_search.py, but we pass these knobs explicitly
GRID_PATH = Path("./config/model_type_grid.json")
CV_FOLDS = 3
CV_MAX_ROWS = 9000
N_ITER_CAP = 20

N_JOBS = 1
VERBOSE = 3
RANDOM_STATE = 42


def run_one(model_type: str, ds: dict, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "random_search.py",
        "--model_type", model_type,
        "--dataset_name", ds["name"],
        "--target_col", ds["target"],
        "--output_path", str(out_dir),
        "--grid_path", str(GRID_PATH),
        "--cv_folds", str(CV_FOLDS),
        "--cv_max_rows", str(CV_MAX_ROWS),
        "--n_iter_cap", str(N_ITER_CAP),
        "--n_jobs", str(N_JOBS),
        "--random_state", str(RANDOM_STATE),
        "--verbose", str(VERBOSE),
    ]

    if ds.get("na_values"):
        cmd.extend(["--na_values", ds["na_values"]])

    print("\n=== Running", model_type, "on", ds["name"], "===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Missing grid file: {GRID_PATH}. "
            f"Expected CPU grid config at ./config/model_type_grid.json"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        for model_type in MODEL_TYPES:
            out_dir = run_dir / f"{model_type}__{ds['name']}"
            start = datetime.now()
            run_one(model_type, ds, out_dir)
            elapsed = datetime.now() - start
            print(f"[OK] {model_type} on {ds['name']} finished in {elapsed}")

    print(f"\nAll done. Results saved under: {run_dir}")


if __name__ == "__main__":
    main()