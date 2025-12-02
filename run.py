#!/usr/bin/env python
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Modele zgodne z config/model_type_grid.json
MODEL_TYPES = [
    "xgb",
    "logreg",
    "rf",
    "extratrees"
]
# "svc_rbf", można sie zastanowić czy to chcemy
# "catboost_gpu", tego raczej nie
# "knn", można sie zastanowić czy to chcemy
# "hgb", można sie zastanowić czy to chcemy

# Konfiguracja zbiorów – dopasuj targety do swoich plików
DATASETS = [
    {
        "name": "adult",
        "target": "income",
        "na_values": None,
    },
    {
        "name": "credit_default",
        "target": "default",
        "na_values": None,
    },
    {
        "name": "breast_cancer",
        "target": "target",
        "na_values": None,
    },
    {
        "name": "bank_marketing",
        "target": "y",
        "na_values": None,
    }
]
'''
{
        "name": "aps_failure",
        "target": "class",
        "na_values": "na",
    },
    {
        "name": "santander_small",
        "target": "target",
        "na_values": None,
    },
'''


# Oznaczamy duże zbiory
#LARGE_DATASETS = {"aps_failure", "santander_small"}
LARGE_DATASETS = {"bank_marketing", "adult"}
# , "knn"
SKIP_MODELS_ON_LARGE = {"svc_rbf"}

BASE_RESULTS_DIR = Path("./results")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Domyślne ustawienia random search – dostosuj wg potrzeb
    # 30
    n_iter = 50
    # 5
    cv = 4

    for ds in DATASETS:
        ds_name = ds["name"]
        target = ds["target"]
        na_values = ds.get("na_values")

        for model_type in MODEL_TYPES:
            if ds_name in LARGE_DATASETS and model_type in SKIP_MODELS_ON_LARGE:
                print(
                    f"[INFO] Pomijam model {model_type} na dużym zbiorze {ds_name}"
                )
                continue

            out_dir = run_dir / f"{model_type}__{ds_name}"

            # ustawienia CPU/GPU
            use_gpu_flag = False
            n_jobs = -1

            if model_type in {"xgb", "catboost_gpu"}:
                use_gpu_flag = True
                n_jobs = 1  # GPU - pojedynczy proces

            start = datetime.now()
            cmd = [
                sys.executable,
                "random_search.py",
                "--model_type",
                model_type,
                "--dataset_name",
                ds_name,
                "--target_col",
                target,
                "--output_path",
                str(out_dir),
                "--n_iter",
                str(n_iter),
                "--cv",
                str(cv),
                "--n_jobs",
                str(n_jobs),
                "--verbose",
                "1",
            ]

            if na_values:
                cmd.extend(["--na_values", na_values])

            if use_gpu_flag:
                cmd.append("--use_gpu")

            print(f"\n=== Running {model_type} on {ds_name} ===")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            end = datetime.now() - start
            print(f"The run took:   {end}")


if __name__ == "__main__":
    main()
