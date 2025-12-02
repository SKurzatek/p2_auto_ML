#!/usr/bin/env python
"""
prepare_datasets.py

Na podstawie obecnej struktury:

data/
  adults/adult.data, adult.test
  aps/aps_failure_training_set.csv, aps_failure_test_set.csv
  archieve/train.csv, test.csv
  bank/bank-full.csv
  breast/wdbc.data
  credit/default of credit card clients.xls

tworzy kanoniczne pliki:

data/adult.csv
data/credit_default.csv
data/aps_failure.csv
data/breast_cancer.csv
data/bank_marketing.csv
data/santander_small.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("data")
RANDOM_STATE = 42


# =============== ADULTS ===============

def prepare_adult():
    """
    data/adults/adult.data + adult.test -> data/adult.csv

    Target: 'income'
    """
    src_dir = DATA_ROOT / "adults"
    out_path = DATA_ROOT / "adult.csv"

    if out_path.exists():
        print("[adult] data/adult.csv już istnieje, pomijam.")
        return

    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    train_path = src_dir / "adult.data"
    test_path = src_dir / "adult.test"

    if not train_path.exists():
        raise FileNotFoundError(f"[adult] nie znalazłem {train_path}")

    print(f"[adult] wczytuję {train_path}")
    train = pd.read_csv(
        train_path,
        header=None,
        names=cols,
        na_values="?",
        skipinitialspace=True,
    )

    frames = [train]

    if test_path.exists():
        print(f"[adult] doklejam {test_path}")
        test = pd.read_csv(
            test_path,
            header=None,
            names=cols,
            na_values="?",
            skiprows=1,  # pierwsza linia to nagłówek z UCI
            skipinitialspace=True,
        )
        # w adult.test etykiety mają kropkę typu ' <=50K.'
        test["income"] = (
            test["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        )
        frames.append(test)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[adult] zapisano {out_path} shape={df.shape}")


# =============== CREDIT DEFAULT ===============

def prepare_credit_default():
    """
    data/credit/default of credit card clients.xls -> data/credit_default.csv

    Target: 'default'
    """
    src_dir = DATA_ROOT / "credit"
    out_path = DATA_ROOT / "credit_default.csv"

    if out_path.exists():
        print("[credit_default] data/credit_default.csv już istnieje, pomijam.")
        return

    xls_path = src_dir / "default of credit card clients.xls"
    if not xls_path.exists():
        raise FileNotFoundError(f"[credit_default] nie znalazłem {xls_path}")

    print(f"[credit_default] wczytuję {xls_path}")
    # w UCI pierwszy wiersz to opis, drugi nagłówki
    df = pd.read_excel(xls_path, header=1)

    # kolumna celu to zwykle 'default payment next month'
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    elif "default" not in df.columns:
        raise ValueError(
            "[credit_default] nie znalazłem kolumny 'default payment next month' ani 'default'"
        )

    df.to_csv(out_path, index=False)
    print(f"[credit_default] zapisano {out_path} shape={df.shape}")


# =============== APS FAILURE ===============

def prepare_aps_failure():
    """
    data/aps/aps_failure_training_set.csv (+ opcjonalnie test) -> data/aps_failure.csv

    Target: 'class'
    """
    src_dir = DATA_ROOT / "aps"
    out_path = DATA_ROOT / "aps_failure.csv"

    if out_path.exists():
        print("[aps_failure] data/aps_failure.csv już istnieje, pomijam.")
        return

    train_path = src_dir / "aps_failure_training_set.csv"
    test_path = src_dir / "aps_failure_test_set.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"[aps_failure] nie znalazłem {train_path}")

    print(f"[aps_failure] wczytuję train: {train_path}")
    # w oryginalnym pliku pierwsze ~20 linii to opis, potem nagłówki
    train = pd.read_csv(train_path, skiprows=20, na_values="na")

    frames = [train]

    if test_path.exists():
        print(f"[aps_failure] doklejam test: {test_path}")
        test = pd.read_csv(test_path, skiprows=20, na_values="na")
        frames.append(test)

    df = pd.concat(frames, ignore_index=True)

    if "class" not in df.columns:
        raise ValueError("[aps_failure] brak kolumny 'class' po wczytaniu")

    df.to_csv(out_path, index=False)
    print(f"[aps_failure] zapisano {out_path} shape={df.shape}")


# =============== BREAST CANCER ===============

def prepare_breast_cancer():
    """
    data/breast/wdbc.data -> data/breast_cancer.csv

    Plik wdbc.data ma 32 kolumny:
      - id
      - diagnosis (M/B)
      - 30 cech numerycznych

    Target: 'target' (po renamie z diagnosis)
    """
    src_dir = DATA_ROOT / "breast"
    out_path = DATA_ROOT / "breast_cancer.csv"

    if out_path.exists():
        print("[breast_cancer] data/breast_cancer.csv już istnieje, pomijam.")
        return

    data_path = src_dir / "wdbc.data"
    if not data_path.exists():
        raise FileNotFoundError(f"[breast_cancer] nie znalazłem {data_path}")

    print(f"[breast_cancer] wczytuję {data_path}")

    # Uproszczone nazwy cech (nie muszą być idealne, kolejność ważniejsza niż nazwy)
    feature_names = [f"f{i}" for i in range(1, 31)]
    cols = ["id", "diagnosis"] + feature_names

    df = pd.read_csv(data_path, header=None, names=cols)

    # Target = diagnosis
    df = df.rename(columns={"diagnosis": "target"})
    df.to_csv(out_path, index=False)
    print(f"[breast_cancer] zapisano {out_path} shape={df.shape}")


# =============== BANK MARKETING ===============

def prepare_bank_marketing():
    """
    data/bank/bank-full.csv -> data/bank_marketing.csv

    UCI Bank Marketing, separator ';'
    Target: 'y'
    """
    src_dir = DATA_ROOT / "bank"
    out_path = DATA_ROOT / "bank_marketing.csv"

    if out_path.exists():
        print("[bank_marketing] data/bank_marketing.csv już istnieje, pomijam.")
        return

    data_path = src_dir / "bank-full.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"[bank_marketing] nie znalazłem {data_path}")

    print(f"[bank_marketing] wczytuję {data_path}")
    df = pd.read_csv(data_path, sep=";")

    if "y" not in df.columns:
        raise ValueError("[bank_marketing] brak kolumny 'y' (target)")

    df.to_csv(out_path, index=False)
    print(f"[bank_marketing] zapisano {out_path} shape={df.shape}")


# =============== SANTANDER (archieve) ===============

def prepare_santander_small(n_samples: int = 100_000):
    """
    data/archieve/train.csv -> data/santander_small.csv  (subsample)

    Target: 'target'
    """
    src_dir = DATA_ROOT / "archieve"   # tak jak w Twoim screenie
    out_path = DATA_ROOT / "santander_small.csv"

    if out_path.exists():
        print("[santander_small] data/santander_small.csv już istnieje, pomijam.")
        return

    train_path = src_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"[santander_small] nie znalazłem {train_path}")

    print(f"[santander_small] wczytuję {train_path}")
    df = pd.read_csv(train_path)

    if "target" not in df.columns:
        raise ValueError("[santander_small] brak kolumny 'target'")

    # Subsampling powtarzalny
    if len(df) > n_samples:
        # Można też użyć stratify=df["target"], ale to jest wolniejsze na bardzo dużych danych.
        df = df.sample(n=n_samples, random_state=RANDOM_STATE).reset_index(drop=True)
        print(
            f"[santander_small] subsampling do {n_samples} wierszy (random_state={RANDOM_STATE})"
        )

    df.to_csv(out_path, index=False)
    print(f"[santander_small] zapisano {out_path} shape={df.shape}")


def main():
    print("=== Przygotowanie kanonicznych datasetów w data/ ===")
    prepare_adult()
    prepare_credit_default()
    prepare_aps_failure()
    prepare_breast_cancer()
    prepare_bank_marketing()
    prepare_santander_small()
    print("=== Gotowe. Możesz odpalać run.py ===")


if __name__ == "__main__":
    main()
