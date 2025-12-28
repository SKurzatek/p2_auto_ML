import time
import pandas as pd

from automl import MiniAutoML

# === choose ONE dataset to test ===
# CSV_PATH = "test_datasets/banknote_authentication.csv"
# CSV_PATH = "test_datasets/sonar.csv"
# CSV_PATH = "test_datasets/ionosphere.csv"
CSV_PATH = "test_datasets/mammographic_mass.csv"
# CSV_PATH = "test_datasets/spambase.csv"
TARGET_COL = "target"

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

automl = MiniAutoML(
    models_config="portfolio/models.json",
    report_or_profiles_path="portfolio/portfolio_report.json",
    report_path_for_evidence="portfolio/portfolio_report.json",
    random_state=42,
    top_k_models=5,
    shortlist_size=12,
    test_size=0.2,
    cv_folds_stack=3,
)

t0 = time.perf_counter()
automl.fit(X, y)
print("ensemble_mode:", automl.ensemble_mode_)
t1 = time.perf_counter()

print("\n=== MiniAutoML results ===")
print("dataset:", CSV_PATH)
print("fit_time_sec:", round(t1 - t0, 3))

print("\nensemble_mode:", automl.ensemble_mode_)
print("val_best:", automl.validation_score_best_)
print("val_avg:", automl.validation_score_avg_)
print("val_mlp_10:", automl.validation_score_mlp_10_)
print("val_mlp_10_10:", automl.validation_score_mlp_10_10_)
print("val_mlp_10_10_10:", automl.validation_score_mlp_10_10_10_)
print("stacker_used:", automl.stacker_ is not None)

print("\n=== chosen models (top 5) ===")
for i, (pm, _) in enumerate(automl.selected_models_, start=1):
    print(f"{i}. id={pm.id} type={pm.model_type} params={pm.params}")

print("\n=== new dataset profile (train split) ===")
print(automl.new_dataset_profile_)