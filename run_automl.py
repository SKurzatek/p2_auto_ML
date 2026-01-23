import time
import pandas as pd

from automl import MiniAutoML

# === choose ONE dataset to test ===
# CSV_PATH = "test_datasets/banknote_authentication.csv"
# CSV_PATH = "test_datasets/sonar.csv"
# CSV_PATH = "test_datasets/ionosphere.csv"
CSV_PATH = "data/titanic.csv"
# CSV_PATH = "test_datasets/spambase.csv"
TARGET_COL = "Survived"

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
print("stacker_used:", automl.stacker_ is not None)

print("\n=== chosen models (top 5) ===")
for i, (pm, _) in enumerate(automl.selected_models_, start=1):
    print(f"{i}. id={pm.id} type={pm.model_type} params={pm.params}")



print("\n=== new dataset profile (train split) ===")
print(automl.new_dataset_profile_)

# ... (reszta kodu bez zmian do linii z print("\n=== MiniAutoML results ==="))

print("\n=== MiniAutoML results ===")
print("dataset:", CSV_PATH)
print("fit_time_sec:", round(t1 - t0, 3))

# ... (existing code up to print("\n=== MiniAutoML results ==="))

print("\n=== MiniAutoML results ===")
print("dataset:", CSV_PATH)
print("fit_time_sec:", round(t1 - t0, 3))

# --- NEW SECTION: DETAILED MODEL PROPOSAL ---
print("\n>>> SELECTED MODEL PROPOSAL <<<")
mode = automl.ensemble_mode_

if mode == "best":
    # Identify the specific single model from the portfolio that won
    best_pm, _ = automl.selected_models_[automl.best_model_idx_]
    print(f"Strategy Type: Best Single Model")
    print(f"Algorithm:     {best_pm.model_type}")
    print(f"Hyperparameters: {best_pm.params}")

elif mode == "avg":
    # Describe the simple averaging ensemble
    print(f"Strategy Type: Simple Averaging Ensemble")
    print(f"Description:   Combined probabilities from the top {len(automl.selected_models_)} models.")

elif mode is not None and mode.startswith("stacker_"):
    # Describe the meta-model used for stacking
    print(f"Strategy Type: Stacking Ensemble")
    print(f"Meta-learner:  {type(automl.stacker_).__name__}")
    print(f"Internal ID:   {mode}")
    # If using a linear meta-learner, show the weights assigned to each base model
    if hasattr(automl.stacker_, "coef_"):
        print(f"Base Model Weights: {automl.stacker_.coef_}")

print("-" * 30)
# --- END OF NEW SECTION ---

print("val_best:", automl.validation_score_best_)
print("val_avg:", automl.validation_score_avg_)

# Display scores for all tested stackers if the attribute exists
if hasattr(automl, 'all_stacker_scores_'):
    print("\nStacker Validation Scores:")
    for s_name, s_val in automl.all_stacker_scores_.items():
        print(f" - {s_name}: {round(s_val, 4)}")

# check if predict works:

y_pred = automl.predict(X)
print("\nPredictions on training data (first 10):", y_pred[:10])
y_proba = automl.predict_proba(X)
print("Predicted probabilities on training data (first 10):", y_proba[:10])
# ... (remaining code for top 5 models and profile)