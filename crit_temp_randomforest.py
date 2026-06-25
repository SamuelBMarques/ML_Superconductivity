import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)

# ============================================================
# Settings
# ============================================================
MODEL  = "RandomForest"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

# How many random parameter combinations to try
N_ITER_SEARCH = 30

# Paths
selection_file_path = "Project/results/best_selection_features.json"
rf_cache_path       = "Project/results/best_rf_params.json"

# ============================================================
# SCALERS
# ============================================================
scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "None"    : None,
}

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_csv("Project/splits/train.csv")
val   = pd.read_csv("Project/splits/val.csv")
test  = pd.read_csv("Project/splits/test.csv")

x_train, y_train = train.drop(columns=["critical_temp"]), train["critical_temp"]
x_val,   y_val   = val.drop(columns=["critical_temp"]),   val["critical_temp"]
x_test,  y_test  = test.drop(columns=["critical_temp"]),  test["critical_temp"]

print(f"Train size : {x_train.shape[0]}")
print(f"Val size   : {x_val.shape[0]}")
print(f"Test size  : {x_test.shape[0]}")
print("=" * 50)

# ============================================================
# POLYNOMIAL FEATURE EXPANSION
# ============================================================
if USE_POLYNOMIAL:
    poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        include_bias=False,
        interaction_only=False
    )
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly   = poly.transform(x_val)
    x_test_poly  = poly.transform(x_test)

    feature_names = poly.get_feature_names_out(x_train.columns)

    print(f"[PolynomialFeatures degree={POLY_DEGREE}]")
    print(f"  Original features        : {x_train.shape[1]}")
    print(f"  Features after expansion : {x_train_poly.shape[1]}")
    print("=" * 50)
else:
    x_train_poly  = x_train.values
    x_val_poly    = x_val.values
    x_test_poly   = x_test.values
    feature_names = x_train.columns.tolist()

# ============================================================
# SCALING — fit only on train
# Random Forest is scale-invariant, but scaling is kept here
# so the pipeline is identical across all models and the same
# saved feature names apply directly to the scaled dataframes.
# ============================================================
scaler = scalers[SCALER]

if scaler is not None:
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_val_scaled   = scaler.transform(x_val_poly)
    x_test_scaled  = scaler.transform(x_test_poly)
else:
    x_train_scaled = x_train_poly
    x_val_scaled   = x_val_poly
    x_test_scaled  = x_test_poly

df_train_scaled = pd.DataFrame(x_train_scaled, columns=feature_names)
df_val_scaled   = pd.DataFrame(x_val_scaled,   columns=feature_names)
df_test_scaled  = pd.DataFrame(x_test_scaled,  columns=feature_names)

# ============================================================
# LOAD PRE-CALCULATED SELECTION FEATURES
# ============================================================
if not os.path.exists(selection_file_path):
    raise FileNotFoundError(
        f"Could not find '{selection_file_path}'. "
        "Run crit_temp_linearregression.py first to generate the selection."
    )

print(f"Loading selection features from {selection_file_path}...")
with open(selection_file_path, "r") as f:
    saved_selection = json.load(f)

best_features    = saved_selection["features"]
selection_method = saved_selection["method"]
selection_rmse   = saved_selection["val_rmse"]

print(f"Loaded {len(best_features)} features "
      f"(method: {selection_method}, LinearRegression Val RMSE: {selection_rmse:.4f})")
print("=" * 50)

x_train_final = df_train_scaled[best_features].values
x_val_final   = df_val_scaled[best_features].values
x_test_final  = df_test_scaled[best_features].values

# ============================================================
# HYPERPARAMETER SEARCH SPACE
#
# Grouped by what they control:
#
# Ensemble size
#   n_estimators   — number of trees; more = more stable, slower
#
# Tree structure (how deep / complex each tree grows)
#   max_depth      — maximum depth per tree; None = fully grown
#   min_samples_split — minimum samples required to split a node
#   min_samples_leaf  — minimum samples required at a leaf node
#   max_leaf_nodes    — maximum number of leaves per tree
#
# Feature usage (how RF samples features at each split)
#   max_features   — features considered at each split:
#                    'sqrt' ~ classic RF, 'log2' = more aggressive
#                    reduction, float = fraction, None = all features
#
# Sample usage (how RF samples rows per tree)
#   bootstrap      — True: each tree trained on a bootstrap sample
#                    False: each tree trained on the full dataset
#   max_samples    — fraction of training rows per tree
#                    (only active when bootstrap=True)
# ============================================================
param_distributions = {
    "n_estimators"     : [50, 100, 200, 300, 500],
    "max_depth"        : [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 4, 8],
    "max_leaf_nodes"   : [None, 50, 100, 200],
    "max_features"     : ["sqrt", "log2", 0.2, 0.4, 0.6, None],
    "bootstrap"        : [True, False],
    "max_samples"      : [None, 0.6, 0.7, 0.8, 0.9],
}

# ============================================================
# RANDOM HYPERPARAMETER SEARCH — validated on val set
# Same pattern as KNN: tune on train, evaluate on val,
# then cache. Delete the JSON file to force a re-run.
# ============================================================
if os.path.exists(rf_cache_path):
    print(f"Loading cached RF parameters from {rf_cache_path}...")
    with open(rf_cache_path, "r") as f:
        rf_cache = json.load(f)
    best_params   = rf_cache["params"]
    best_val_rmse = rf_cache["val_rmse"]
    print(f"Best params loaded (Val RMSE: {best_val_rmse:.4f}). Skipping search.")
    print("=" * 50)

else:
    print(f"[Random Search — {N_ITER_SEARCH} configurations, evaluated on val set]")
    print("=" * 50)

    random.seed(78)
    best_params   = None
    best_val_rmse = float("inf")

    for i in range(N_ITER_SEARCH):
        # Sample one random combination
        params = {k: random.choice(v) for k, v in param_distributions.items()}

        # max_samples only applies when bootstrap=True; avoid sklearn ValueError
        if not params["bootstrap"]:
            params["max_samples"] = None

        rf = RandomForestRegressor(**params, random_state=78, n_jobs=-1)
        rf.fit(x_train_final, y_train)

        val_preds = rf.predict(x_val_final)
        val_rmse  = np.sqrt(mean_squared_error(y_val, val_preds))

        status = " <=== best so far" if val_rmse < best_val_rmse else ""
        print(f"  [{i+1:2d}/{N_ITER_SEARCH}]  Val RMSE: {val_rmse:.4f}{status}")
        print(f"          {params}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params   = params

    print(f"\n=> Best configuration (Val RMSE: {best_val_rmse:.4f}):")
    print(json.dumps(best_params, indent=4))
    print("=" * 50)

    os.makedirs("Project/results", exist_ok=True)
    with open(rf_cache_path, "w") as f:
        json.dump({"params": best_params, "val_rmse": best_val_rmse}, f, indent=2)
    print(f"Best parameters saved to {rf_cache_path}")

# ============================================================
# FINAL TRAINING — refit scaler on train+val, then train RF
# ============================================================
x_trainval_poly = np.vstack([x_train_poly, x_val_poly])
y_trainval      = pd.concat([y_train, y_val]).reset_index(drop=True)

if scaler is not None:
    final_scaler        = sklearn.preprocessing.StandardScaler()
    x_trainval_scaled_f = final_scaler.fit_transform(x_trainval_poly)
    x_test_scaled_f     = final_scaler.transform(x_test_poly)
else:
    x_trainval_scaled_f = x_trainval_poly
    x_test_scaled_f     = x_test_poly

df_trainval_f = pd.DataFrame(x_trainval_scaled_f, columns=feature_names)
df_test_f     = pd.DataFrame(x_test_scaled_f,     columns=feature_names)

x_trainval_final = df_trainval_f[best_features].values
x_test_final_f   = df_test_f[best_features].values

print(f"\n=== Final Training ===")
print(f"Model      : {MODEL}")
print(f"Scaler     : {SCALER}")
print(f"Polynomial : degree={POLY_DEGREE}" if USE_POLYNOMIAL else "Polynomial : No")
print(f"Features   : {len(best_features)} (from {selection_method} selection)")
print(f"Train size : {x_trainval_final.shape[0]} (train+val)")
print(f"Parameters : {best_params}")
print("-" * 40)

final_model = RandomForestRegressor(**best_params, random_state=78, n_jobs=-1)
final_model.fit(x_trainval_final, y_trainval)

# ============================================================
# EVALUATION (test set — never touched until here)
# ============================================================
y_pred = final_model.predict(x_test_final_f)


def evaluate_model(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"MSE  : {mse:.4f} K²")
    print(f"RMSE : {rmse:.4f} K")
    print(f"MAE  : {mae:.4f} K")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.4f}")

    n_neg = (y_pred < 0).sum()
    print(f"Negative predictions (physically impossible): "
          f"{n_neg} ({100 * n_neg / len(y_pred):.1f}%)")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


print(f"=== Final Test Metrics ({MODEL}) ===")
metrics = evaluate_model(y_test, y_pred)

# ============================================================
# PLOT
# ============================================================
def plot_predictions(y_true, y_pred):
    comparison_df = pd.DataFrame({
        "Real Value"     : y_true.values,
        "Predicted"      : y_pred,
        "Absolute Error" : abs(y_true.values - y_pred),
    })
    print(f"\n--- First 15 samples from the test set ---")
    print(comparison_df.head(15))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="forestgreen", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=2,
        label="Perfect Prediction (y=x)"
    )

    poly_tag = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else ""
    title    = f"Real vs. Predicted — {MODEL} ({SCALER}{poly_tag})"
    plt.title(title)
    plt.xlabel("Real Value (K)")
    plt.ylabel("Predicted Value (K)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    os.makedirs("Project/results", exist_ok=True)
    name = f"Project/results/grafico_{MODEL.lower()}_{SCALER.lower()}{poly_tag}.png"
    plt.savefig(name, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {name}")


plot_predictions(y_test, y_pred)