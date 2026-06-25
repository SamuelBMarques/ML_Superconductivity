import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)

# ============================================================
# Settings
# ============================================================
MODEL  = "KNN"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

NEIGHBORS_CANDIDATES = [1, 3, 4, 5, 6, 7, 9, 11, 13, 15, 20, 25, 30, 40, 50]

# Paths
selection_file_path = "Project/results/best_selection_features.json"
knn_cache_path      = "Project/results/best_knn_k.json"

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
# Scaling happens before feature selection to match the pipeline
# in which the selection features were originally computed.
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
# These were determined by forward selection (LinearRegression
# proxy) over the top 100 MRMR features and saved to JSON.
# ============================================================
if not os.path.exists(selection_file_path):
    raise FileNotFoundError(
        f"Could not find '{selection_file_path}'. "
        "Run crit_temp_linearregression.py first to generate the selection."
    )

print(f"Loading selection features from {selection_file_path}...")
with open(selection_file_path, "r") as f:
    saved_selection = json.load(f)

best_features   = saved_selection["features"]
selection_method = saved_selection["method"]
selection_rmse  = saved_selection["val_rmse"]

print(f"Loaded {len(best_features)} features "
      f"(method: {selection_method}, LinearRegression Val RMSE: {selection_rmse:.4f})")
print("=" * 50)

# Apply feature selection
x_train_final = df_train_scaled[best_features].values
x_val_final   = df_val_scaled[best_features].values
x_test_final  = df_test_scaled[best_features].values

# ============================================================
# KNN HYPERPARAMETER TUNING (k) — cached after first run
# ============================================================
if os.path.exists(knn_cache_path):
    print(f"Loading cached KNN hyperparameters from {knn_cache_path}...")
    with open(knn_cache_path, "r") as f:
        knn_cache = json.load(f)
    best_k        = knn_cache["best_k"]
    best_val_rmse = knn_cache["val_rmse"]
    print(f"Best k={best_k} (Val RMSE: {best_val_rmse:.4f}). Skipping tuning.")
    print("=" * 50)

else:
    print("[Tuning KNN: k neighbors]")
    best_k        = None
    best_val_rmse = float("inf")

    for k in NEIGHBORS_CANDIDATES:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(x_train_final, y_train)

        val_preds = knn.predict(x_val_final)
        val_rmse  = np.sqrt(mean_squared_error(y_val, val_preds))

        print(f"  k={k:2d}  |  Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_k        = k

    print(f"\n  => Best k={best_k}  |  Val RMSE: {best_val_rmse:.4f}")
    print("=" * 50)

    os.makedirs("Project/results", exist_ok=True)
    with open(knn_cache_path, "w") as f:
        json.dump({"best_k": best_k, "val_rmse": best_val_rmse}, f, indent=2)
    print(f"KNN hyperparameters saved to {knn_cache_path}")

# ============================================================
# FINAL TRAINING — refit scaler on train+val, then train KNN
#
# The scaler used during k-tuning was fit on train only (correct
# for hyperparameter selection). For the final model we refit on
# train+val so the full available data informs the scaling.
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
print(f"Neighbors  : k={best_k}")
print(f"Train size : {x_trainval_final.shape[0]} (train+val)")
print("-" * 40)

final_model = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
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
    plt.scatter(y_true, y_pred, alpha=0.5, color="teal", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=2,
        label="Perfect Prediction (y=x)"
    )

    poly_tag = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else ""
    title    = f"Real vs. Predicted — {MODEL} (k={best_k}, {SCALER}{poly_tag})"
    plt.title(title)
    plt.xlabel("Real Value (K)")
    plt.ylabel("Predicted Value (K)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    os.makedirs("Project/results", exist_ok=True)
    name = f"Project/results/grafico_{MODEL.lower()}_k{best_k}_{SCALER.lower()}{poly_tag}.png"
    plt.savefig(name, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {name}")


plot_predictions(y_test, y_pred)