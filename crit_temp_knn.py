import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# ============================================================
# Settings
# ============================================================
MODEL  = "KNN"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

# We can test a dense range of neighbors to find the sweet spot
NEIGHBORS_CANDIDATES = [1, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 40, 50]

# ============================================================
# SCALER SELECTION
# ============================================================
scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "None": None,
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
else:
    x_train_poly = x_train.values
    x_val_poly   = x_val.values
    x_test_poly  = x_test.values
    feature_names = x_train.columns.tolist()

# Convert back to temporary DataFrames so we can match MRMR feature names easily
df_train_poly = pd.DataFrame(x_train_poly, columns=feature_names)
df_val_poly   = pd.DataFrame(x_val_poly,   columns=feature_names)
df_test_poly  = pd.DataFrame(x_test_poly,  columns=feature_names)

# ============================================================
# LOAD PRE-CALCULATED MRMR FEATURES
# ============================================================
features_file_path = "Project/results/best_mrmr_features.json"

NUM_FEATURES_TO_USE = 775

if not os.path.exists(features_file_path):
    raise FileNotFoundError(
        f"Could not find '{features_file_path}'. Please run your MRMR selection "
        "script first and save the JSON feature list!"
    )

print(f"Loading pre-calculated features from {features_file_path}...")
with open(features_file_path, "r") as f:
    ranked_features = json.load(f)

ranked_features = ranked_features[:NUM_FEATURES_TO_USE]

print(f"Loaded {len(ranked_features)} features successfully. Skipping MRMR recalculation!")
print("=" * 50)

# Keep only the columns selected by MRMR from our expanded feature sets
x_train_mrmr = df_train_poly[ranked_features]
x_val_mrmr   = df_val_poly[ranked_features]
x_test_mrmr  = df_test_poly[ranked_features]

# ============================================================
# SCALING (Crucial for KNN distance metrics)
# ============================================================
scaler = scalers[SCALER]

if scaler is not None:
    x_train_scaled = scaler.fit_transform(x_train_mrmr)
    x_val_scaled   = scaler.transform(x_val_mrmr)
    x_test_scaled  = scaler.transform(x_test_mrmr)
else:
    x_train_scaled = x_train_mrmr.values
    x_val_scaled   = x_val_mrmr.values
    x_test_scaled  = x_test_mrmr.values

# ============================================================
# TUNING KNN VIA VALIDATION LOOP
# ============================================================
best_k = None
best_val_rmse = float('inf')

print("[Validating KNN Hyperparameters]")
for k in NEIGHBORS_CANDIDATES:
    temp_knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    temp_knn.fit(x_train_scaled, y_train)
    
    val_preds = temp_knn.predict(x_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    print(f"  Tested Neighbors k={k:2d} | Validation RMSE: {val_rmse:.4f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_k = k

print(f"\n=> Best KNN Hyperparameter selected: k={best_k} (Validation RMSE: {best_val_rmse:.4f})")
print("=" * 50)

# ============================================================
# FINAL TRAINING (Combine Train + Val)
# ============================================================
x_trainval_final = np.vstack([x_train_scaled, x_val_scaled])
y_trainval       = pd.concat([y_train, y_val]).reset_index(drop=True)

print(f"=== Final Model Settings ===")
print(f"Model      : {MODEL} Regressor")
print(f"Scaler     : {SCALER}")
print(f"Polynomial : degree={POLY_DEGREE}" if USE_POLYNOMIAL else "Polynomial : No")
print(f"Neighbors  : k={best_k}")
print(f"Features   : {len(ranked_features)}")
print(f"Final Train Size (train+val): {x_trainval_final.shape[0]}")
print("-" * 40)

final_model = KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
final_model.fit(x_trainval_final, y_trainval)

# ============================================================
# EVALUATION 
# ============================================================
y_pred = final_model.predict(x_test_scaled)

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
    print(f"Negative predictions (physically impossible): {n_neg} ({100 * n_neg / len(y_pred):.1f}%)")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

metrics = evaluate_model(y_test, y_pred)

# ============================================================
# PLOT RESULTS
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
    title = f"Real vs. Predicted — {MODEL} (k={best_k}, {SCALER}{poly_tag})"
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