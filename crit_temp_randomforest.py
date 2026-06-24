import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# ============================================================
# Settings
# ============================================================
MODEL  = "RandomForest"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

# Number of different parameter settings to sample randomly
N_ITER_SEARCH = 15 

# ============================================================
# SCALER SELECTION
# ============================================================
scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "None": None,
}

# ============================================================
# DATA LOADING (Using Pre-split Files)
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
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False, interaction_only=False)
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly   = poly.transform(x_val)
    x_test_poly  = poly.transform(x_test)
    feature_names = poly.get_feature_names_out(x_train.columns)
else:
    x_train_poly = x_train.values
    x_val_poly   = x_val.values
    x_test_poly  = x_test.values
    feature_names = x_train.columns.tolist()

df_train_poly = pd.DataFrame(x_train_poly, columns=feature_names)
df_val_poly   = pd.DataFrame(x_val_poly,   columns=feature_names)
df_test_poly  = pd.DataFrame(x_test_poly,  columns=feature_names)

# ============================================================
# LOAD PRE-CALCULATED MRMR FEATURES
# ============================================================
features_file_path = "Project/results/best_mrmr_features.json"

if not os.path.exists(features_file_path):
    raise FileNotFoundError(f"Could not find '{features_file_path}'. Please run your MRMR script first!")

print(f"Loading pre-calculated features from {features_file_path}...")
with open(features_file_path, "r") as f:
    ranked_features = json.load(f)
print("=" * 50)

# Filter columns
x_train_mrmr = df_train_poly[ranked_features]
x_val_mrmr   = df_val_poly[ranked_features]
x_test_mrmr  = df_test_poly[ranked_features]

# ============================================================
# SCALING
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
# COMBINING TRAIN + VAL FOR RANDOMIZED SEARCH
# RandomizedSearchCV performs internal cross-validation (CV),
# so we pass it the combined dataset.
# ============================================================
x_trainval_search = np.vstack([x_train_scaled, x_val_scaled])
y_trainval_search = pd.concat([y_train, y_val]).reset_index(drop=True)

# ============================================================
# PARAMETER DISTRIBUTION DEFINITION
# ============================================================
# Define the search space. RandomizedSearchCV will pull random setups from here.
param_distributions = {
    'n_estimators': [50, 100, 200, 300],          # Number of trees
    'max_depth': [None, 10, 20, 30],               # Max depth of trees
    'min_samples_split': [2, 5, 10],               # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],                 # Minimum samples at a leaf node
    'max_features': ['sqrt', 'log2', 0.3, 0.5]     # Number of features to consider at each split
}

# Initialize a base Random Forest
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores

# Setup the Randomized Search
rf_random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=N_ITER_SEARCH,
    cv=3,                                          # 3-fold cross validation
    scoring='neg_mean_squared_error',              # Optimize for lowest MSE
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print(f"[Starting Randomized Search trying {N_ITER_SEARCH} different configurations...]")
rf_random_search.fit(x_trainval_search, y_trainval_search)

print(f"\n=> Best Parameters Found:")
print(json.dumps(rf_random_search.best_params_, indent=4))
print("=" * 50)

# Extract the winning model configuration
final_model = rf_random_search.best_estimator_

# ============================================================
# EVALUATION (On Unseen Test Data Only)
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

print(f"=== Final Test Metrics ({MODEL}) ===")
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
    plt.scatter(y_true, y_pred, alpha=0.5, color="forestgreen", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Prediction (y=x)")

    poly_tag = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else ""
    title = f"Real vs. Predicted — {MODEL} (Randomized Search Best Fit)"
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