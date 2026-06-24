import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
#from mrmr import mrmr_regression

# ============================================================
# Settings
# ============================================================
MODEL  = "KNN"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

# K confirmed via validation sweep 
BEST_K = 775

# ============================================================
# MODEL AND SCALER
# ============================================================
models = {
    "Linear": LinearRegression(),
}

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
    print(f"  Original features             : {x_train.shape[1]}") 
    print(f"  Features after expansion      : {x_train_poly.shape[1]}")
    print("=" * 50)
else:
    x_train_poly = x_train.values
    x_val_poly   = x_val.values
    x_test_poly  = x_test.values
    feature_names = x_train.columns.tolist() 

# ============================================================
# SCALING (fit only on train)
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

# ============================================================
# MRMR FEATURE SELECTION
# ============================================================
df_train_mrmr = pd.DataFrame(x_train_scaled, columns=feature_names)
df_val_mrmr   = pd.DataFrame(x_val_scaled,   columns=feature_names)
df_test_mrmr  = pd.DataFrame(x_test_scaled,  columns=feature_names)

features_file_path = "Project/results/best_mrmr_features.json"

print(f"Loading pre-calculated features from {features_file_path}...")
with open(features_file_path, "w" if False else "r") as f: # Open in read mode
    ranked_features = json.load(f)

print(f"Loaded {len(ranked_features)} features successfully. Skipping MRMR recalculation!")



# ============================================================
# BUILD FINAL DATASETS
# ============================================================
x_train_final    = df_train_mrmr[ranked_features].values
x_val_final      = df_val_mrmr[ranked_features].values
x_test_final     = df_test_mrmr[ranked_features].values

# Combine train + val for final model training.
x_trainval_final = np.vstack([x_train_final, x_val_final])
y_trainval       = pd.concat([y_train, y_val]).reset_index(drop=True)

# ============================================================
# TRAINING (on train + val combined)
# ============================================================
print(f"=== Settings ===")
print(f"Model      : {MODEL}")
print(f"Scaler     : {SCALER}")
print(f"Polynomial : degree={POLY_DEGREE}" if USE_POLYNOMIAL else "Polynomial : No")
print(f"Features   : {BEST_K} (selected by MRMR)")
print(f"Train size (train+val) : {x_trainval_final.shape[0]}")
print("-" * 40)

model = models[MODEL]
model.fit(x_trainval_final, y_trainval)

if hasattr(model, 'intercept_'):
    print(f"Intercept: {model.intercept_:.4f}")
print("-" * 40)

# ============================================================
# EVALUATION (on unseen test data only)
# ============================================================
y_pred = model.predict(x_test_final)

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
    plt.scatter(y_true, y_pred, alpha=0.5, color="blue", edgecolor="k")

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