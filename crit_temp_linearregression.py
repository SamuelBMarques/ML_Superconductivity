import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# Settings
# ============================================================
MODEL  = "Linear"
SCALER = "Standard"

USE_POLYNOMIAL = True
POLY_DEGREE    = 2

BEST_K         = 775   # Confirmed MRMR feature count (full pipeline)
SELECTION_POOL = 100   # Top N MRMR features used as the candidate pool
                       # for forward / backward selection

# ============================================================
# MODELS AND SCALERS
# ============================================================
models = {
    "Linear": LinearRegression(),
}

scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "None"    : None,
}

# ============================================================
# DATA LOADING (pre-split files — never re-split)
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
# LOAD PRE-CALCULATED MRMR FEATURES
# ============================================================
df_train_mrmr = pd.DataFrame(x_train_scaled, columns=feature_names)
df_val_mrmr   = pd.DataFrame(x_val_scaled,   columns=feature_names)
df_test_mrmr  = pd.DataFrame(x_test_scaled,  columns=feature_names)

features_file_path = "Project/results/best_mrmr_features.json"
selection_file_path   = "Project/results/best_selection_features.json"

print(f"Loading MRMR features from {features_file_path}...")
with open(features_file_path, "r") as f:
    ranked_features = json.load(f)

print(f"Loaded {len(ranked_features)} features.")
print(f"Using top {SELECTION_POOL} as candidate pool for wrapper selection.")
print("=" * 50)

# ============================================================
# FORWARD / BACKWARD SELECTION
# ============================================================
candidate_features = ranked_features[:SELECTION_POOL]

df_train_fs = df_train_mrmr[candidate_features]
df_val_fs   = df_val_mrmr[candidate_features]


def _val_rmse(x_train_df, y_train, x_val_df, y_val, features):
    """Fit LinearRegression on features and return validation RMSE."""
    m = LinearRegression()
    m.fit(x_train_df[features], y_train)
    preds = m.predict(x_val_df[features])
    return np.sqrt(mean_squared_error(y_val, preds))


def forward_selection(x_train_df, x_val_df, y_train, y_val, candidates):
    """
    Start from an empty set and greedily add the feature that
    most reduces validation RMSE at each step. Stop when no
    remaining feature improves the current RMSE.
    """
    selected  = []
    remaining = list(candidates)
    best_rmse = float("inf")

    print(f"\n{'='*55}")
    print(f"[Forward Selection]  Pool: {len(candidates)} features")
    print(f"{'='*55}")

    while remaining:
        step_best_feature = None
        step_best_rmse    = float("inf")

        for feature in remaining:
            rmse = _val_rmse(x_train_df, y_train, x_val_df, y_val,
                             selected + [feature])
            if rmse < step_best_rmse:
                step_best_rmse    = rmse
                step_best_feature = feature

        if step_best_rmse < best_rmse:
            best_rmse = step_best_rmse
            selected.append(step_best_feature)
            remaining.remove(step_best_feature)
            print(f"  + [{len(selected):3d} features]  "
                  f"Added '{step_best_feature[:45]}'  |  Val RMSE: {best_rmse:.4f}")
        else:
            print(f"\n  No improvement found. Stopping at {len(selected)} features.")
            break

    print(f"\n  => Result: {len(selected)} features  |  Val RMSE: {best_rmse:.4f}")
    return selected, best_rmse


def backward_elimination(x_train_df, x_val_df, y_train, y_val, candidates):
    """
    Start from the full candidate set and greedily remove the
    feature whose removal most reduces validation RMSE at each
    step. Stop when removing any feature no longer improves RMSE.
    """
    selected  = list(candidates)
    best_rmse = _val_rmse(x_train_df, y_train, x_val_df, y_val, selected)

    print(f"\n{'='*55}")
    print(f"[Backward Elimination]  Pool: {len(candidates)} features")
    print(f"  Start  |  Val RMSE: {best_rmse:.4f}")
    print(f"{'='*55}")

    while len(selected) > 1:
        step_best_feature = None
        step_best_rmse    = float("inf")

        for feature in selected:
            temp = [f for f in selected if f != feature]
            rmse = _val_rmse(x_train_df, y_train, x_val_df, y_val, temp)
            if rmse < step_best_rmse:
                step_best_rmse    = rmse
                step_best_feature = feature

        if step_best_rmse < best_rmse:
            best_rmse = step_best_rmse
            selected.remove(step_best_feature)
            print(f"  - [{len(selected):3d} features]  "
                  f"Removed '{step_best_feature[:45]}'  |  Val RMSE: {best_rmse:.4f}")
        else:
            print(f"\n  No improvement found. Stopping at {len(selected)} features.")
            break

    print(f"\n  => Result: {len(selected)} features  |  Val RMSE: {best_rmse:.4f}")
    return selected, best_rmse


# ============================================================
# FORWARD / BACKWARD SELECTION (run once, then cached)
# ============================================================
candidate_features = ranked_features[:SELECTION_POOL]

df_train_fs = df_train_mrmr[candidate_features]
df_val_fs   = df_val_mrmr[candidate_features]

if os.path.exists(selection_file_path):
    print(f"Loading cached selection features from {selection_file_path}...")
    with open(selection_file_path, "r") as f:
        saved = json.load(f)
    best_features = saved["features"]
    best_method   = saved["method"]
    best_rmse_val = saved["val_rmse"]
    print(f"Loaded {len(best_features)} features "
          f"(method: {best_method}, Val RMSE: {best_rmse_val:.4f}). "
          f"Skipping selection.")
    print("=" * 55)

else:
    print("No cached selection found. Running forward and backward selection...")

    features_fwd, rmse_fwd = forward_selection(
        df_train_fs, df_val_fs, y_train, y_val, candidate_features
    )
    features_bwd, rmse_bwd = backward_elimination(
        df_train_fs, df_val_fs, y_train, y_val, candidate_features
    )

    print(f"\n{'='*55}")
    print(f"[Comparison]")
    print(f"  Forward selection    : {len(features_fwd):3d} features  |  Val RMSE: {rmse_fwd:.4f}")
    print(f"  Backward elimination : {len(features_bwd):3d} features  |  Val RMSE: {rmse_bwd:.4f}")

    if rmse_fwd <= rmse_bwd:
        best_features = features_fwd
        best_method   = "Forward"
        best_rmse_val = rmse_fwd
    else:
        best_features = features_bwd
        best_method   = "Backward"
        best_rmse_val = rmse_bwd

    print(f"\n  => Winner: {best_method} "
          f"({len(best_features)} features  |  Val RMSE: {best_rmse_val:.4f})")
    print(f"{'='*55}")

    # Save so future runs skip the search entirely
    os.makedirs("Project/results", exist_ok=True)
    with open(selection_file_path, "w") as f:
        json.dump({
            "method"   : best_method,
            "val_rmse" : best_rmse_val,
            "features" : best_features
        }, f, indent=2)
    print(f"Selection results saved to {selection_file_path}")

# ============================================================
# BUILD FINAL DATASETS
# ============================================================
x_train_final    = df_train_mrmr[best_features].values
x_val_final      = df_val_mrmr[best_features].values
x_test_final     = df_test_mrmr[best_features].values

# Val set has served its purpose (feature selection) — merge it
# back into training before the final model fit.
x_trainval_final = np.vstack([x_train_final, x_val_final])
y_trainval       = pd.concat([y_train, y_val]).reset_index(drop=True)

# ============================================================
# FINAL TRAINING (train + val combined)
# ============================================================
print(f"\n=== Final Training ===")
print(f"Model      : {MODEL}")
print(f"Scaler     : {SCALER}")
print(f"Polynomial : degree={POLY_DEGREE}" if USE_POLYNOMIAL else "Polynomial : No")
print(f"Selection  : {best_method} ({len(best_features)} features)")
print(f"Train size : {x_trainval_final.shape[0]}")
print("-" * 40)

model = models[MODEL]
model.fit(x_trainval_final, y_trainval)

if hasattr(model, "intercept_"):
    print(f"Intercept: {model.intercept_:.4f}")
print("-" * 40)

# ============================================================
# EVALUATION (test set — never touched until here)
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
    plt.scatter(y_true, y_pred, alpha=0.5, color="blue", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=2,
        label="Perfect Prediction (y=x)"
    )

    poly_tag   = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else ""
    method_tag = "fwd" if best_method == "Forward" else "bwd"
    title      = f"Real vs. Predicted — {MODEL} ({SCALER}{poly_tag}, {method_tag})"

    plt.title(title)
    plt.xlabel("Real Value (K)")
    plt.ylabel("Predicted Value (K)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    os.makedirs("Project/results", exist_ok=True)
    name = (f"Project/results/grafico_{MODEL.lower()}_"
            f"{SCALER.lower()}{poly_tag}_{method_tag}.png")
    plt.savefig(name, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {name}")


plot_predictions(y_test, y_pred)