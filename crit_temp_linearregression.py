import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# Settings / Pipeline Configuration Toggles
#
# Valid combinations:
#   Baseline (no engineering)  : USE_POLYNOMIAL=False, USE_MRMR=False, USE_WRAPPER_SELECTION=False
#   MRMR only                  : USE_POLYNOMIAL=True,  USE_MRMR=True,  USE_WRAPPER_SELECTION=False
#   MRMR + Wrapper             : USE_POLYNOMIAL=True,  USE_MRMR=True,  USE_WRAPPER_SELECTION=True
#   Wrapper on raw features    : USE_POLYNOMIAL=False, USE_MRMR=False, USE_WRAPPER_SELECTION=True
#
# Invalid: USE_MRMR=True with USE_POLYNOMIAL=False
#   The saved MRMR features were ranked from the polynomial
#   space and contain interaction/squared term names that do
#   not exist in the original 81-feature space.
# ============================================================
MODEL  = "Linear"
SCALER = "Standard"

USE_POLYNOMIAL        = True
POLY_DEGREE           = 2

USE_MRMR              = True
BEST_K                = 775    # Features used when USE_WRAPPER_SELECTION=False
SELECTION_POOL        = 100    # Candidate pool size when USE_WRAPPER_SELECTION=True

USE_WRAPPER_SELECTION = False

# ============================================================
# GUARD — catch invalid combination early
# ============================================================
if USE_MRMR and not USE_POLYNOMIAL:
    raise ValueError(
        "USE_MRMR=True requires USE_POLYNOMIAL=True because the saved MRMR "
        "features were ranked from the polynomial-expanded space. Either enable "
        "polynomial expansion or run MRMR again on the original 81 features."
    )

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
# 1. POLYNOMIAL FEATURE EXPANSION (fit on train only)
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
# 2. SCALING (fit only on train to prevent leakage)
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

df_train_processed = pd.DataFrame(x_train_scaled, columns=feature_names)
df_val_processed   = pd.DataFrame(x_val_scaled,   columns=feature_names)
df_test_processed  = pd.DataFrame(x_test_scaled,  columns=feature_names)

# ============================================================
# HELPER FUNCTIONS FOR WRAPPER SELECTION
# ============================================================
def _val_rmse(x_train_df, y_train, x_val_df, y_val, features):
    """Fit LinearRegression on features and return validation RMSE."""
    m = LinearRegression()
    m.fit(x_train_df[features], y_train)
    preds = m.predict(x_val_df[features])
    return np.sqrt(mean_squared_error(y_val, preds))


def forward_selection(x_train_df, x_val_df, y_train, y_val, candidates):
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
# 3. FEATURE SELECTION PIPELINE
# ============================================================
features_file_path  = "Project/results/best_mrmr_features.json"
poly_suffix         = "_poly" if USE_POLYNOMIAL else ""
selection_file_path = f"Project/results/best_selection_features{poly_suffix}.json"

if USE_MRMR:
    print(f"Loading MRMR features from {features_file_path}...")
    with open(features_file_path, "r") as f:
        ranked_features = json.load(f)
    print(f"Loaded {len(ranked_features)} features.")

    if USE_WRAPPER_SELECTION:
        candidate_features = ranked_features[:SELECTION_POOL]
        print(f"Using top {SELECTION_POOL} MRMR features as candidate pool for wrapper selection.")
    else:
        best_features = ranked_features[:BEST_K]
        best_method   = "MRMR"
        print(f"Using top {len(best_features)} MRMR features directly (skipping wrapper selection).")
else:
    candidate_features = list(feature_names)
    if not USE_WRAPPER_SELECTION:
        best_features = list(feature_names)
        best_method   = "None"
        print("Using all available features directly (no feature selection).")

if USE_WRAPPER_SELECTION:
    if os.path.exists(selection_file_path):
        print(f"Loading cached selection features from {selection_file_path}...")
        with open(selection_file_path, "r") as f:
            saved = json.load(f)

        # Validate that every cached feature exists in the current feature space.
        # A mismatch means the cache was created with different pipeline settings
        # (e.g. different USE_POLYNOMIAL). Treat it as stale and regenerate.
        missing = [f for f in saved["features"] if f not in set(feature_names)]
        if missing:
            print(f"WARNING: cached selection contains {len(missing)} features that "
                  f"do not exist in the current feature space.")
            print(f"  First missing: '{missing[0]}'")
            print(f"  This cache was likely created with different pipeline settings.")
            print(f"  Deleting stale cache and re-running selection...")
            os.remove(selection_file_path)
            # fall through to the else block below
        else:
            best_features = saved["features"]
            best_method   = saved["method"]
            best_rmse_val = saved["val_rmse"]
            print(f"Loaded {len(best_features)} features "
                  f"(method: {best_method}, Val RMSE: {best_rmse_val:.4f}). "
                  f"Skipping selection.")
            print("=" * 55)

    if not os.path.exists(selection_file_path):  # covers both missing and just-deleted
        print("Running forward and backward selection...")
        features_fwd, rmse_fwd = forward_selection(
            df_train_processed, df_val_processed, y_train, y_val, candidate_features
        )
        features_bwd, rmse_bwd = backward_elimination(
            df_train_processed, df_val_processed, y_train, y_val, candidate_features
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

        os.makedirs("Project/results", exist_ok=True)
        with open(selection_file_path, "w") as f:
            json.dump({
                "method"   : best_method,
                "val_rmse" : best_rmse_val,
                "features" : best_features
            }, f, indent=2)
        print(f"Selection results saved to {selection_file_path}")

# ============================================================
# 4. UNIQUE OUTPUT FOLDER — one per pipeline configuration
#
# Naming pattern:  linear[_poly2][_mrmr775|_mrmr100][_forward|_backward|_none]
# Examples:
#   linear                            — baseline, no engineering
#   linear_poly2_mrmr775              — poly + MRMR top-775, no wrapper
#   linear_poly2_mrmr100_forward      — poly + MRMR top-100 pool + forward selection
#   linear_poly2_mrmr100_backward     — poly + MRMR top-100 pool + backward elimination
# ============================================================
parts = ["linear"]
if USE_POLYNOMIAL:
    parts.append(f"poly{POLY_DEGREE}")
if USE_MRMR:
    k_label = SELECTION_POOL if USE_WRAPPER_SELECTION else BEST_K
    parts.append(f"mrmr{k_label}")
if USE_WRAPPER_SELECTION:
    parts.append(best_method.lower())   # "forward" or "backward"
elif not USE_MRMR:
    parts.append("none")                # baseline: no selection at all

subfolder  = "_".join(parts)
output_dir = os.path.join("results_final", subfolder)
os.makedirs(output_dir, exist_ok=True)

print(f"\nOutput folder: {output_dir}")

# ============================================================
# 5. FINAL DATASET PREPARATION
#    Refit scaler on train+val before final training so scaling
#    parameters reflect all non-test data, matching the final
#    training set size consistently with KNN and RF scripts.
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
x_test_final     = df_test_f[best_features].values

# ============================================================
# 6. FINAL TRAINING (train + val combined)
# ============================================================
print(f"\n=== Final Training ===")
print(f"Model      : {MODEL}")
print(f"Scaler     : {SCALER}")
print(f"Polynomial : degree={POLY_DEGREE}" if USE_POLYNOMIAL else "Polynomial : No")
print(f"Selection  : {best_method} ({len(best_features)} features)")
print(f"Train size : {x_trainval_final.shape[0]} (train+val)")
print("-" * 40)

model = models[MODEL]
model.fit(x_trainval_final, y_trainval)

if hasattr(model, "intercept_"):
    print(f"Intercept: {model.intercept_:.4f}")
print("-" * 40)

# ============================================================
# 7. EVALUATION (completely untouched test set)
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

    return {
        "model"                       : MODEL,
        "scaler"                      : SCALER,
        "polynomial"                  : USE_POLYNOMIAL,
        "poly_degree"                 : POLY_DEGREE if USE_POLYNOMIAL else None,
        "mrmr"                        : USE_MRMR,
        "wrapper_selection"           : USE_WRAPPER_SELECTION,
        "selection_method"            : best_method,
        "n_features"                  : len(best_features),
        "mse"                         : float(mse),
        "rmse"                        : float(rmse),
        "mae"                         : float(mae),
        "r2"                          : float(r2),
        "mape"                        : float(mape),
        "negative_predictions_count"  : int(n_neg),
        "negative_predictions_pct"    : float(100 * n_neg / len(y_pred)),
    }


metrics = evaluate_model(y_test, y_pred)

# ============================================================
# 8. SAVE METRICS
# ============================================================
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to : {metrics_path}")

# ============================================================
# 9. PLOT
# ============================================================
def plot_predictions(y_true, y_pred, output_dir):
    comparison_df = pd.DataFrame({
        "Real Value"     : y_true.values,
        "Predicted"      : y_pred,
        "Absolute Error" : abs(y_true.values - y_pred),
    })
    print(f"\n--- First 15 samples from the test set ---")
    print(comparison_df.head(15))

    plt.clf()
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
    method_tag = best_method.lower()
    title      = f"Real vs. Predicted — {MODEL} ({SCALER}{poly_tag}, {method_tag})"

    plt.title(title)
    plt.xlabel("Real Value (K)")
    plt.ylabel("Predicted Value (K)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    plot_path = os.path.join(output_dir, "predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to    : {plot_path}")

pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
plot_predictions(y_test, y_pred, output_dir)