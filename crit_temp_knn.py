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
# Settings / Pipeline Configuration Toggles
#
# Valid combinations:
#   Baseline (no engineering)  : USE_POLYNOMIAL=False, USE_MRMR=False, USE_WRAPPER_SELECTION=False
#   MRMR only                  : USE_POLYNOMIAL=True,  USE_MRMR=True,  USE_WRAPPER_SELECTION=False
#   MRMR + Wrapper (KNN-based) : USE_POLYNOMIAL=True,  USE_MRMR=True,  USE_WRAPPER_SELECTION=True
#   Wrapper on raw features    : USE_POLYNOMIAL=False, USE_MRMR=False, USE_WRAPPER_SELECTION=True
#
# Feature selection flow when USE_WRAPPER_SELECTION=True:
#   Step 1 — Tune k on the pre-selected feature set (from MRMR
#            or linear regression JSON, depending on
#            USE_LINEAR_SELECTION). This avoids re-tuning k
#            inside every wrapper step, which would be O(n²·fits).
#   Step 2 — Run forward/backward selection using KNN with that
#            fixed k as the internal evaluation model. This is
#            correct per the professor's guidance: "quality
#            criteria computed with an actual classifier."
#
# USE_LINEAR_SELECTION: when True, the candidate pool for wrapper
#   selection is loaded from the linear regression forward/backward
#   result (best_selection_features_poly.json). When False, the
#   top SELECTION_POOL features from MRMR are used directly.
#   This lets you compare: does KNN-based selection on the linear
#   regression candidates differ from KNN-based selection on raw
#   MRMR candidates?
#
# Invalid: USE_MRMR=True with USE_POLYNOMIAL=False
# ============================================================
MODEL  = "KNN"
SCALER = "Standard"

USE_POLYNOMIAL        = True
POLY_DEGREE           = 2

USE_MRMR              = True
BEST_K_FEATURES       = 775    # MRMR features used when USE_WRAPPER_SELECTION=False
SELECTION_POOL        = 100    # Candidate pool size for wrapper selection

USE_WRAPPER_SELECTION = True  # Toggle KNN-based forward/backward selection
USE_LINEAR_SELECTION  = False   # Use linear regression selection as candidate pool
                                # (ignored when USE_WRAPPER_SELECTION=False)

NEIGHBORS_CANDIDATES  = [1, 3, 4, 5, 6, 7, 9, 11, 13, 15, 20, 25, 30, 40, 50]

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
# PATHS
# ============================================================
mrmr_file_path          = "Project/results/best_mrmr_features.json"
linear_selection_path   = "Project/results/best_selection_features_poly.json"
poly_suffix             = "_poly" if USE_POLYNOMIAL else ""
knn_k_cache_path        = f"Project/results/knn_best_k{poly_suffix}.json"
knn_selection_cache_path = f"Project/results/knn_best_selection{poly_suffix}.json"

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
# 2. SCALING — fit only on train
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
# 3. DETERMINE INITIAL FEATURE SET FOR K-TUNING
#    Load whatever pre-selected features will be used as the
#    starting point for tuning k. This set is also used as
#    the candidate pool if USE_LINEAR_SELECTION=True.
# ============================================================
if USE_MRMR:
    print(f"Loading MRMR features from {mrmr_file_path}...")
    with open(mrmr_file_path, "r") as f:
        ranked_features = json.load(f)
    print(f"Loaded {len(ranked_features)} MRMR features.")

if USE_WRAPPER_SELECTION and USE_LINEAR_SELECTION:
    # Use the features the linear regression script already found
    if not os.path.exists(linear_selection_path):
        raise FileNotFoundError(
            f"Could not find '{linear_selection_path}'. "
            "Run crit_temp_linearregression.py with USE_WRAPPER_SELECTION=True first."
        )
    print(f"Loading linear regression selection from {linear_selection_path}...")
    with open(linear_selection_path, "r") as f:
        lin_saved = json.load(f)
    k_tuning_features  = lin_saved["features"]
    candidate_features = lin_saved["features"]   # pool for KNN wrapper selection
    print(f"  {len(k_tuning_features)} features loaded "
          f"(linear {lin_saved['method']}, Val RMSE: {lin_saved['val_rmse']:.4f})")

elif USE_MRMR:
    k_tuning_features  = ranked_features[:BEST_K_FEATURES]
    candidate_features = ranked_features[:SELECTION_POOL]
    print(f"Using top {len(k_tuning_features)} MRMR features for k-tuning.")

else:
    k_tuning_features  = list(feature_names)
    candidate_features = list(feature_names)
    print("Using all features for k-tuning (no MRMR, no linear selection).")

print("=" * 50)

# ============================================================
# 4. TUNE K — evaluated on val set using k_tuning_features
#    This step always runs before wrapper selection so that
#    the wrapper uses a fixed, known-good k (otherwise each
#    wrapper step would need its own k search, which is
#    O(candidates² × len(NEIGHBORS_CANDIDATES)) fits).
# ============================================================
if os.path.exists(knn_k_cache_path):
    print(f"Loading cached k from {knn_k_cache_path}...")
    with open(knn_k_cache_path, "r") as f:
        k_cache = json.load(f)
    best_k        = k_cache["best_k"]
    best_k_rmse   = k_cache["val_rmse"]
    print(f"  best_k={best_k}  |  Val RMSE: {best_k_rmse:.4f}. Skipping k-tuning.")
    print("=" * 50)

else:
    print(f"[Tuning k on {len(k_tuning_features)} features]")
    x_tr_k = df_train_scaled[k_tuning_features].values
    x_va_k = df_val_scaled[k_tuning_features].values

    best_k      = None
    best_k_rmse = float("inf")

    for k in NEIGHBORS_CANDIDATES:
        knn      = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(x_tr_k, y_train)
        val_rmse = np.sqrt(mean_squared_error(y_val, knn.predict(x_va_k)))

        tag = "  <=== best" if val_rmse < best_k_rmse else ""
        print(f"  k={k:2d}  |  Val RMSE: {val_rmse:.4f}{tag}")

        if val_rmse < best_k_rmse:
            best_k_rmse = val_rmse
            best_k      = k

    print(f"\n  => best_k={best_k}  |  Val RMSE: {best_k_rmse:.4f}")
    print("=" * 50)

    os.makedirs("Project/results", exist_ok=True)
    with open(knn_k_cache_path, "w") as f:
        json.dump({"best_k": best_k, "val_rmse": best_k_rmse,
                   "n_features_used_for_tuning": len(k_tuning_features)}, f, indent=2)
    print(f"k saved to {knn_k_cache_path}")

# ============================================================
# HELPER FUNCTIONS — KNN-based wrapper selection
# ============================================================
def _knn_val_rmse(x_train_df, y_train, x_val_df, y_val, features, k):
    """Fit KNN(k) on features, return validation RMSE."""
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(x_train_df[features].values, y_train)
    return np.sqrt(mean_squared_error(y_val, knn.predict(x_val_df[features].values)))


def knn_forward_selection(x_train_df, x_val_df, y_train, y_val, candidates, k):
    selected  = []
    remaining = list(candidates)
    best_rmse = float("inf")

    print(f"\n{'='*55}")
    print(f"[KNN Forward Selection]  Pool: {len(candidates)} features  k={k}")
    print(f"{'='*55}")

    while remaining:
        step_best_feature = None
        step_best_rmse    = float("inf")

        for feature in remaining:
            rmse = _knn_val_rmse(x_train_df, y_train, x_val_df, y_val,
                                 selected + [feature], k)
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


def knn_backward_elimination(x_train_df, x_val_df, y_train, y_val, candidates, k):
    selected  = list(candidates)
    best_rmse = _knn_val_rmse(x_train_df, y_train, x_val_df, y_val, selected, k)

    print(f"\n{'='*55}")
    print(f"[KNN Backward Elimination]  Pool: {len(candidates)} features  k={k}")
    print(f"  Start  |  Val RMSE: {best_rmse:.4f}")
    print(f"{'='*55}")

    while len(selected) > 1:
        step_best_feature = None
        step_best_rmse    = float("inf")

        for feature in selected:
            temp = [f for f in selected if f != feature]
            rmse = _knn_val_rmse(x_train_df, y_train, x_val_df, y_val, temp, k)
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
# 5. WRAPPER SELECTION (KNN-based) or direct feature assignment
# ============================================================
if USE_WRAPPER_SELECTION:
    if os.path.exists(knn_selection_cache_path):
        print(f"Loading cached KNN selection from {knn_selection_cache_path}...")
        with open(knn_selection_cache_path, "r") as f:
            saved = json.load(f)
        best_features    = saved["features"]
        best_method      = saved["method"]
        best_sel_rmse    = saved["val_rmse"]
        print(f"  {len(best_features)} features "
              f"(method: {best_method}, Val RMSE: {best_sel_rmse:.4f}). Skipping selection.")
        print("=" * 55)

    else:
        print(f"Running KNN wrapper selection with k={best_k}...")
        features_fwd, rmse_fwd = knn_forward_selection(
            df_train_scaled, df_val_scaled, y_train, y_val, candidate_features, best_k
        )
        features_bwd, rmse_bwd = knn_backward_elimination(
            df_train_scaled, df_val_scaled, y_train, y_val, candidate_features, best_k
        )

        print(f"\n{'='*55}")
        print(f"[Comparison]")
        print(f"  Forward selection    : {len(features_fwd):3d} features  |  Val RMSE: {rmse_fwd:.4f}")
        print(f"  Backward elimination : {len(features_bwd):3d} features  |  Val RMSE: {rmse_bwd:.4f}")

        if rmse_fwd <= rmse_bwd:
            best_features = features_fwd
            best_method   = "Forward"
            best_sel_rmse = rmse_fwd
        else:
            best_features = features_bwd
            best_method   = "Backward"
            best_sel_rmse = rmse_bwd

        print(f"\n  => Winner: {best_method} "
              f"({len(best_features)} features  |  Val RMSE: {best_sel_rmse:.4f})")
        print(f"{'='*55}")

        os.makedirs("Project/results", exist_ok=True)
        with open(knn_selection_cache_path, "w") as f:
            json.dump({
                "method"  : best_method,
                "val_rmse": best_sel_rmse,
                "k"       : best_k,
                "features": best_features,
            }, f, indent=2)
        print(f"KNN selection saved to {knn_selection_cache_path}")

elif USE_MRMR:
    best_features = ranked_features[:BEST_K_FEATURES]
    best_method   = "MRMR"
else:
    best_features = list(feature_names)
    best_method   = "None"

# ============================================================
# 6. UNIQUE OUTPUT FOLDER — one per pipeline configuration
#
# Naming pattern:
#   knn_k{k}[_poly2][_mrmr775|_mrmr100][_linsel][_forward|_backward|_none]
# Examples:
#   knn_k5                                — baseline
#   knn_k5_poly2_mrmr775                  — poly + MRMR, no wrapper
#   knn_k5_poly2_linsel_forward           — KNN wrapper on linear selection pool
#   knn_k5_poly2_mrmr100_forward          — KNN wrapper on MRMR pool
# ============================================================
parts = [f"knn_k{best_k}"]
if USE_POLYNOMIAL:
    parts.append(f"poly{POLY_DEGREE}")
if USE_MRMR and not USE_WRAPPER_SELECTION:
    parts.append(f"mrmr{BEST_K_FEATURES}")
if USE_WRAPPER_SELECTION:
    if USE_LINEAR_SELECTION:
        parts.append("linsel")
    else:
        parts.append(f"mrmr{SELECTION_POOL}")
    parts.append(best_method.lower())
elif not USE_MRMR:
    parts.append("none")

subfolder  = "_".join(parts)
output_dir = os.path.join("results_final", subfolder)
os.makedirs(output_dir, exist_ok=True)

print(f"\nOutput folder: {output_dir}")

# ============================================================
# 7. FINAL TRAINING — refit scaler on train+val
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
print(f"Selection  : {best_method} ({len(best_features)} features)")
print(f"Neighbors  : k={best_k}")
print(f"Train size : {x_trainval_final.shape[0]} (train+val)")
print("-" * 40)

final_model = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
final_model.fit(x_trainval_final, y_trainval)

# ============================================================
# 8. EVALUATION (test set — never touched until here)
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

    return {
        "model"                      : MODEL,
        "scaler"                     : SCALER,
        "polynomial"                 : USE_POLYNOMIAL,
        "poly_degree"                : POLY_DEGREE if USE_POLYNOMIAL else None,
        "mrmr"                       : USE_MRMR,
        "wrapper_selection"          : USE_WRAPPER_SELECTION,
        "use_linear_selection"       : USE_LINEAR_SELECTION,
        "selection_method"           : best_method,
        "n_features"                 : len(best_features),
        "best_k"                     : best_k,
        "mse"                        : float(mse),
        "rmse"                       : float(rmse),
        "mae"                        : float(mae),
        "r2"                         : float(r2),
        "mape"                       : float(mape),
        "negative_predictions_count" : int(n_neg),
        "negative_predictions_pct"   : float(100 * n_neg / len(y_pred)),
    }


metrics = evaluate_model(y_test, y_pred)

# ============================================================
# 9. SAVE METRICS
# ============================================================
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to : {metrics_path}")

# ============================================================
# 10. PLOT
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
    plt.scatter(y_true, y_pred, alpha=0.5, color="teal", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=2,
        label="Perfect Prediction (y=x)"
    )

    poly_tag = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else ""
    title    = f"Real vs. Predicted — {MODEL} (k={best_k}, {SCALER}{poly_tag}, {best_method.lower()})"
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