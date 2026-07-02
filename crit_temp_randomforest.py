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
from itertools import product as iterproduct

# ============================================================
# Settings / Pipeline Configuration Toggles
#
# Recommended configurations for RF:
#
#   Academically correct (RF as designed):
#     USE_POLYNOMIAL = False
#     USE_SCALER     = False   — RF is scale-invariant
#     USE_MRMR       = True    — reduces 81 → top-K raw features
#     MRMR_K         = 40      — RF handles redundancy internally,
#                                so a smaller K than linear is fine
#
#   Controlled comparison (same pipeline as Linear/KNN):
#     USE_POLYNOMIAL = True
#     USE_SCALER     = True
#     USE_MRMR       = True    — uses polynomial MRMR features
#     MRMR_K         = 775
#
# Why no wrapper selection for RF:
#   Forward/backward selection evaluates one feature at a time.
#   RF already handles feature relevance and redundancy internally
#   through max_features (random subsets at each split) and the
#   ensemble averaging. Adding a wrapper on top introduces a
#   linear-model assumption about feature independence that
#   conflicts with how RF works. The hyperparameter search
#   (including max_features) IS the RF equivalent of feature
#   selection tuning.
#
# Invalid: USE_MRMR=True + USE_POLYNOMIAL=False requires
#   best_mrmr_features_raw.json (MRMR ranked on original 81
#   features). Run your MRMR script with USE_POLYNOMIAL=False
#   and save to that path before using this combination.
# ============================================================
MODEL = "RandomForest"

USE_POLYNOMIAL = True   # Recommended False for RF
POLY_DEGREE    = 2

USE_SCALER     = True   # Recommended False — RF is scale-invariant
SCALER_TYPE    = "Standard"

USE_MRMR       = True
MRMR_K         = 100    # When USE_POLYNOMIAL=False: top-K raw features
                         # When USE_POLYNOMIAL=True : top-K poly features (e.g. 775)

N_ITER_RANDOM  = 25
N_TOP_KEEP     = 5
MAX_GRID_SIZE  = 500

# ============================================================
# GUARD
# ============================================================
if USE_MRMR and not USE_POLYNOMIAL:
    raw_mrmr_path = "Project/results/best_mrmr_features_raw.json"
    if not os.path.exists(raw_mrmr_path):
        raise FileNotFoundError(
            f"USE_MRMR=True with USE_POLYNOMIAL=False requires "
            f"'{raw_mrmr_path}' — MRMR ranked on the original 81 features.\n"
            "Run your MRMR script without polynomial expansion and save the "
            "result to that path first."
        )

# ============================================================
# PATHS
# ============================================================
poly_suffix        = "_poly" if USE_POLYNOMIAL else "_raw"
mrmr_file_path     = ("Project/results/best_mrmr_features.json"
                      if USE_POLYNOMIAL else
                      "Project/results/best_mrmr_features_raw.json")
random_cache_path  = f"Project/results/rf_random_search{poly_suffix}.json"
grid_cache_path    = f"Project/results/rf_grid_search{poly_suffix}.json"

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
# 1. POLYNOMIAL FEATURE EXPANSION (optional)
# ============================================================
if USE_POLYNOMIAL:
    poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        include_bias=False,
        interaction_only=False
    )
    x_train_expanded = poly.fit_transform(x_train)
    x_val_expanded   = poly.transform(x_val)
    x_test_expanded  = poly.transform(x_test)
    feature_names    = poly.get_feature_names_out(x_train.columns)

    print(f"[PolynomialFeatures degree={POLY_DEGREE}]")
    print(f"  Original features        : {x_train.shape[1]}")
    print(f"  Features after expansion : {x_train_expanded.shape[1]}")
    print("=" * 50)
else:
    x_train_expanded = x_train.values
    x_val_expanded   = x_val.values
    x_test_expanded  = x_test.values
    feature_names    = x_train.columns.tolist()
    print(f"[No polynomial expansion — using {len(feature_names)} raw features]")
    print("=" * 50)

# ============================================================
# 2. SCALING (optional — RF is scale-invariant)
# ============================================================
if USE_SCALER:
    scaler           = sklearn.preprocessing.StandardScaler()
    x_train_ready    = scaler.fit_transform(x_train_expanded)
    x_val_ready      = scaler.transform(x_val_expanded)
    x_test_ready     = scaler.transform(x_test_expanded)
    print(f"[StandardScaler applied]")
else:
    x_train_ready    = x_train_expanded
    x_val_ready      = x_val_expanded
    x_test_ready     = x_test_expanded
    print(f"[No scaling — RF is scale-invariant]")

df_train = pd.DataFrame(x_train_ready, columns=feature_names)
df_val   = pd.DataFrame(x_val_ready,   columns=feature_names)
df_test  = pd.DataFrame(x_test_ready,  columns=feature_names)

# ============================================================
# 3. MRMR FEATURE FILTERING (optional)
# ============================================================
if USE_MRMR:
    print(f"Loading MRMR features from {mrmr_file_path}...")
    with open(mrmr_file_path, "r") as f:
        ranked_features = json.load(f)
    best_features = ranked_features[:MRMR_K]
    print(f"  Using top {len(best_features)} of {len(ranked_features)} MRMR features.")
else:
    best_features = list(feature_names)
    print(f"Using all {len(best_features)} features (no MRMR filtering).")

print("=" * 50)

x_train_final = df_train[best_features].values
x_val_final   = df_val[best_features].values
x_test_final  = df_test[best_features].values

# ============================================================
# UNIQUE OUTPUT FOLDER
#
# Naming pattern:
#   rf[_poly2|_raw][_scaler|_noscaler][_mrmrK|_allfeatures]
# Examples:
#   rf_raw_noscaler_mrmr40        — recommended RF pipeline
#   rf_poly2_scaler_mrmr775       — controlled comparison pipeline
#   rf_raw_noscaler_allfeatures   — RF with no feature filtering
# ============================================================
parts = ["rf"]
parts.append(f"poly{POLY_DEGREE}" if USE_POLYNOMIAL else "raw")
parts.append("scaler" if USE_SCALER else "noscaler")
parts.append(f"mrmr{MRMR_K}" if USE_MRMR else "allfeatures")

subfolder  = "_".join(parts)
output_dir = os.path.join("results_final", subfolder)
os.makedirs(output_dir, exist_ok=True)
print(f"Output folder: {output_dir}")

# ============================================================
# FULL HYPERPARAMETER SPACE
#
# Ensemble
#   n_estimators          — number of trees
#
# Splitting criterion
#   criterion             — squared_error : standard MSE
#                           absolute_error: MAE, robust to outliers
#                           friedman_mse  : often best in practice
#
# Tree structure
#   max_depth             — None = fully grown
#   min_samples_split     — min samples to split a node
#   min_samples_leaf      — min samples at a leaf
#   min_weight_fraction_leaf — same as above but as fraction
#   max_leaf_nodes        — hard cap on leaves per tree
#   min_impurity_decrease — only split if gain >= this value
#
# Feature sampling per split (RF's internal feature selection)
#   max_features          — 'sqrt' : classic RF default
#                           'log2' : more aggressive reduction
#                           float  : explicit fraction
#                           None   : all features (becomes bagging)
#
# Cost-complexity pruning
#   ccp_alpha             — 0.0 = no pruning; higher = simpler trees
#
# Bootstrap / row sampling
#   bootstrap             — True : sample rows with replacement
#                           False: each tree sees all rows
#   max_samples           — fraction of rows per tree
#                           (only active when bootstrap=True)
# ============================================================
param_distributions = {
    "n_estimators"            : [50, 100, 200, 300, 500],
    "criterion"               : ["squared_error", "absolute_error", "friedman_mse"],
    "max_depth"               : [None, 5, 10, 20, 30, 50],
    "min_samples_split"       : [2, 5, 10, 20],
    "min_samples_leaf"        : [1, 2, 4, 8],
    "min_weight_fraction_leaf": [0.0, 0.001, 0.01],
    "max_leaf_nodes"          : [None, 50, 100, 200],
    "min_impurity_decrease"   : [0.0, 0.001, 0.01, 0.1],
    "max_features"            : ["sqrt", "log2", 0.2, 0.4, 0.6, None],
    "ccp_alpha"               : [0.0, 0.001, 0.01, 0.05],
    "bootstrap"               : [True, False],
    "max_samples"             : [None, 0.6, 0.7, 0.8, 0.9],
}

# ============================================================
# HELPERS
# ============================================================
def fix_params(params):
    """max_samples is only valid when bootstrap=True."""
    p = dict(params)
    if not p["bootstrap"]:
        p["max_samples"] = None
    return p


def eval_config(params, x_tr, y_tr, x_v, y_v):
    rf = RandomForestRegressor(**params, random_state=78, n_jobs=-1)
    rf.fit(x_tr, y_tr)
    return np.sqrt(mean_squared_error(y_v, rf.predict(x_v)))


def params_key(params):
    return tuple(sorted((k, str(v)) for k, v in params.items()))


# ============================================================
# PHASE 1 — RANDOM SEARCH
# Cache: rf_random_search{poly_suffix}.json — delete to re-run.
# ============================================================
if os.path.exists(random_cache_path):
    print(f"Loading random search results from {random_cache_path}...")
    with open(random_cache_path, "r") as f:
        random_cache = json.load(f)
    top_configs = random_cache["top_configs"]
    print(f"  Top {len(top_configs)} configs loaded.")
    print(f"  Best val RMSE : {top_configs[0]['val_rmse']:.4f}")
    print(f"  Worst of top  : {top_configs[-1]['val_rmse']:.4f}")
    print("Skipping random search.")
    print("=" * 50)

else:
    print(f"[Phase 1: Random Search — {N_ITER_RANDOM} configurations]")
    print("=" * 50)

    random.seed(78)
    all_results  = []
    current_best = float("inf")

    for i in range(N_ITER_RANDOM):
        params   = fix_params({k: random.choice(v) for k, v in param_distributions.items()})
        val_rmse = eval_config(params, x_train_final, y_train, x_val_final, y_val)
        all_results.append({"params": params, "val_rmse": val_rmse})

        tag = "  <=== best so far" if val_rmse < current_best else ""
        if val_rmse < current_best:
            current_best = val_rmse

        print(f"  [{i+1:2d}/{N_ITER_RANDOM}]  Val RMSE: {val_rmse:.4f}{tag}")
        print(f"           {params}")

    all_results.sort(key=lambda r: r["val_rmse"])
    top_configs = all_results[:N_TOP_KEEP]

    print(f"\n=> Top {N_TOP_KEEP} by Val RMSE:")
    for rank, cfg in enumerate(top_configs, 1):
        print(f"  [{rank:2d}]  {cfg['val_rmse']:.4f}  |  {cfg['params']}")

    os.makedirs("Project/results", exist_ok=True)
    with open(random_cache_path, "w") as f:
        json.dump({
            "n_iter"     : N_ITER_RANDOM,
            "all_results": all_results,
            "top_configs": top_configs,
        }, f, indent=2)
    print(f"\nPhase 1 results saved to {random_cache_path}")
    print("=" * 50)

# ============================================================
# PHASE 2 — FOCUSED GRID SEARCH
# Builds a refined grid from unique values across the top
# N_TOP_KEEP configs, then searches exhaustively (or samples
# if the grid exceeds MAX_GRID_SIZE).
# Cache: rf_grid_search{poly_suffix}.json — delete to re-run.
# ============================================================
if os.path.exists(grid_cache_path):
    print(f"Loading grid search results from {grid_cache_path}...")
    with open(grid_cache_path, "r") as f:
        grid_cache = json.load(f)
    best_params   = grid_cache["params"]
    best_val_rmse = grid_cache["val_rmse"]
    search_type   = grid_cache.get("search_type", "unknown")
    grid_size     = grid_cache.get("grid_size", "?")
    print(f"  Best val RMSE : {best_val_rmse:.4f}")
    print(f"  Search type   : {search_type} (grid size: {grid_size})")
    print("Skipping grid search.")
    print("=" * 50)

else:
    print(f"[Phase 2: Focused Grid Search from top {N_TOP_KEEP} configurations]")

    top_params_list = [cfg["params"] for cfg in top_configs]
    grid = {}
    for param in param_distributions.keys():
        seen = []
        for p in top_params_list:
            v = p[param]
            if v not in seen:
                seen.append(v)
        grid[param] = seen

    print("\n  Refined grid (unique values from top configs):")
    raw_size = 1
    for param, values in grid.items():
        print(f"    {param:30s} {len(values)} value(s): {values}")
        raw_size *= len(values)
    print(f"\n  Raw grid size before constraints: {raw_size}")

    keys           = list(grid.keys())
    all_combos     = [fix_params(dict(zip(keys, combo)))
                      for combo in iterproduct(*grid.values())]

    seen_keys      = set()
    unique_configs = []
    for cfg in all_combos:
        k = params_key(cfg)
        if k not in seen_keys:
            seen_keys.add(k)
            unique_configs.append(cfg)

    actual_size = len(unique_configs)
    print(f"  Unique configs after deduplication: {actual_size}")

    if actual_size > MAX_GRID_SIZE:
        random.seed(78)
        random.shuffle(unique_configs)
        unique_configs = unique_configs[:MAX_GRID_SIZE]
        search_type    = "sampled_grid"
        print(f"  Exceeds MAX_GRID_SIZE={MAX_GRID_SIZE}. Sampling {MAX_GRID_SIZE}.")
    else:
        search_type = "full_grid"
        print(f"  Full exhaustive grid search.")

    print(f"\n  Evaluating {len(unique_configs)} configurations...")
    print("=" * 50)

    best_params   = None
    best_val_rmse = float("inf")
    grid_results  = []

    for i, params in enumerate(unique_configs):
        val_rmse = eval_config(params, x_train_final, y_train, x_val_final, y_val)
        grid_results.append({"params": params, "val_rmse": val_rmse})

        tag = "  <=== best so far" if val_rmse < best_val_rmse else ""
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params   = params

        print(f"  [{i+1:3d}/{len(unique_configs)}]  Val RMSE: {val_rmse:.4f}{tag}")

    grid_results.sort(key=lambda r: r["val_rmse"])

    print(f"\n=> Best configuration (Val RMSE: {best_val_rmse:.4f}):")
    print(json.dumps(best_params, indent=4))
    print("=" * 50)

    os.makedirs("Project/results", exist_ok=True)
    with open(grid_cache_path, "w") as f:
        json.dump({
            "params"     : best_params,
            "val_rmse"   : best_val_rmse,
            "grid_size"  : actual_size,
            "search_type": search_type,
            "top_results": grid_results[:10],
        }, f, indent=2)
    print(f"Phase 2 results saved to {grid_cache_path}")

# ============================================================
# FINAL TRAINING — refit on train+val combined
# If scaler was used, refit it on train+val before transforming.
# ============================================================
x_trainval_expanded = np.vstack([x_train_expanded, x_val_expanded])
y_trainval          = pd.concat([y_train, y_val]).reset_index(drop=True)

if USE_SCALER:
    final_scaler          = sklearn.preprocessing.StandardScaler()
    x_trainval_ready_f    = final_scaler.fit_transform(x_trainval_expanded)
    x_test_ready_f        = final_scaler.transform(x_test_expanded)
else:
    x_trainval_ready_f    = x_trainval_expanded
    x_test_ready_f        = x_test_expanded

df_trainval_f = pd.DataFrame(x_trainval_ready_f, columns=feature_names)
df_test_f     = pd.DataFrame(x_test_ready_f,     columns=feature_names)

x_trainval_final = df_trainval_f[best_features].values
x_test_final_f   = df_test_f[best_features].values

print(f"\n=== Final Training ===")
print(f"Model      : {MODEL}")
print(f"Polynomial : {'degree=' + str(POLY_DEGREE) if USE_POLYNOMIAL else 'No (raw features)'}")
print(f"Scaler     : {SCALER_TYPE if USE_SCALER else 'None (scale-invariant)'}")
print(f"MRMR       : {'top-' + str(MRMR_K) + ' features' if USE_MRMR else 'all features'}")
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

    return {
        "model"                      : MODEL,
        "polynomial"                 : USE_POLYNOMIAL,
        "poly_degree"                : POLY_DEGREE if USE_POLYNOMIAL else None,
        "scaler"                     : SCALER_TYPE if USE_SCALER else None,
        "mrmr"                       : USE_MRMR,
        "mrmr_k"                     : MRMR_K if USE_MRMR else None,
        "n_features"                 : len(best_features),
        "best_params"                : best_params,
        "mse"                        : float(mse),
        "rmse"                       : float(rmse),
        "mae"                        : float(mae),
        "r2"                         : float(r2),
        "mape"                       : float(mape),
        "negative_predictions_count" : int(n_neg),
        "negative_predictions_pct"   : float(100 * n_neg / len(y_pred)),
    }


print(f"=== Final Test Metrics ({MODEL}) ===")
metrics = evaluate_model(y_test, y_pred)

# ============================================================
# SAVE METRICS
# ============================================================
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to : {metrics_path}")

# ============================================================
# PLOT
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
    plt.scatter(y_true, y_pred, alpha=0.5, color="forestgreen", edgecolor="k")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=2,
        label="Perfect Prediction (y=x)"
    )

    poly_tag   = f"_poly{POLY_DEGREE}" if USE_POLYNOMIAL else "_raw"
    scaler_tag = f"_{SCALER_TYPE.lower()}" if USE_SCALER else "_noscaler"
    mrmr_tag   = f"_mrmr{MRMR_K}" if USE_MRMR else "_allfeatures"
    title      = f"Real vs. Predicted — {MODEL}{poly_tag}{scaler_tag}{mrmr_tag}"

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