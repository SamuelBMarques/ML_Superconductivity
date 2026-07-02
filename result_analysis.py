"""
analysis.py — Cross-model error analysis and presentation plots
===============================================================
Run AFTER all three model scripts have been executed with their
final settings. Each model script must have saved a predictions.csv
in its output folder (see the snippet below to add to each script).

Snippet to add at the end of each model script, before plot_predictions():
---------------------------------------------------------------------------
    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
---------------------------------------------------------------------------

Then set the three paths below to the matching output folders.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ============================================================
# CONFIGURATION — point these to your final experiment folders
# ============================================================
RESULTS = {
    "Linear"       : "results_final/linear_none",
    "KNN"          : "results_final/knn_k5_none",
    "RandomForest" : "results_final/rf_raw_noscaler_allfeatures",
}

MODEL_COLORS = {
    "Linear"       : "#4C72B0",   # blue
    "KNN"          : "#2CA02C",   # teal-green
    "RandomForest" : "#D62728",   # red
}

# Temperature bins for range analysis (in Kelvin)
TEMP_BINS   = [0, 10, 20, 40, 60, 80, 100, 200]
TEMP_LABELS = ["0–10", "10–20", "20–40", "40–60", "60–80", "80–100", "100+"]

OUTPUT_DIR = "results_final/analysis_rawmodels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
data    = {}
metrics = {}

for name, folder in RESULTS.items():
    pred_path    = os.path.join(folder, "predictions.csv")
    metrics_path = os.path.join(folder, "metrics.json")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Missing {pred_path}.\n"
            "Add this snippet before plot_predictions() in each model script:\n\n"
            "    pred_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})\n"
            "    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)\n"
        )

    df = pd.read_csv(pred_path)
    df["error"]    = df["y_pred"] - df["y_true"]     # signed: + = overestimate
    df["abs_error"] = df["error"].abs()
    df["temp_bin"] = pd.cut(df["y_true"], bins=TEMP_BINS, labels=TEMP_LABELS, right=False)
    data[name] = df

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics[name] = json.load(f)

print("Loaded predictions for:", list(data.keys()))

# ============================================================
# HELPER
# ============================================================
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ============================================================
# PLOT 1 — MODEL COMPARISON TABLE (summary metrics)
# ============================================================
rows = []
for name, m in metrics.items():
    rows.append({
        "Model"     : name,
        "RMSE (K)"  : round(m["rmse"], 2),
        "MAE (K)"   : round(m["mae"], 2),
        "R²"        : round(m["r2"], 4),
        "Features"  : m.get("n_features", "—"),
    })

summary_df = pd.DataFrame(rows).set_index("Model")
print("\n=== Summary ===")
print(summary_df.to_string())

fig, ax = plt.subplots(figsize=(8, 1.8))
ax.axis("off")
tbl = ax.table(
    cellText  = summary_df.reset_index().values,
    colLabels = ["Model"] + list(summary_df.columns),
    cellLoc   = "center",
    loc       = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F2F3F4")
plt.tight_layout()
save(fig, "01_summary_table.png")

# ============================================================
# PLOT 2 — RMSE PER TEMPERATURE RANGE (main requested analysis)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bin_stats = {}
for name, df in data.items():
    stats = df.groupby("temp_bin", observed=False).apply(
        lambda g: pd.Series({
            "rmse"  : np.sqrt((g["error"] ** 2).mean()),
            "mae"   : g["abs_error"].mean(),
            "count" : len(g),
        })
    )
    bin_stats[name] = stats

x      = np.arange(len(TEMP_LABELS))
width  = 0.25
names  = list(data.keys())

for i, name in enumerate(names):
    rmse_vals = [bin_stats[name].loc[lbl, "rmse"] if lbl in bin_stats[name].index else 0
                 for lbl in TEMP_LABELS]
    axes[0].bar(x + i * width, rmse_vals, width, label=name,
                color=MODEL_COLORS[name], alpha=0.85, edgecolor="white")

axes[0].set_title("RMSE by Temperature Range", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Temperature Range (K)")
axes[0].set_ylabel("RMSE (K)")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(TEMP_LABELS, rotation=20)
axes[0].legend()
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

# Sample count per bin (shared x-axis to show data density)
counts = {name: [bin_stats[name].loc[lbl, "count"] if lbl in bin_stats[name].index else 0
                 for lbl in TEMP_LABELS]
          for name in names}

axes[1].bar(x, counts[names[0]], color="#AAB7B8", edgecolor="white", label="Sample count")
axes[1].set_title("Test Samples per Temperature Range", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Temperature Range (K)")
axes[1].set_ylabel("Number of samples")
axes[1].set_xticks(x)
axes[1].set_xticklabels(TEMP_LABELS, rotation=20)
axes[1].grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
save(fig, "02_rmse_by_temperature_range.png")

# ============================================================
# PLOT 3 — SIGNED ERROR BY TEMPERATURE RANGE
#          (are models systematically over/under-estimating?)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

for i, name in enumerate(names):
    mean_errors = [bin_stats[name].loc[lbl, "mae"]
                   if lbl in bin_stats[name].index else 0
                   for lbl in TEMP_LABELS]
    signed = [data[name][data[name]["temp_bin"] == lbl]["error"].mean()
              if lbl in data[name]["temp_bin"].values else 0
              for lbl in TEMP_LABELS]
    ax.plot(TEMP_LABELS, signed, marker="o", linewidth=2,
            color=MODEL_COLORS[name], label=name)

ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.fill_between(range(len(TEMP_LABELS)), -5, 5, alpha=0.05, color="green",
                label="±5K band")
ax.set_title("Mean Signed Error by Temperature Range\n"
             "(positive = overestimate, negative = underestimate)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Temperature Range (K)")
ax.set_ylabel("Mean Error (K)")
ax.set_xticks(range(len(TEMP_LABELS)))
ax.set_xticklabels(TEMP_LABELS, rotation=20)
ax.legend()
ax.grid(linestyle=":", alpha=0.6)
plt.tight_layout()
save(fig, "03_signed_error_by_range.png")

# ============================================================
# PLOT 4 — ERROR DISTRIBUTION (histogram per model)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (name, df) in zip(axes, data.items()):
    errors = df["error"].values
    ax.hist(errors, bins=60, color=MODEL_COLORS[name], alpha=0.8, edgecolor="white")
    ax.axvline(0,            color="black", linestyle="--", linewidth=1.5, label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="-",  linewidth=1.5,
               label=f"Mean = {errors.mean():.1f} K")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Error (Predicted − Real) (K)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)

fig.suptitle("Error Distribution per Model", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "04_error_distribution.png")

# ============================================================
# PLOT 5 — CUMULATIVE ERROR (what % of predictions are within X K?)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

thresholds = np.linspace(0, 50, 300)
for name, df in data.items():
    abs_err = df["abs_error"].values
    cdf     = [(abs_err <= t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, cdf, color=MODEL_COLORS[name], linewidth=2, label=name)

for pct in [50, 75, 90]:
    ax.axhline(pct, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(51, pct + 0.5, f"{pct}%", fontsize=8, color="gray")

ax.set_title("Cumulative Distribution of Absolute Errors\n"
             "(read as: X% of predictions are within Y K of the true value)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Absolute Error Threshold (K)")
ax.set_ylabel("% of test samples within threshold")
ax.set_xlim(0, 50)
ax.set_ylim(0, 101)
ax.legend()
ax.grid(linestyle=":", alpha=0.6)
plt.tight_layout()
save(fig, "05_cumulative_error.png")

# ============================================================
# PLOT 6 — RESIDUAL PLOT (predicted vs residual)
#          Should look like random noise — patterns = problems
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (name, df) in zip(axes, data.items()):
    ax.scatter(df["y_pred"], df["error"], alpha=0.25, s=8,
               color=MODEL_COLORS[name], edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Tc (K)")
    ax.set_ylabel("Residual (K)")
    ax.grid(linestyle=":", alpha=0.5)

fig.suptitle("Residual Plots — Predicted vs. Error\n"
             "(random scatter = good; patterns = systematic bias)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "06_residual_plots.png")

# ============================================================
# PLOT 7 — SCATTER colored by absolute error
#          Makes it visually clear WHERE errors are large
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vmax = max(df["abs_error"].quantile(0.95) for df in data.values())
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.YlOrRd

for ax, (name, df) in zip(axes, data.items()):
    sc = ax.scatter(df["y_true"], df["y_pred"], c=df["abs_error"],
                    cmap=cmap, norm=norm, alpha=0.6, s=10, edgecolors="none")
    lims = [min(df["y_true"].min(), df["y_pred"].min()),
            max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Real Tc (K)")
    ax.set_ylabel("Predicted Tc (K)")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
             label="Absolute Error (K)", shrink=0.8)
fig.suptitle("Real vs. Predicted — colored by Absolute Error",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "07_scatter_colored_by_error.png")

# ============================================================
# PLOT 8 — WORST PREDICTIONS ANALYSIS
#          Where do the 5% worst predictions come from?
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, df) in zip(axes, data.items()):
    threshold = df["abs_error"].quantile(0.95)
    worst     = df[df["abs_error"] >= threshold]
    best      = df[df["abs_error"] <  threshold]

    ax.hist(best["y_true"],  bins=30, alpha=0.6, color="steelblue",
            label=f"Normal (n={len(best)})")
    ax.hist(worst["y_true"], bins=30, alpha=0.8, color="crimson",
            label=f"Worst 5% (n={len(worst)}, ≥{threshold:.1f} K error)")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Real Tc (K)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)

fig.suptitle("Temperature Distribution of Worst 5% Predictions",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "08_worst_predictions_distribution.png")

# ============================================================
# PRINT SUMMARY TEXT TABLE — per-range RMSE for all models
# ============================================================
print("\n=== RMSE by Temperature Range ===")
header = f"{'Range':>10}" + "".join(f"{n:>16}" for n in names) + f"{'Samples':>10}"
print(header)
print("-" * len(header))
for lbl in TEMP_LABELS:
    row = f"{lbl:>10}"
    for name in names:
        try:
            val = bin_stats[name].loc[lbl, "rmse"]
            row += f"{val:>16.2f}"
        except KeyError:
            row += f"{'—':>16}"
    try:
        cnt = bin_stats[names[0]].loc[lbl, "count"]
        row += f"{int(cnt):>10}"
    except KeyError:
        row += f"{'0':>10}"
    print(row)

print(f"\nAll analysis plots saved to: {OUTPUT_DIR}/")