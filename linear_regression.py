import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# Settings
# ============================================================
MODEL  = "Linear"
SCALER = "Standard"

# Feature Elimination
VARIANCE_THRESHOLD   = 0.01   # Removes features 
CORRELATION_THRESHOLD = 0.95  # Remove features that are highly correlated with each other (redundant)

# Feature Addition
USE_POLYNOMIAL = True   # Activates feature squaring (x², x₁·x₂)
POLY_DEGREE    = 2      # Degree (2 = quadratic)

# ============================================================
# MODEL AND SCALER
# ============================================================
models = {
    "Linear": LinearRegression(),
}

scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "Nenhum": None,
}

# ============================================================
# DATA LOADING
# ============================================================
train_file_path = "Project/superconductivty+data/train.csv"
labels_path     = "Project/superconductivty+data/unique_m.csv"

df       = pd.read_csv(train_file_path)
df_labels = pd.read_csv(labels_path)

X = df.drop(columns=["critical_temp"])
Y = df_labels["critical_temp"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=78
)

print(f"Features originais: {X.shape[1]}")
print("=" * 50)

# ============================================================
# ETAPA 1 — FEATURE ELIMINATION
# ============================================================

# --- VarianceThreshold: remove almost constant features ---
var_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
x_train_var  = var_selector.fit_transform(x_train)
x_test_var   = var_selector.transform(x_test)

features_mantidas_var = X.columns[var_selector.get_support()].tolist()
n_removidas_var = X.shape[1] - len(features_mantidas_var)
print(f"[VarianceThreshold < {VARIANCE_THRESHOLD}]")
print(f"  Removidas: {n_removidas_var}  |  Restantes: {len(features_mantidas_var)}")

# --- Correlation Filter: remove redundant features---
x_train_df = pd.DataFrame(x_train_var, columns=features_mantidas_var)

corr_matrix = x_train_df.corr().abs()
upper_tri   = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

features_to_remove = [
    col for col in upper_tri.columns
    if any(upper_tri[col] > CORRELATION_THRESHOLD)
]
features_selecionadas = [f for f in features_mantidas_var if f not in features_to_remove]

x_train_df_sel = x_train_df[features_selecionadas]
x_test_df_sel  = pd.DataFrame(x_test_var, columns=features_mantidas_var)[features_selecionadas]

print(f"\n[Correlation Filter > {CORRELATION_THRESHOLD}]")
print(f"  Removidas: {len(features_to_remove)}  |  Restantes: {len(features_selecionadas)}")
if features_to_remove:
    print(f"  Features removidas: {features_to_remove}")

x_train_sel = x_train_df_sel.values
x_test_sel  = x_test_df_sel.values

print(f"\nFeatures após eliminação: {len(features_selecionadas)}")
print("=" * 50)

# ============================================================
# PreProcessing: Scaling
# Sclaing before polynomial features is crucial to prevent the new squared/interacted features from having vastly different magnitudes,
# which can help the model converge better and ensure that the optimization process is not dominated by features with larger scales.
# ============================================================
scaler = scalers[SCALER]

if scaler is not None:
    x_train_scaled = scaler.fit_transform(x_train_sel)
    x_test_scaled  = scaler.transform(x_test_sel)
else:
    x_train_scaled = x_train_sel
    x_test_scaled  = x_test_sel

# ============================================================
# FEATURE ADDITION (POLYNOMIAL / SQUARING)
# degree=2 adds: x₁², x₂², ..., xₙ² (quadráticas)
#                  + x₁·x₂, x₁·x₃, ... (interações)
# lets the model capture non-linear relationships without needing a non-linear model
# ============================================================
if USE_POLYNOMIAL:
    poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        include_bias=False,      
        interaction_only=False    
    )
    x_train_final = poly.fit_transform(x_train_scaled)
    x_test_final  = poly.transform(x_test_scaled)

    n_features_orig = len(features_selecionadas)
    n_features_poly = x_train_final.shape[1]
    n_novas = n_features_poly - n_features_orig
    print(f"[PolynomialFeatures degree={POLY_DEGREE}]")
    print(f"  Original features: {n_features_orig}")
    print(f"  New features (x², x₁x₂, ...): {n_novas}")
    print(f"  Total after squaring: {n_features_poly}")
    print("=" * 50)
else:
    x_train_final = x_train_scaled
    x_test_final  = x_test_scaled

# ============================================================
# TRAINING
# ============================================================
print(f"=== Settings ===")
print(f"Model: {MODEL}")
print(f"Scaler: {SCALER}")
print(f"Polynomial: {'Yes (degree=' + str(POLY_DEGREE) + ')' if USE_POLYNOMIAL else 'NNo'}")
print("-" * 40)

model = models[MODEL]
model.fit(x_train_final, y_train)

if hasattr(model, 'intercept_'):
    print(f"Intercept: {model.intercept_:.4f}")

print("-" * 40)

# ============================================================
# AVALIAÇÃO
# ============================================================
y_pred = model.predict(x_test_final)

def evaluate_model(y_true, y_pred):
    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    mape   = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"MSE      : {mse:.4f} K²")
    print(f"RMSE     : {rmse:.4f} K ")
    print(f"MAE      : {mae:.4f} K ")
    print(f"Median AE: {med_ae:.4f} K")
    print(f"R²       : {r2:.4f}")
    print(f"MAPE     : {mape:.4f}")
 
    n_neg = (y_pred < 0).sum()
    print(f"Negative preds (physically impossible): {n_neg} ({100 * n_neg / len(y_pred):.1f}%)")
 
    return {"rmse": rmse, "mae": mae, "median_ae": med_ae, "r2": r2, "mape": mape}
 
metrics = evaluate_model(y_test, y_pred)

# ============================================================
# GRÁFICO E ANÁLISE
# ============================================================
def compare_predictions(y_true, y_pred, num_samples=15):
    comparison_df = pd.DataFrame({
        "Real Value":    y_true.values,
        "Predicted": y_pred,
        "Absolute Error": abs(y_true.values - y_pred),
    })

    print(f"\n--- First {num_samples} samples from the test set ---")
    print(comparison_df.head(num_samples))

    n_negatives = (y_pred < 0).sum()
    print(f"\nNegative predictions (physically impossible): {n_negatives} ({100*n_negatives/len(y_pred):.1f}%)")

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
    plt.xlabel("Real Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    os.makedirs("Project/results", exist_ok=True)
    name = f"Project/results/grafico_{MODEL.lower()}_{SCALER.lower()}{poly_tag}.png"
    plt.savefig(name, dpi=300, bbox_inches="tight")
    print(f"\nGraphic saved to: {name}")

compare_predictions(y_test, y_pred)