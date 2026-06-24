import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from mrmr import mrmr_regression

# ============================================================
# Settings
# ============================================================
MODEL  = "Linear"
SCALER = "Standard"

# Feature Addition
USE_POLYNOMIAL = True   # Activates feature squaring (x², x₁·x₂)
POLY_DEGREE    = 2      # Degree (2 = quadratic)

# MRMR Settings (Expanded Search Space)
MAX_MRMR_FEATURES = 825 # Maximum number of features to rank
K_CANDIDATES = [ 700, 725, 750, 762, 775, 787, 800, 810, 818, 825] # Cut-offs to test in validation

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
# DATA LOADING & SPLITTING (Train / Val / Test)
# ============================================================
train_file_path = "Project/superconductivty+data/train.csv"
labels_path     = "Project/superconductivty+data/unique_m.csv"

df       = pd.read_csv(train_file_path)
df_labels = pd.read_csv(labels_path)

X = df.drop(columns=["critical_temp"])
Y = df_labels["critical_temp"]

# 1. Split off 15% for the final TEST set
x_temp, x_test, y_temp, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=78
)

# 2. Split the remaining 85% into TRAIN and VALIDATION
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.1765, random_state=78
)

print(f"Features originais: {X.shape[1]}")
print(f"Train size: {x_train.shape[0]} | Val size: {x_val.shape[0]} | Test size: {x_test.shape[0]}")
print("=" * 50)

if USE_POLYNOMIAL:
    poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        include_bias=False,      
        interaction_only=False    
    )
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly   = poly.transform(x_val)
    x_test_poly  = poly.transform(x_test)

    feature_names = poly.get_feature_names_out(X.columns)
    
    print(f"[PolynomialFeatures degree={POLY_DEGREE}]")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Total after squaring/interactions: {x_train_poly.shape[1]}")
    print("=" * 50)
else:
    x_train_poly = x_train.values
    x_val_poly   = x_val.values
    x_test_poly  = x_test.values
    feature_names = X.columns.tolist()

# ============================================================
# SCALING (Post-Expansion)
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
# MRMR FEATURE SELECTION & VALIDATION
# ============================================================
df_train_mrmr = pd.DataFrame(x_train_scaled, columns=feature_names)
df_val_mrmr   = pd.DataFrame(x_val_scaled, columns=feature_names)
df_test_mrmr  = pd.DataFrame(x_test_scaled, columns=feature_names)

print(f"\n[Starting MRMR Ranking for top {MAX_MRMR_FEATURES} features...]")
# 1. Run MRMR exactly ONCE to rank the top MAX_MRMR_FEATURES
ranked_features = mrmr_regression(X=df_train_mrmr, y=y_train.values, K=MAX_MRMR_FEATURES)

# 2. Use the validation set to find the optimal cut-off K
best_k = None
best_val_rmse = float('inf')
best_features = []

print(f"\n[Validating Cut-offs for Top K Features]")
for k in K_CANDIDATES:
    if k > len(ranked_features):
        continue
        
    features_to_test = ranked_features[:k]
    
    temp_model = LinearRegression()
    temp_model.fit(df_train_mrmr[features_to_test], y_train)
    
    val_preds = temp_model.predict(df_val_mrmr[features_to_test])
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    print(f"  Tested Top K={k:3d} | Validation RMSE: {val_rmse:.4f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_k = k
        best_features = features_to_test

print(f"\n=> Best K selected: {best_k} (Validation RMSE: {best_val_rmse:.4f})")
print("=" * 50)

# 3. Finalize datasets with the optimal features
x_train_final = df_train_mrmr[best_features].values
x_test_final  = df_test_mrmr[best_features].values

# ============================================================
# TRAINING (Final Model)
# ============================================================
print(f"=== Settings ===")
print(f"Model: {MODEL}")
print(f"Scaler: {SCALER}")
print(f"Polynomial: {'Yes (degree=' + str(POLY_DEGREE) + ')' if USE_POLYNOMIAL else 'No'}")
print("-" * 40)

model = models[MODEL]
model.fit(x_train_final, y_train)

if hasattr(model, 'intercept_'):
    print(f"Intercept: {model.intercept_:.4f}")

print("-" * 40)

# ============================================================
# EVALUATION(On Unseen Test Data)
# ============================================================
y_pred = model.predict(x_test_final)

def evaluate_model(y_true, y_pred):
    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    mape   = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"MSE      : {mse:.4f} K²")
    print(f"RMSE     : {rmse:.4f} K ")
    print(f"MAE      : {mae:.4f} K ")
    print(f"R²       : {r2:.4f}")
    print(f"MAPE     : {mape:.4f}")
 
    n_neg = (y_pred < 0).sum()
    print(f"Negative preds (physically impossible): {n_neg} ({100 * n_neg / len(y_pred):.1f}%)")
 
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
 
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