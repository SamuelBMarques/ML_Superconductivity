import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


MODELO_ESCOLHIDO = "Linear"  
SCALER_ESCOLHIDO = "Standard" 

#Models and scalers for comparison
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=78),
    "Lasso": Lasso(alpha=1.0, random_state=78, max_iter=10000),
    "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5),                     #0.001, 0.01, 0.1, 1, 10, 100, 1000
    "LassoCV": LassoCV(alphas=10, cv=5, random_state=78, max_iter=50000)
}

scalers = {
    "Standard": sklearn.preprocessing.StandardScaler(),
    "Nenhum": None
}

#Data loading and splitting
train_file_path = "Project/superconductivty+data/train.csv"
labels_path = "Project/superconductivty+data/unique_m.csv"

df = pd.read_csv(train_file_path)
df_labels = pd.read_csv(labels_path)

X = df.drop(columns=["critical_temp"])
Y = df_labels["critical_temp"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=78
)

#Data preprocessing
scaler = scalers[SCALER_ESCOLHIDO]

if scaler is not None:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
else:
    x_train = x_train.values
    x_test = x_test.values

#Training
model = models[MODELO_ESCOLHIDO]
model.fit(x_train, y_train)

print(f"=== Configuração Executada ===")
print(f"Modelo: {MODELO_ESCOLHIDO}")
print(f"Scaler: {SCALER_ESCOLHIDO}")
print("-" * 30)

if hasattr(model, 'alpha_'):
    print(f"Melhor Alpha encontrado por CV: {model.alpha_}")
if hasattr(model, 'coef_'):
    print(f"Primeiro Slope (coef_): {model.coef_[0]}")
    if MODELO_ESCOLHIDO in ["Lasso", "LassoCV"]:
        features_ativas = np.sum(model.coef_ != 0)
        print(f"Features mantidas (coef != 0): {features_ativas}/{X.shape[1]}")
if hasattr(model, 'intercept_'):
    print(f"Intercept: {model.intercept_}")

print("-" * 30)

#Prediction and evaluation
y_pred = model.predict(x_test)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'MSE: {mse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}')

evaluate_model(y_test, y_pred)

#Plotting predictions vs real values
def compare_predictions(y_true, y_pred, num_samples=15):
    comparison_df = pd.DataFrame({
        'Valor Real': y_true,
        'Valor Predito': y_pred,
        'Erro Absoluto': abs(y_true - y_pred)
    })
    
    print(f"\n--- Primeiras {num_samples} Amostras do Teste ---")
    print(comparison_df.head(num_samples))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolor='k')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Predição Perfeita (y=x)')

    plt.title(f"Reais vs. Preditos - {MODELO_ESCOLHIDO} ({SCALER_ESCOLHIDO})")
    plt.xlabel("Temperatura Crítica Real")
    plt.ylabel("Temperatura Crítica Predita")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Salva o gráfico dinamicamente com o nome do modelo para não sobrescrever os anteriores
    os.makedirs("Project/results", exist_ok=True)
    nome_arquivo = f"Project/results/grafico_{MODELO_ESCOLHIDO.lower()}_{SCALER_ESCOLHIDO.lower()}.png"
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em: {nome_arquivo}")

compare_predictions(y_test, y_pred)