import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error, mean_absolute_percentage_error



train_file_path = "Project/superconductivty+data/train.csv"
labels_path = "Project/superconductivty+data/unique_m.csv"

df = pd.read_csv(train_file_path)
df_labels = pd.read_csv(labels_path)

X = df.drop(columns=["critical_temp"])
Y = df_labels["critical_temp"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(x_train,y_train)

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(x_test)

def evaluate_model(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mape = mean_absolute_percentage_error(y_true,y_pred)

    print(f'MSE: {mse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}')

print(y_pred.shape)
evaluate_model(y_pred,y_test)

def compare_predictions(y_true, y_pred, num_samples=15):

    comparison_df = pd.DataFrame({
        'Valor Real': y_true,
        'Valor Predito': y_pred,
        'Erro Absoluto': abs(y_true - y_pred)
    })
    
    print(f"--- Primeiras {num_samples} amostras do Teste ---")
    print(comparison_df.head(num_samples))
    print("\n")
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolor='k')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Predição Perfeita (y=x)')

    plt.title("Valores Reais vs. Valores Preditos")
    plt.xlabel("Temperatura Crítica Real")
    plt.ylabel("Temperatura Crítica Predita pelo Modelo")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig("Project/results/grafico_linear_regression.png", dpi=300, bbox_inches='tight')

compare_predictions(y_test, y_pred)

