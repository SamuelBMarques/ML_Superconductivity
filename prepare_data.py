import pandas as pd
from sklearn.model_selection import train_test_split
import os

df        = pd.read_csv("Project/superconductivty+data/train.csv")
df_labels = pd.read_csv("Project/superconductivty+data/unique_m.csv")

X = df.drop(columns=["critical_temp"])
Y = df_labels["critical_temp"]

# 1. Carve out the test set permanently
x_temp, x_test, y_temp, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=78
)

# 2. Split the remainder into train and val
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.1765, random_state=78
)

os.makedirs("Project/splits", exist_ok=True)

x_train.assign(critical_temp=y_train.values).to_csv("Project/splits/train.csv", index=False)
x_val.assign(critical_temp=y_val.values).to_csv("Project/splits/val.csv",   index=False)
x_test.assign(critical_temp=y_test.values).to_csv("Project/splits/test.csv",  index=False)

print(f"Train : {x_train.shape[0]} samples")
print(f"Val   : {x_val.shape[0]} samples")
print(f"Test  : {x_test.shape[0]} samples")
print("Splits saved to Project/splits/")