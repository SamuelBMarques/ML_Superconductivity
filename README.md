# 🧪 Machine Learning for Superconductivity

A **Machine Learning project applied to materials science**, focused on predicting the **critical temperature (`Tc`) of superconducting materials** from their chemical and physical properties.

The project uses the publicly available superconductivity dataset provided by **Kam Hamidieh** through the UCI Machine Learning Repository.

---

## 🎯 Objective

The **critical temperature (`Tc`)** is the temperature below which a material exhibits superconducting behavior.

The goal of this project is to use **regression techniques** to estimate the critical temperature of superconducting materials based on their properties.

The dataset contains:

* **21,263 superconducting compounds**
* **81 features** derived from their chemical formulas

---

## 🔬 Machine Learning Pipeline

The project follows a typical Machine Learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
PCA
   ↓
Model Training
   ↓
Model Evaluation
```

---

## 🤖 Models

Different regression algorithms were implemented to compare linear and non-linear approaches:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Random Forest
* Gradient Boosting

The models are evaluated using **K-Fold Cross-Validation**.

---

## 📊 Evaluation

The models are compared using common regression metrics:

* **R²** — Coefficient of Determination
* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error

The project also includes exploratory analyses using **correlation matrices** and **2D/3D PCA visualizations**.

---

## 🛠️ Technologies

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* Machine Learning
* Data Analysis
* Data Visualization

---

## 📁 Project Structure

```text
ML_Superconductivity/
│
├── exploration/
├── results/
├── results_final/
├── splits/
├── superconductivty+data/
│
├── crit_temp_knn.py
├── crit_temp_linearregression.py
├── crit_temp_randomforest.py
├── linear_regression.py
├── prepare_data.py
├── result_analysis.py
│
└── README.md
```

* `prepare_data.py` — Data preparation and preprocessing
* `crit_temp_*.py` — Training and evaluation of different models
* `linear_regression.py` — Linear regression implementation
* `result_analysis.py` — Analysis and visualization of model results

---

## 💻 Requirements

* Python `3.8+`
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

Install the dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## ▶️ Usage

First, prepare the dataset:

```bash
python prepare_data.py
```

Then, individual models can be trained and evaluated:

```bash
python crit_temp_knn.py
python crit_temp_linearregression.py
python crit_temp_randomforest.py
```

---

## 📚 Dataset

**Superconductivity Dataset**

Provided by **Kam Hamidieh** through the **UCI Machine Learning Repository**.

The dataset contains chemical and physical properties of superconducting materials, with the goal of predicting their critical temperature (`Tc`).

---

## 👨‍💻 Author

**Samuel Braga Marques**

GitHub: [@SamuelBMarques](https://github.com/SamuelBMarques)
