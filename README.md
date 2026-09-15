# ML_Superconductivity

This project focuses on predicting the Critical Temperature ($T_c$) of superconducting materials based on their chemical and physical properties using machine learning techniques. It utilizes the public dataset provided by Dr. Kam Hamidieh from the UCI Machine Learning Repository.

## How It Works

The application performs data preprocessing, exploratory data analysis (EDA), and builds regression models to estimate the critical temperature of superconductors.

1. **Data Processing**: Loads dataset consisting of 21,263 superconducting compounds and 81 features extracted from their chemical formulas.
2. **Feature Engineering & Scaling**: Handles missing values/duplicates and applies `StandardScaler` to standardize feature scales across different physical properties.
3. **Dimensionality Reduction**: Utilizes Principal Component Analysis (PCA) for visualization and feature exploration.
4. **Model Training & Evaluation**: Evaluates multiple regression models (Linear Regression, Support Vector Regressors, Random Forests, Gradient Boosting) using K-Fold Cross-Validation.

## Features

* **Exploratory Data Analysis**: Includes correlation matrix visualization and PCA projections in 2D/3D.
* **Comparative Model Benchmark**: Compares linear vs. non-linear algorithms based on metrics such as $R^2$, RMSE, and MAE.
* **Robust Preprocessing**: Standardizes raw physical properties (e.g., atomic mass, thermal conductivity, electron affinity) for machine learning compatibility.

## Requirements

* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib
