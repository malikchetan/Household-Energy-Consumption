# Household Energy Consumption Forecasting


## Project Overview

This project focuses on predicting household energy consumption using the KAGGLE's Household Power Consumption dataset. 

The work involves data cleaning, preprocessing, feature engineering, model training, and evaluation using multiple machine learning techniques. 

A Streamlit application has also been developed to allow interactive forecasting and visualization of results.

---

## Dataset

* **Source:** Household Power Consumption Kaggle dataset.
* **Size:** 10,48,575 rows and 9 features



## Preprocessing and Feature Engineering

1. **Missing Values:** Replaced '?' with NaN and handled missing data with forward fill and drop operations.
2. **Resampling:** Converted data to 30-minute intervals with sum or average aggregations.
3. **Feature Engineering:** Created time-based features (`Hour`, `Weekday`, `Month`, `Is_weekend`).
4. **Lag and Rolling Features:** Added lag features (1–12), rolling means (3, 6, 12 steps), and lagged values for Voltage and Current.
5. **Scaling:** Applied `StandardScaler` for KNN, while tree-based models did not require scaling.


---

## Models Used:

* **K-Nearest Neighbors (KNN)**
* **Random Forest Regressor:** 
* **Gradient Boosting Regressor** 
* **XGBoost Regressor**

---


## Evaluation and Metrics

Evaluation metrics used:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score (Coefficient of Determination)**

**Results Achieved:**

|             Model |  RMSE  |   MAE  |   R²   |
| ----------------: | :----: | :----: | :----: |
|               KNN | 0.1823 | 0.1153 | 0.8490 |
|     Random Forest | 0.1617 | 0.0262 | 0.8811 |
| Gradient Boosting | 0.1401 | 0.0932 | 0.9024 |
|  XGBoost (Optuna) | 0.1378 | 0.0857 | 0.9138 |

**Naive Baseline:** 
R² = 0.6387 
RMSE = 0.2820
MAE = 0.1643

**XGBoost demonstrated the best performance with the lowest RMSE and the highest R².**

---

## Streamlit Application

The Streamlit application provides an interactive platform to forecast energy consumption:

* Users can input **Hour, Weekday, Month**, and **select a Model**.
* Predictions are generated for the next **6 intervals** .
* Outputs include both **numerical predictions** and a **line chart visualization**.

---

## Live Demo

The deployed application can be accessed here:
 **[Streamlit App Link](https://time-series-phcfztzgaaaeappyquxaf6t.streamlit.app/)**

---

