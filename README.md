# 📈 Volatility Smile Predictor

A robust machine learning pipeline for predicting implied volatility smiles of NIFTY50 index options using LightGBM. This project features advanced preprocessing, cross-IV feature engineering, and K-Fold model training to deliver accurate IV surface forecasts from high-dimensional market data.

---

## 🚀 Project Overview

This repository provides a complete pipeline for forecasting the implied volatility (IV) surface of NIFTY50 index options, developed for the Kaggle Volatility Smile Prediction Challenge. The approach leverages high-frequency market data and state-of-the-art modeling to capture the complex dynamics of option IVs.

---

## 🧠 Key Features

- **LightGBM Regression Models** with regularization and early stopping
- **Cross-IV Features** to capture call-put strike interactions
- **Outlier Detection** to remove invalid IV entries
- **Denoising** using Exponentially Weighted Moving Average (EWM)
- **Zero-dominant Feature Removal** and **Low Variance Filtering**
- **Highly Correlated Feature Elimination**
- **NaN-aware Feature Engineering** (missing value flags)
- **K-Fold Cross Validation** with strike-wise partitioning
- **Failsafe Prediction Filling** for missing outputs

---

## 📂 Directory Structure

volatility-smile-prediction/
├── train_inputs_cleaned.csv # Cleaned and denoised training data
├── test_inputs_cleaned.csv # Cleaned and denoised test data
├── submission.csv # Final prediction output for Kaggle submission
├── notebook.ipynb # Core notebook with full modeling pipeline
└── README.md # Project documentation

text

---

## 📊 Data Sources

- **train_data.parquet**: Raw training set with features (`X` columns), call/put IVs, and metadata
- **test_data.parquet**: Raw test set with input features only
- **sample_submission.csv**: Template for constructing the prediction file

---

## 🔍 Preprocessing Steps

- **Outlier Removal:** Exclude IV values outside [0.0, 1.0]
- **Feature Filtering:** Drop columns with >50% zeros or std < 0.01; remove features with Pearson correlation > 0.98
- **IV Denoising:** Apply EWM smoothing across timestamps
- **Cross-IV Feature Generation:** Add cross-referenced call/put IV features for each strike
- **Clipping:** Winsorize all features at the 1st and 99th percentiles
- **Feature Engineering:** Add `iv_mean`, `iv_std`, and missing value indicators (`col_na`)

---

## ⚙️ Model Configuration

model = LGBMRegressor(
objective='regression',
learning_rate=0.01,
num_leaves=128,
max_depth=10,
feature_fraction=0.9,
bagging_fraction=0.9,
bagging_freq=1,
n_estimators=4000,
lambda_l1=1.0,
lambda_l2=1.0,
min_child_samples=20,
random_state=42,
n_jobs=-1,
)

text
- **Evaluation Metric:** Root Mean Squared Error (RMSE)

---

## 🔄 Training Strategy

- Train a separate model for each target IV column (per strike)
- Early stopping based on validation RMSE
- 5-Fold cross-validation (first fold used for efficiency)

---

## 📤 Submission Logic

- Generate predictions for each target column
- For targets lacking training data, fill with `0.2 ± 0.05` random value
- Save final predictions in `submission.csv` for competition upload

---

## 🛠️ How to Run

1. **Install dependencies:**
pip install lightgbm pandas numpy scikit-learn tqdm matplotlib

text
2. **Place data files** in a `/data` directory or update `data_path` in the notebook.
3. **Run** `notebook.ipynb` or the script version to reproduce results.

---

## 📚 Acknowledgments

Developed for the Kaggle Volatility Smile Prediction Challenge. Inspired by quantitative finance and options modeling strategies, with a focus on robust, scalable machine learning for financial data.
