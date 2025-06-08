# volatility-smile-predictor
A robust machine learning pipeline for predicting implied volatility smiles of NIFTY50 index options using LightGBM. The model incorporates outlier removal, denoising, feature engineering (including cross-IV features), and K-Fold training. It aims to deliver accurate IV surface forecasts from high-dimensional market data.


📈 Volatility Smile Prediction for NIFTY50 Options
This repository contains a complete machine learning pipeline built to predict the implied volatility (IV) surface of NIFTY50 index options, leveraging historical high-frequency market data. The solution was developed for the Volatility Smile Prediction Challenge on Kaggle, with a focus on modeling the complex behavior of option IVs using advanced preprocessing and robust LightGBM models.

🧠 Project Highlights
✅ LightGBM Regression Models with regularization and early stopping
✅ Cross-IV Features capturing call-put strike interactions
✅ Outlier Detection to remove invalid or unrealistic IV entries
✅ Denoising using EWM (Exponentially Weighted Moving Average)
✅ Zero-dominant Feature Removal and Low Variance Filtering
✅ Highly Correlated Feature Elimination
✅ NaN-aware Feature Engineering
✅ K-Fold Cross Validation with strike-wise data partitioning
✅ Failsafe Prediction Filling for missing outputs

📂 Files & Structure

📦 volatility-smile-prediction/
├── train_inputs_cleaned.csv     # Cleaned and denoised training data
├── test_inputs_cleaned.csv      # Cleaned and denoised test data
├── submission.csv               # Final prediction output for Kaggle submission
├── notebook.ipynb               # Core notebook with full modeling pipeline
└── README.md                    # You're here!
📊 Data Sources
train_data.parquet: Raw training set with input features (X columns), call/put IVs, and metadata

test_data.parquet: Raw test set with input features only

sample_submission.csv: Template for constructing the prediction file

🔍 Preprocessing Steps

Outlier Removal
Implied volatility values outside [0.0, 1.0] are removed to ensure clean training targets.

Feature Filtering

Columns with >50% zeros are dropped.

Features with std < 0.01 are dropped as uninformative.

Highly correlated features (Pearson corr > 0.98) are removed.

IV Denoising
EWM smoothing is applied across timestamps to reduce high-frequency noise in IV values.

Cross-IV Feature Generation
For each strike with both call and put IVs, cross-referencing is used to enhance model inputs.

Clipping Outliers (1% Winsorization)
Applied on all features to prevent model sensitivity to extreme values.

Feature Engineering

Aggregates: iv_mean, iv_std

Missing-value indicators: col_na flags

⚙️ Model Configuration
The core model uses LGBMRegressor with the following hyperparameters:

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
The evaluation metric is Root Mean Squared Error (RMSE).

🔄 Training Strategy
Separate model trained per target IV column (per strike)

Early stopping based on validation RMSE

5-Fold cross-validation used, with only the first fold trained for efficiency

📤 Submission Logic
Predictions generated per target column

For targets with no training data, fallback to 0.2 ± 0.05 random fill

Final predictions saved in submission.csv

🛠️ How to Run
Note: This project was designed for the Kaggle environment with competition-specific data. To run it locally, ensure that you replicate the directory structure and have the required data files.

Install requirements:

pip install lightgbm pandas numpy scikit-learn tqdm matplotlib
Place data files into a /data directory or modify data_path.

Run notebook.ipynb or script version.

📚 Acknowledgments
This work was built as part of the Kaggle Volatility Smile Prediction challenge. Inspired by quantitative finance modeling strategies, especially in options trading and volatility surface estimation.
