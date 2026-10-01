# Walmart Weekly Sales Prediction

## Project Overview

This project develops and evaluates multiple machine learning regression models to predict **Walmart weekly sales** using historical store, economic, and temporal information.

Because the dataset contains weekly observations over time, the project follows a **chronological machine learning workflow** rather than a random train-test split. This helps preserve the temporal structure of the data and reduces the risk of information from future weeks leaking into model training.

Several regression algorithms were compared, and the final model was selected based on validation performance using **RMSE as the primary evaluation metric**.

The final selected model was a **tuned K-Nearest Neighbors (KNN) Regressor**.

---

## Business Problem

Accurate weekly sales prediction can help retailers improve:

- Inventory planning
- Resource allocation
- Demand forecasting
- Operational decision-making
- Store-level planning

The objective of this project is to build a regression model capable of estimating `weekly_sales` from historical store and economic information while respecting the chronological nature of the dataset.

---

## Dataset

The dataset contains **6,435 weekly observations** from **45 Walmart stores**.

The original data covers the period from:

**February 5, 2010 → October 26, 2012**

### Original Features

| Feature | Description |
|---|---|
| `store` | Store identifier |
| `date` | Weekly observation date |
| `weekly_sales` | Weekly store sales — target variable |
| `is_holiday` | Indicates whether the week contains a holiday |
| `temperature` | Temperature during the week |
| `fuel_price` | Regional fuel price |
| `cpi` | Consumer Price Index |
| `unemployment` | Unemployment rate |

---

## Project Workflow

The project follows the workflow below:

1. Data loading and inspection
2. Data cleaning
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. Outlier handling
6. Chronological train-validation-test splitting
7. Feature preprocessing
8. Time-aware cross-validation
9. Model training
10. Hyperparameter tuning
11. Validation-based model comparison
12. Final model selection
13. Final test evaluation
14. Error analysis
15. Model serialization and verification

---

## Feature Engineering

Several temporal features were extracted from the `date` column for exploratory analysis and model development.

These included:

- Year
- Quarter
- Season
- Month
- Month name
- Week number

For model training, `month` and `week` were retained as temporal predictors.

The original `date` variable was used to create the chronological data split and was not passed directly to the regression models.

---

## Exploratory Data Analysis

EDA was performed to investigate:

- Distribution of weekly sales
- Sales differences across stores
- Holiday vs. non-holiday sales
- Monthly and yearly sales patterns
- Seasonal patterns
- Relationships between sales and economic variables
- Temporal changes in sales
- Correlations among numerical variables
- Potential outliers

The analysis also considered the incomplete time coverage of 2010 and 2012 when interpreting yearly and seasonal aggregates.

---

## Outlier Handling

Potential extreme observations were identified during exploratory analysis.

For this experiment, observations meeting the following conditions were removed:

- `temperature < 7`
- `unemployment < 4.4`
- `unemployment > 11`

This filtering strategy was retained as part of the experimental preprocessing workflow.

However, later error analysis showed that this preprocessing decision contributed to some stores being absent from the training and validation periods. This limitation is discussed in the **Limitations** section.

---

## Chronological Data Split

Because this is time-dependent data, a **random split was avoided**.

The dataset was sorted chronologically and split using unique weekly dates so that observations belonging to the same week remained in the same subset.

Approximately:

- **64%** of unique weeks → Training
- **16%** → Validation
- **20%** → Test

The test set represents the latest period and was kept separate during model selection and hyperparameter tuning.

The final test period covers:

**April 13, 2012 → October 26, 2012**

---

## Data Preprocessing

A Scikit-learn `ColumnTransformer` was used to apply different transformations according to feature type.

### Numerical Features

The following numerical features were processed using:

- Median imputation
- Standard scaling

Features:

- `temperature`
- `fuel_price`
- `cpi`
- `unemployment`
- `month`
- `week`

### Categorical Feature

`store` was:

- Imputed using the most frequent value
- One-hot encoded

The encoder was configured with:

```python
OneHotEncoder(handle_unknown='ignore')
```

This prevents the pipeline from failing when an unseen store category appears during prediction.

### Binary Feature

`is_holiday` was passed through without scaling.

---

## Time-Aware Cross-Validation

Time-aware cross-validation was used during hyperparameter tuning.

Instead of applying cross-validation randomly across individual observations, folds were created using **unique weekly dates**.

This preserves chronological order and helps prevent future observations from leaking into earlier training periods.

---

## Models Evaluated

The following regression models were evaluated:

- Linear Regression
- Polynomial Regression with Ridge Regularization
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor

Hyperparameter tuning was performed where appropriate using time-aware cross-validation.

---

## Validation Results

Models were compared using the validation set before the final test set was accessed.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **KNN (Tuned)** | **107,001.87** | **180,165.53** | **0.9208** |
| Random Forest (Default) | 116,053.45 | 181,645.85 | 0.9195 |
| Random Forest (Tuned) | 127,894.47 | 197,748.98 | 0.9046 |
| Polynomial + Ridge (Tuned) | 130,151.61 | 203,711.80 | 0.8988 |
| Decision Tree (Tuned) | 135,817.54 | 214,637.13 | 0.8876 |
| Linear Regression | 193,553.10 | 281,574.55 | 0.8066 |

### Selected Model

RMSE was used as the primary model-selection metric.

The tuned **KNN Regressor** achieved the lowest validation RMSE and was therefore selected as the final model.

Selected hyperparameters:

```python
KNeighborsRegressor(
    n_neighbors=3,
    weights='distance',
    p=1
)
```

---

## Final Model Training

After model selection, the training and validation datasets were combined.

The selected KNN configuration was then retrained on the combined data before being evaluated on the held-out test set.

The test data was not used for model selection or hyperparameter tuning.

---

## Final Test Results

The final model was evaluated on **1,182 observations** from the held-out test period.

| Metric | Result |
|---|---:|
| **MAE** | **$72,030.09** |
| **RMSE** | **$124,586.80** |
| **R²** | **0.9435** |

The final model explained approximately **94.35% of the variance in weekly sales within the held-out test set**.

The difference between MAE and RMSE also indicated the presence of some observations with relatively large prediction errors.

---

## Error Analysis

Additional analysis was performed to understand where the final model produced its largest errors.

A major difference was observed between stores that were represented during training and stores that were unseen by the model.

| Test Group | Observations | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Seen Stores** | 1,131 | $60,637.24 | $94,567.81 | 0.9681 |
| **Unseen Stores** | 51 | $324,683.38 | $401,766.69 | -0.3566 |

Stores **12, 28, and 38** appeared in the test set but were absent from the combined training and validation data after outlier filtering.

The model therefore performed substantially better on stores represented during training.

Although:

```python
OneHotEncoder(handle_unknown='ignore')
```

allowed unseen store categories to pass through the preprocessing pipeline without producing an error, the model had no learned store-specific representation for those stores.

Several of the largest individual prediction errors were associated with stores **28 and 38**.

---

## Limitations

### 1. Unseen Store Categories

Stores 12, 28, and 38 were not represented in the final training data but appeared during testing.

The model therefore lacked store-specific information for these locations.

### 2. Effect of Outlier Filtering

Outlier filtering was performed before model training and contributed to some stores being absent from the training and validation periods.

The reported results therefore represent performance under this specific preprocessing strategy.

### 3. Generalization to New Stores

The model performed strongly on stores represented during training but substantially worse on previously unseen store identities.

Therefore, the overall test performance should not be interpreted as evidence of equally strong generalization to completely new stores.

### 4. Large Individual Errors

Despite strong overall evaluation metrics, several observations produced large prediction errors.

This is also reflected by RMSE being noticeably higher than MAE.

### 5. Limited Historical Period

The dataset covers February 2010 through October 2012.

The model therefore reflects the economic conditions and sales patterns represented within this historical period and should not automatically be assumed to perform identically in later periods.

---

## Model Saving and Verification

The complete trained pipeline was serialized using **Joblib**:

```python
import joblib

joblib.dump(
    final_knn,
    'walmart_sales_knn_pipeline.pkl'
)
```

The saved model was then reloaded and evaluated again to verify successful serialization.

```python
loaded_model = joblib.load(
    'walmart_sales_knn_pipeline.pkl'
)

loaded_test_results = model_evaluation(
    loaded_model,
    'Reloaded Final Test',
    X_test,
    y_test
)
```

The reloaded model reproduced the same test results.

> **Important:** The saved pipeline contains the preprocessing transformations and KNN model, but `month` and `week` must already be derived from the original date before data is passed to the pipeline.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
Walmart-Weekly-Sales-Prediction/
│
├── walmart-store-prediction.ipynb
├── Walmart.csv
├── walmart_sales_knn_pipeline.pkl
└── README.md
```

---

## Conclusion

This project developed and compared multiple machine learning regression models for predicting Walmart weekly sales while preserving the chronological structure of the data.

Among the evaluated models, the tuned KNN model achieved the lowest validation RMSE and was selected as the final model.

After retraining on the combined training and validation data, the model achieved:

- **MAE:** $72,030.09
- **RMSE:** $124,586.80
- **R²:** 0.9435

The model demonstrated strong predictive performance during the held-out test period for stores represented during training.

However, error analysis revealed substantially weaker performance for stores that were unseen during training, highlighting the importance of evaluating not only aggregate model performance but also model behavior across different subsets of the data.

Overall, the project demonstrates a complete regression workflow involving **time-aware validation, preprocessing pipelines, hyperparameter tuning, model comparison, final test evaluation, error analysis, and model serialization**.