# Student Performance Prediction — Project Report

**Team:** Tanmay Raghav, Aditya Sirohi, Ritivik Sharma  
**Submitted to:** Professor Nirbhay, GLA University, Greater Noida  
**Programme:** B.Tech CSE | 2026

## 1. Project overview

This project uses the Mathematics portion of the UCI Student Performance dataset to predict the final grade (G3) from student academic, demographic and behavioural attributes. Two prediction settings are compared: one with first- and second-period grades (G1 and G2) available, and one without those columns.

## 2. Dataset

The included `student-mat.csv` contains **395 records and 33 columns**. The target, `G3`, is the final Mathematics grade on a 0–20 scale. The included CSV has no missing values. There are **38 records with G3 = 0** (about 9.6%); all 38 also show zero absences. The dataset does not explain why these grades are zero, so no cause is assumed.

Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).  
Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).

## 3. Methodology

- A fixed 80:20 train/test split is used with `random_state=42`.
- Models: mean baseline, Linear Regression, default Random Forest and tuned Random Forest.
- Five-fold cross-validation and GridSearchCV use training data only.
- Preprocessing uses Scikit-learn Pipelines and ColumnTransformer: numeric median imputation, categorical most-frequent imputation and one-hot encoding. Numeric features are scaled for Linear Regression.
- The tuned Random Forest searches 100/200 trees, maximum depth None/10/20, and minimum leaf size 1/2.
- Hold-out metrics: MAE, MSE, RMSE and R².
- Notebook visuals include the grade distribution, study-time comparison, G2-vs-G3 plot, actual-vs-predicted plot, residual plot and feature importance.

## 4. Prediction settings

**With G1/G2:** predicts G3 when first- and second-period grades are known at prediction time.

**Without G1/G2:** predicts G3 using the other attributes when earlier grades are not available.

Earlier grades are not automatically data leakage when they are genuinely available at prediction time. The two settings are different prediction tasks.

## 5. Hold-out test results

| Prediction setting | Model | MAE | MSE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| With G1/G2 | Mean baseline | 3.646 | 20.704 | 4.550 | -0.010 |
| With G1/G2 | Linear Regression | 1.647 | 5.657 | 2.378 | 0.724 |
| With G1/G2 | Random Forest (default) | **1.188** | **4.028** | **2.007** | **0.804** |
| With G1/G2 | Random Forest (tuned) | 1.195 | 4.076 | 2.019 | 0.801 |
| Without G1/G2 | Mean baseline | 3.646 | 20.704 | 4.550 | -0.010 |
| Without G1/G2 | Linear Regression | 3.395 | 17.604 | 4.196 | 0.141 |
| Without G1/G2 | Random Forest (default) | **2.994** | **14.051** | **3.748** | **0.315** |
| Without G1/G2 | Random Forest (tuned) | 3.052 | 14.610 | 3.822 | 0.287 |

MAE and RMSE are in grade points. Lower MAE/RMSE and higher R² indicate better test performance.

## 6. Cross-validation results

Five-fold cross-validation RMSE on the training split:

| Model | With G1/G2 (mean ± std) | Without G1/G2 (mean ± std) |
|---|---:|---:|
| Mean baseline | 4.575 ± 0.372 | 4.575 ± 0.372 |
| Linear Regression | 1.895 ± 0.209 | 4.699 ± 0.217 |
| Random Forest (default) | 1.420 ± 0.253 | 3.878 ± 0.277 |
| Random Forest (tuned) | 1.386 (best mean) | 3.875 (best mean) |

The tuned Random Forest achieved slightly lower cross-validation RMSE but did not improve the held-out test scores in this run. The default Random Forest therefore performed better on this saved test split.

## 7. Interpretation and sensitivity analysis

The default Random Forest achieved hold-out RMSE 2.007 and R² 0.804 with G1/G2; without them, RMSE was 3.748 and R² was 0.315. This indicates that earlier grades contain strong predictive information in this dataset.

The 38 zero-grade records remain in the primary experiment. On positive-grade test records only, the tuned Random Forest achieved:
- With G1/G2: MAE 0.978, RMSE 1.562, R² 0.820.
- Without G1/G2: MAE 2.734, RMSE 3.298, R² 0.195.

This secondary check uses a smaller test subset and does not replace the primary full-test results. Feature importance is descriptive and is not evidence of causality.

## 8. Limitations and conclusion

The dataset is small, limited to Mathematics and drawn from one educational context. No external validation is available. A single hold-out split can vary, and the grid search covers only a small parameter set. Predictions should not be used alone for high-impact decisions about individual students.

The project demonstrates exploratory data analysis, fold-safe preprocessing, baseline/model comparison, cross-validation, hyperparameter tuning, held-out evaluation and residual/feature-importance analysis.
