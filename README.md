<p align="center"><img src="assets/project-banner.svg" alt="Student Performance Prediction" width="100%"></p>

<p align="center"><b>Machine Learning Project | B.Tech CSE</b><br>GLA University, Greater Noida · 2026</p>

# Student Performance Prediction

A machine-learning project using the UCI Student Performance Mathematics dataset to estimate final grade (G3).

## Two prediction settings

1. **With G1 and G2:** estimates G3 when first- and second-period grades are already known.
2. **Without G1 and G2:** estimates G3 using the other student attributes.

These are different prediction tasks. Including earlier grades is not automatically data leakage when they are genuinely available at prediction time.

## Models and evaluation

- Mean baseline, Linear Regression, Random Forest and tuned Random Forest.
- Five-fold cross-validation and GridSearchCV use training data only.
- One 80:20 hold-out test split is kept for final evaluation.
- Pipeline and ColumnTransformer handle imputation, encoding and scaling.
- Metrics: MAE, MSE, RMSE and R².
- Visuals: actual vs predicted, residual plots and feature importance.

**Results:** Run the notebook or Python script to generate updated metrics. Previous single-split scores are not presented as results for these expanded experiments.

## G3 = 0 records

Zero-grade records are retained. The project reports their count and checks performance on positive-grade test records separately. The dataset does not establish why an individual record has a zero grade, so no cause is assumed.

## Dataset

[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance). The Mathematics file contains 395 records and 33 attributes. The target is G3 (0–20).

## Run

**Colab:** [Open notebook](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/notebooks/Student_Performance_Prediction.ipynb) and select Runtime → Run all. The notebook downloads the CSV if it is not found locally.

**Local:** Install with `python -m pip install -r requirements.txt`, then run `python src/student_performance_prediction.py` from the repository root.

Graphs are displayed directly; graph and result files are not saved.

## Repository structure

```text
DATA-SCIENCE-/
├── assets/project-banner.svg
├── data/student-mat.csv
├── notebooks/Student_Performance_Prediction.ipynb
├── reports/Student_Performance_Prediction_Project_Report.pdf
├── src/student_performance_prediction.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Team

- Tanmay Raghav
- Aditya Sirohi
- Ritivik Sharma

**Submitted to:** Professor Nirbhay · **Institution:** GLA University, Greater Noida

## Limitations

- The dataset contains 395 Mathematics records.
- Cross-validation supports model selection but is not external validation.
- GridSearchCV tests a small predefined parameter grid.
- Feature importance is not causal evidence.
- Predictions should not be used alone for important decisions about individual students.

## Reference

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).