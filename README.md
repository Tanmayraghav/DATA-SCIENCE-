# Student Performance Prediction

A beginner-friendly Data Science project using the **UCI Student Performance Dataset**. The goal is to understand student data and predict the final grade (`G3`) using a simple Linear Regression model.

## Project contents

- [Project explanations](Student_Performance_Prediction/PROJECT_EXPLANATION.md) — the notebook's explanatory text, without Python code cells.
- [Report outline](Student_Performance_Prediction/REPORT_OUTLINE.md) — a simple structure for the college report.
- [Viva notes](Student_Performance_Prediction/VIVA_NOTES.md) — easy definitions and common viva questions.
- [UCI dataset](https://archive.ics.uci.edu/dataset/320/student+performance) — the original dataset and documentation.

## Topics covered

- Exploratory Data Analysis (EDA) and graphs
- Data analysis and data quality checks
- Missing values and duplicate checks
- Outlier detection and removal using the IQR method
- Categorical encoding (one-hot encoding)
- Min-Max scaling and standardization
- Train-test split
- Linear Regression model training
- Evaluation using MAE, RMSE, and R²

## Dataset and target

The notebook uses the UCI Student Performance dataset. The target is `G3`, the final grade. Earlier grades `G1` and `G2` are among the inputs, so this predicts the final grade when earlier-period grades are already known.

## How to use

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the project notebook from your local copy.
3. Run cells from top to bottom.
4. Record the actual MAE, RMSE, and R² values shown by the notebook in the report. Do not use made-up results.

## Dataset citation

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
