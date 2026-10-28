# Student Performance Prediction

A beginner-friendly Data Science and Machine Learning project that estimates a student's final Mathematics grade (**G3**) using Linear Regression and the UCI Student Performance dataset.

## Team

- **TANMAY RAGHAV**
- **ADITYA SIROHI**
- **RITIVIK SHARMA**

**Submitted to:** Professor Nirbhay  
**Academic Year:** 2026

## Project objective

Explore student-related data, visualise important patterns, prepare the data for machine learning, train a Linear Regression model, and evaluate predictions using MAE, MSE, RMSE, and R².

## Repository files

| File | Purpose |
|---|---|
| `student_performance_prediction.py` | Python code for data inspection, EDA, preprocessing, training, and evaluation |
| `student-mat.csv` | Mathematics subset of the UCI Student Performance dataset (395 records, 33 attributes) |
| `reports/Student_Performance_Prediction_Project_Report.docx` | Full project report |

## Dataset and attribution

The dataset is the Mathematics subset (`student-mat.csv`) of the **Student Performance** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance). It contains student demographic, family, school, study-time, attendance, and grade information. The original target **G3** is the final grade on a 0–20 scale.

Dataset citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. The dataset is licensed under **CC BY 4.0**; attribution is provided here.

## Requirements

Python 3.10+ is recommended.

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run the project

Keep `student_performance_prediction.py` and `student-mat.csv` in the same folder, then run:

```bash
python student_performance_prediction.py
```

The script prints data-quality checks and evaluation metrics, and opens EDA and prediction plots. It does not hard-code or invent the model results.

## Methodology

1. Load and inspect the CSV data.
2. Examine grade distribution, study-time patterns, absences, and numerical correlations.
3. Split data into 80% training and 20% test sets (`random_state=42`).
4. Encode categorical variables and align test columns with training columns.
5. Fit `StandardScaler` using training features only.
6. Train Linear Regression and evaluate it with MAE, MSE, RMSE, and R².

## Important modelling note

The current model includes **G1** and **G2** as input features to predict **G3**. This is appropriate when earlier-period grades are available at prediction time. If the intended task is to predict final grades without earlier grades, remove G1 and G2 from the features and evaluate that separate scenario.

This project is for academic learning and should not be used alone to make high-stakes decisions about students.
