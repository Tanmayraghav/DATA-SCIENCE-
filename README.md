# Student Performance Prediction

A beginner-friendly Data Science and Machine Learning project that estimates a student's final Mathematics grade (**G3**) using Linear Regression and the UCI Student Performance dataset.

## Team
- **TANMAY RAGHAV**
- **ADITYA SIROHI**
- **RITIVIK SHARMA**

**Submitted to:** Professor Nirbhay  
**Academic Year:** 2026

## Project objective
Explore student-related data, visualise patterns, prepare the dataset for machine learning, train a Linear Regression model, and evaluate predictions using MAE, MSE, RMSE, and R².

## Repository files
| File | Purpose |
|---|---|
| student_performance_prediction.py | Data inspection, EDA, preprocessing, model training, and evaluation |
| student-mat.csv | Mathematics subset of UCI Student Performance (395 records, 33 attributes) |
| reports/Student_Performance_Prediction_Project_Report.md | Full project report |

## Dataset attribution
The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance). Dataset citation: Cortez, P. (2008). *Student Performance* [Dataset]. https://doi.org/10.24432/C5TG7T. The dataset is licensed under **CC BY 4.0**; attribution is provided here.

## Requirements
Python 3.10+ is recommended. Install dependencies:

~~~bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
~~~

## Run
Keep student_performance_prediction.py and student-mat.csv in the same folder, then run:

~~~bash
python student_performance_prediction.py
~~~

The script prints data-quality checks and evaluation metrics, and opens EDA and prediction plots.

## Methodology
1. Load and inspect the CSV.
2. Explore grade distribution, study-time patterns, absences, and numerical correlations.
3. Split data into 80% training and 20% test sets (random_state=42).
4. Encode categorical features and align test columns with training columns.
5. Fit StandardScaler using training features only.
6. Train Linear Regression and evaluate MAE, MSE, RMSE, and R².

## Important modelling note
The current model includes **G1** and **G2** as inputs when predicting **G3**. If the goal is to predict final grades without earlier grades, remove G1 and G2 and evaluate that scenario separately.

This project is for academic learning and should not be used alone to make high-stakes decisions about students.
