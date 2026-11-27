# Student Performance Prediction

A beginner-friendly Data Science and Machine Learning project that estimates a student's final Mathematics grade (G3) using the UCI Student Performance dataset.

## Team

- **TANMAY RAGHAV**
- **ADITYA SIROHI**
- **RITIVIK SHARMA**

**Submitted to:** Professor Nirbhay  
**Academic Year:** 2026

## Project objective

Explore student-related data, visualise patterns, prepare the dataset for machine learning, train two regression models, and compare their predictions using MAE, MSE, RMSE, and R².

## Repository files

| File | Purpose |
|---|---|
| student_performance_prediction.py | Data inspection, EDA, preprocessing, model training, and evaluation |
| student-mat.csv | Mathematics subset of UCI Student Performance (395 records, 33 attributes) |
| requirements.txt | Python libraries required to run the project |
| reports/Student_Performance_Prediction_Project_Report.md | Project report |

The script creates a results folder automatically. It saves the graphs, model comparison table, and sample predictions there.

## Dataset attribution

The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

Dataset citation: Cortez, P. (2008). *Student Performance* [Dataset]. https://doi.org/10.24432/C5TG7T.

The dataset is licensed under **CC BY 4.0**; attribution is provided here.

## Requirements

Python 3.10 or newer is recommended.

Install the required libraries:

    python -m pip install -r requirements.txt

## Run the project

Keep student_performance_prediction.py and student-mat.csv in the same folder, then run:

    python student_performance_prediction.py

The script prints data-quality checks, evaluation metrics, and sample predictions. It also displays the graphs and saves them in the results folder.

## Methodology

1. Load and inspect the Mathematics dataset.
2. Explore final-grade distribution, study-time patterns, absences, and numerical correlations.
3. Split the data into 80% training and 20% test sets using random_state=42.
4. Convert categorical features into numeric columns and align the test columns with the training columns.
5. Scale the features for Linear Regression. The scaler is fitted only on training data.
6. Train and compare Linear Regression and Random Forest using the same train-test split.
7. Evaluate both models using MAE, MSE, RMSE, and R².

## Models

- **Linear Regression:** A simple baseline regression model.
- **Random Forest Regressor:** Uses 200 decision trees and random_state=42.

Lower MAE, MSE, and RMSE indicate smaller prediction errors. A higher R² generally indicates a better fit relative to a mean-baseline model.

## Important modelling note

The models include **G1** and **G2** as inputs when predicting **G3**. This represents a situation where first- and second-period grades are already available. For a different task that predicts final grades without earlier grades, remove G1 and G2 and evaluate that scenario separately.

The project is for academic learning and should not be used alone to make high-stakes decisions about students.
