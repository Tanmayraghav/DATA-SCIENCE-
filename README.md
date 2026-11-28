<p align="center">
  <img src="assets/project-banner.svg" alt="Student Performance Prediction project banner" width="100%">
</p>

<p align="center">
  <b>Machine Learning Project | B.Tech Computer Science &amp; Engineering</b><br>
  GLA University, Greater Noida · Academic Year 2026
</p>

<p align="center">
  <a href="reports/Student_Performance_Prediction_Project_Report.pdf"><img src="https://img.shields.io/badge/Read-Project%20Report-20B8A6?style=for-the-badge" alt="Read project report"></a>
  <a href="notebooks/Student_Performance_Prediction.ipynb"><img src="https://img.shields.io/badge/Open-Notebook-183852?style=for-the-badge" alt="Open notebook"></a>
  <a href="https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/notebooks/Student_Performance_Prediction.ipynb"><img src="https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=for-the-badge&amp;logo=googlecolab&amp;logoColor=white" alt="Run in Google Colab"></a>
</p>

# Student Performance Prediction

An end-to-end beginner-friendly machine learning project that explores student information and predicts the **final Mathematics grade (G3)** using Linear Regression and Random Forest. It covers data inspection, exploratory data analysis (EDA), preprocessing, model training and evaluation.

## Project overview

| Dataset | Target | Models compared | Best test R² |
|:--|:--|:--|--:|
| 395 records · 33 columns | G3 (0–20) | Linear Regression, Random Forest | **0.811** |

## Model results

Results from the fixed 80:20 train-test split (random_state=42). Lower MAE, MSE and RMSE are better; higher R² is better.

| Model | MAE | MSE | RMSE | R² |
|:--|--:|--:|--:|--:|
| **Random Forest** | **1.173** | **3.868** | **1.967** | **0.811** |
| Linear Regression | 1.647 | 5.657 | 2.378 | 0.724 |

Random Forest performed better on this particular test split. This does not guarantee better performance on every dataset or school.

## Workflow

1. Load and inspect the Mathematics dataset, including data types, missing values and duplicates.
2. Explore grade distribution, study time, G2 vs G3, absences and numerical correlations.
3. Split the data into training and test sets (80:20).
4. Encode categorical features; scale features for Linear Regression using a scaler fitted only on training data.
5. Train both models and compare MAE, MSE, RMSE and R².
6. Save plots, model metrics and sample predictions in the results folder.

## Dataset

The project uses the Mathematics CSV (student-mat.csv) from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

- 395 student records and 33 attributes.
- Prediction target: G3, the final Mathematics grade on a 0–20 scale.
- No missing values or duplicate rows were found in the saved project run.

**Important:** G1 and G2 (first- and second-period grades) are included as input features. The model estimates G3 when earlier grades are already available; it is not an early prediction before those grades are known.

## Run locally

### 1. Install dependencies

~~~bash
python -m pip install -r requirements.txt
~~~

### 2. Run the Python script

Run this command from the repository root:

~~~bash
python src/student_performance_prediction.py
~~~

The script reads data/student-mat.csv and creates a results folder containing graphs and CSV result files.

### 3. Run the notebook

Open notebooks/Student_Performance_Prediction.ipynb in Jupyter or use the **Open in Colab** link above. Run the cells from top to bottom. The notebook looks for the dataset locally and downloads it from this repository when running in a fresh Colab session.

## Repository structure

~~~text
DATA-SCIENCE-/
├── assets/
│   └── project-banner.svg
├── data/
│   └── student-mat.csv
├── notebooks/
│   └── Student_Performance_Prediction.ipynb
├── reports/
│   └── Student_Performance_Prediction_Project_Report.pdf
├── src/
│   └── student_performance_prediction.py
├── .gitignore
├── README.md
└── requirements.txt
~~~

## Team

- Tanmay Raghav
- Aditya Sirohi
- Ritivik Sharma

**Submitted to:** Professor Nirbhay  
**Institution:** GLA University, Greater Noida

## Limitations

This is a classroom project based on 395 records and one hold-out split. Results may change with a different split or student population. Future improvements could include cross-validation, a separate experiment without G1 and G2, hyperparameter tuning and validation on data from other schools. Predictions should not be used alone to make important decisions about students.

## Reference

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
