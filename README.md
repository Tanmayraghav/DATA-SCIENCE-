# Student Performance Prediction

A small machine learning project that looks at student information and estimates the final Mathematics grade (G3). It was made as a B.Tech CSE project to practise data analysis and basic machine learning.

## Team

- Tanmay Raghav
- Aditya Sirohi
- Ritivik Sharma

**Submitted to:** Professor Nirbhay  
**Academic year:** 2026

## Dataset

We used the Mathematics file (`student-mat.csv`) from the UCI Student Performance dataset.

- 395 student records
- 33 columns
- Target column: `G3` (final Mathematics grade, from 0 to 20)

Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

Citation: Cortez, P. (2008). *Student Performance* [Dataset]. https://doi.org/10.24432/C5TG7T.

## Files in this repository

| File | What it contains |
|---|---|
| `Student_Performance_Prediction.ipynb` | Notebook with the project steps and code |
| `student_performance_prediction.py` | Python version of the project |
| `student-mat.csv` | Mathematics dataset |
| `requirements.txt` | Libraries needed to run the code |
| `reports/Student_Performance_Prediction_Project_Report.md` | Project report |

## What we did

1. Loaded the dataset and checked its rows, columns, data types, missing values and duplicate records.
2. Made graphs to understand the final-grade distribution, study-time groups, G2 and G3, absences and numerical correlations.
3. Split the data into training and testing parts (80:20).
4. Converted categorical columns into numeric columns.
5. Scaled the features for Linear Regression.
6. Trained Linear Regression and Random Forest.
7. Compared their results using MAE, MSE, RMSE and R².

We kept G1 and G2 as input columns because the project assumes that earlier-period grades are known before estimating G3.

## Results from the notebook run

The following values are from the notebook's 80:20 test split (random_state=42). Your results may change if the data or split changes.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 1.173 | 3.868 | 1.967 | 0.811 |
| Linear Regression | 1.647 | 5.657 | 2.378 | 0.724 |

Random Forest had the lower error on this test split. This does not mean it will always perform better on every dataset.

## Run the project

Install the libraries:

```bash
python -m pip install -r requirements.txt
```

Run the Python file:

```bash
python student_performance_prediction.py
```

For the notebook, open `Student_Performance_Prediction.ipynb` in Jupyter or Google Colab and run the cells from top to bottom. The notebook downloads the CSV from this repository if the file is not available locally.

The Python script saves graphs and result tables in a `results` folder.

## A note about the prediction

G1 and G2 are earlier grades, while G3 is the final grade. Since G1 and G2 are included as inputs, this project estimates G3 when those earlier grades are already available. It is a different problem from predicting a student's final grade before any exam marks are known.

This is a classroom project for learning. Its predictions should not be used on their own to make important decisions about students.
