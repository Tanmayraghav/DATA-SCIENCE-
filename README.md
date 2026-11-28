<p align="center"><img src="assets/project-banner.svg" alt="Student Performance Prediction" width="100%"></p>

<p align="center"><b>Machine Learning Project | B.Tech CSE</b><br>GLA University, Greater Noida · 2026</p>

<p align="center"><a href="reports/Student_Performance_Prediction_Project_Report.pdf">Project Report (PDF)</a> · <a href="notebooks/Student_Performance_Prediction.ipynb">Jupyter Notebook</a> · <a href="https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/notebooks/Student_Performance_Prediction.ipynb">Open in Google Colab</a></p>

# Student Performance Prediction

A beginner-friendly machine-learning project that estimates the final Mathematics grade (G3) using Linear Regression and Random Forest.

## Project overview

| Dataset | Target | Models | Best test R² |

|---|---|---|---:|

| 395 records · 33 attributes | G3 (0–20) | Linear Regression, Random Forest | **0.811** |

## Model results

Results from one 80:20 train-test split (random_state=42). Lower MAE, MSE and RMSE are better; higher R² is better.

| Model | MAE | MSE | RMSE | R² |

|---|---:|---:|---:|---:|

| **Random Forest** | **1.173** | **3.868** | **1.967** | **0.811** |

| Linear Regression | 1.647 | 5.657 | 2.378 | 0.724 |

Random Forest performed better on this particular test split; this is not a guarantee for other splits or populations.

## Workflow

1. Load and inspect the Mathematics dataset.

2. Explore grade distribution, study time, G2 vs G3, absences and numerical correlations.

3. Split into training and test sets (80:20).

4. Encode categorical features and scale Linear Regression inputs using a scaler fitted only on training data.

5. Train and compare both models using MAE, MSE, RMSE and R².

6. Display graphs, metrics and sample predictions directly; no output files are saved.

## Dataset

Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance). It contains 395 Mathematics records and 33 attributes. The target is G3, the final grade on a 0–20 scale.

**Important:** G1 and G2 are input features. The model estimates G3 when earlier-period grades are already available; it is not an early prediction before those grades are known.

## Run the project

### Google Colab

Open the [notebook in Google Colab](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/notebooks/Student_Performance_Prediction.ipynb) and select Runtime → Run all. The notebook downloads the CSV from this repository when no local copy is found.

### Run locally

Install dependencies with: `python -m pip install -r requirements.txt`

Run from the repository root: `python src/student_performance_prediction.py`

The script reads data/student-mat.csv and displays graphs without saving them.

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

## Limitations and future work

This project uses 395 records and one hold-out split; no separate validation set or cross-validation was used. Future work could add cross-validation, hyperparameter tuning and an experiment excluding G1 and G2. Predictions should not be used alone for important student decisions.

## Reference

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).