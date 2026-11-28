<p align="center"><img src="05_Assets/project-banner.svg" alt="Student Performance Prediction" width="100%"></p>

<p align="center"><b>Machine Learning Project | B.Tech CSE</b><br>GLA University, Greater Noida · 2026</p>

# Student Performance Prediction

A regression-based machine-learning project that explores student attributes and predicts the final Mathematics grade (G3) using the UCI Student Performance dataset.

**Team:** Tanmay Raghav · Aditya Sirohi · Ritivik Sharma  
**Submitted to:** Professor Nirbhay, GLA University

## Project files

| Resource | Open |
|---|---|
| Main Python code | [student_performance_prediction.py](01_Code/student_performance_prediction.py) |
| Project report (PDF) | [Student_Performance_Prediction_Project_Report.pdf](02_Report/Student_Performance_Prediction_Project_Report.pdf) |
| Dataset | [student-mat.csv](03_Dataset/student-mat.csv) |
| Executed notebook | [Student_Performance_Prediction.ipynb](04_Notebook/Student_Performance_Prediction.ipynb) |
| Run notebook in Colab | [Open in Google Colab](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/04_Notebook/Student_Performance_Prediction.ipynb) |

## Dataset

The project uses the Mathematics (student-mat.csv) file from the UCI Student Performance dataset. It contains **395 student records and 33 columns**, with G3 as the final-grade target on a 0–20 scale. The included dataset has no missing values.

Dataset source: [UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)  
Reference: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)

## Prediction settings

The experiment compares two different use cases:

1. **With G1 and G2:** predicts the final grade when first- and second-period grades are already available.
2. **Without G1 and G2:** predicts the final grade from the remaining attributes when earlier grades are unavailable.

G1 and G2 are earlier-period grades. Their inclusion is not automatically data leakage when they are known at prediction time. The two settings represent different prediction tasks.

## Methodology

- One fixed 80:20 train/test split using random_state=42.
- Mean baseline, Linear Regression and Random Forest comparisons.
- Five-fold cross-validation performed on the training split.
- GridSearchCV for Random Forest hyperparameter selection.
- Pipeline and ColumnTransformer for fold-safe imputation, scaling and one-hot encoding.
- Hold-out metrics: MAE, MSE, RMSE and R².
- Exploratory charts, actual-vs-predicted plots, residual plots and Random Forest feature importance.

## Results from the executed notebook

The values below are from the saved notebook execution. MAE and RMSE are in grade points; higher R² is better.

| Prediction setting | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| With G1/G2 | Mean baseline | 3.646 | 4.550 | -0.010 |
| With G1/G2 | Linear Regression | 1.647 | 2.378 | 0.724 |
| With G1/G2 | Random Forest (default) | **1.188** | **2.007** | **0.804** |
| With G1/G2 | Random Forest (tuned) | 1.195 | 2.019 | 0.801 |
| Without G1/G2 | Mean baseline | 3.646 | 4.550 | -0.010 |
| Without G1/G2 | Linear Regression | 3.395 | 4.196 | 0.141 |
| Without G1/G2 | Random Forest (default) | **2.994** | **3.748** | **0.315** |
| Without G1/G2 | Random Forest (tuned) | 3.052 | 3.822 | 0.287 |

The default Random Forest achieved the lowest hold-out RMSE in both settings. The tuned Random Forest had slightly lower mean cross-validation RMSE on the training split (1.386 with G1/G2; 3.875 without G1/G2), but that did not translate into better hold-out scores in this run. Cross-validation and final test results are both reported.

When G1 and G2 were excluded, Random Forest hold-out R² fell from 0.804 to 0.315. Earlier grades provide strong predictive information for this task. Feature importance is descriptive and must not be interpreted as causal evidence.

## G3 = 0 records

There are **38 records** with G3 = 0 (about 9.6% of the dataset). All 38 also have absences = 0 in the included CSV. The dataset does not explain why their grades are zero, so no cause is assumed. The main experiment retains these records; positive-grade test-record metrics are shown separately in the report and notebook.

## Run the project

### Google Colab

Open the [notebook in Colab](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/04_Notebook/Student_Performance_Prediction.ipynb), then choose **Runtime → Run all**. The notebook downloads the CSV from this repository if student-mat.csv is not present in the Colab working directory.

### Local machine

    git clone https://github.com/Tanmayraghav/DATA-SCIENCE-.git
    cd DATA-SCIENCE-
    python -m pip install -r requirements.txt
    python 01_Code/student_performance_prediction.py

Python 3.10 or later is recommended. The script reads the CSV from 03_Dataset/ and displays plots and evaluation tables.

## Repository structure

    DATA-SCIENCE-/
    ├── 01_Code/
    │   └── student_performance_prediction.py
    ├── 02_Report/
    │   └── Student_Performance_Prediction_Project_Report.pdf
    ├── 03_Dataset/
    │   └── student-mat.csv
    ├── 04_Notebook/
    │   └── Student_Performance_Prediction.ipynb
    ├── 05_Assets/
    │   └── project-banner.svg
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## Limitations

- The dataset contains 395 records from one subject and one source; external validation is not performed.
- The train/test split is a single hold-out sample. Cross-validation is performed only within the training portion.
- The hyperparameter search covers a small, predefined parameter grid.
- The model estimates grades from historical data; it does not establish that changing a feature will change a student's grade.
- Predictions should not be used alone to make high-impact decisions about individual students.
