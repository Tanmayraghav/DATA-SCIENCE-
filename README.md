# Student Performance Prediction

<p align="center"><img src="05_Assets/project-banner.svg" alt="Student Performance Prediction" width="100%"></p>

<p align="center"><b>Machine Learning Project | B.Tech CSE</b><br>GLA University, Greater Noida · 2026</p>

A regression-based machine-learning project using the UCI Student Performance Mathematics dataset to estimate final grade (G3).

**Team:** Tanmay Raghav · Aditya Sirohi · Ritivik Sharma  
**Submitted to:** Professor Nirbhay, GLA University

## Project files

| Resource | Open |
|---|---|
| Main Python code | [student_performance_prediction.py](01_Code/student_performance_prediction.py) |
| Project report | [Student_Performance_Prediction_Project_Report.md](02_Report/Student_Performance_Prediction_Project_Report.md) |
| Dataset | [student-mat.csv](03_Dataset/student-mat.csv) |
| Executed notebook | [Student_Performance_Prediction.ipynb](04_Notebook/Student_Performance_Prediction.ipynb) |
| Open notebook in Colab | [Run in Google Colab](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/04_Notebook/Student_Performance_Prediction.ipynb) |

## Dataset

The Mathematics file contains **395 records and 33 columns**. The target is `G3`, final grade on a 0–20 scale. The included CSV has no missing values.

Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

## Prediction settings

1. **With G1/G2:** predicts G3 when earlier-period grades are available at prediction time.
2. **Without G1/G2:** predicts G3 using the remaining student attributes.

These are different prediction tasks. Earlier-period grades are not automatically data leakage when they are known at prediction time.

## Actual saved hold-out results

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

The default Random Forest has the best hold-out RMSE in both settings in the saved run. The tuned model had marginally lower cross-validation RMSE but slightly worse held-out test metrics. Full metrics, CV results, and interpretation are in the [project report](02_Report/Student_Performance_Prediction_Project_Report.md).

## Run the project

**Google Colab:** Open the [notebook](https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/04_Notebook/Student_Performance_Prediction.ipynb) and choose **Runtime → Run all**. The notebook downloads the CSV if it is not present in the Colab working directory.

**Local machine:**
```bash
git clone https://github.com/Tanmayraghav/DATA-SCIENCE-.git
cd DATA-SCIENCE-
python -m pip install -r requirements.txt
python 01_Code/student_performance_prediction.py
```

## Repository structure

```text
DATA-SCIENCE-/
├── 01_Code/
│   └── student_performance_prediction.py
├── 02_Report/
│   └── Student_Performance_Prediction_Project_Report.md
├── 03_Dataset/
│   └── student-mat.csv
├── 04_Notebook/
│   └── Student_Performance_Prediction.ipynb
├── 05_Assets/
│   └── project-banner.svg
├── .gitignore
├── README.md
└── requirements.txt
```

## Limitations

- Dataset has 395 Mathematics records; external validation is not performed.
- Cross-validation uses only the training split.
- GridSearchCV tests a small predefined parameter grid.
- Feature importance is not causal evidence.
- Predictions should not be used alone for important individual decisions.

## Reference

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
