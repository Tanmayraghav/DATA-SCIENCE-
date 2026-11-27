<p align="center">
  <img src="assets/project-banner.svg" alt="Student Performance Prediction project banner" width="100%">
</p>

<p align="center">
  <b>Machine Learning Project | B.Tech Computer Science &amp; Engineering</b><br>
  GLA University, Greater Noida · Academic Year 2026
</p>

<p align="center">
  <a href="reports/Student_Performance_Prediction_Project_Report.md"><img src="https://img.shields.io/badge/Read-Project%20Report-20B8A6?style=for-the-badge" alt="Read project report"></a>
  <a href="Student_Performance_Prediction.ipynb"><img src="https://img.shields.io/badge/Open-Notebook-183852?style=for-the-badge" alt="Open notebook"></a>
  <a href="https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/Student_Performance_Prediction.ipynb"><img src="https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=for-the-badge&amp;logo=googlecolab&amp;logoColor=white" alt="Run in Google Colab"></a>
</p>

# Student Performance Prediction

A small end-to-end machine learning project that explores student information and predicts the <strong>final Mathematics grade (G3)</strong> using Linear Regression and Random Forest. The project covers data inspection, exploratory data analysis (EDA), preprocessing, model training and evaluation.

## Project at a glance

| Dataset | Target | Models compared | Best test R-squared |
|:--|:--|:--|--:|
| 395 records · 33 columns | <code>G3</code> (0-20) | Linear Regression, Random Forest | <strong>0.811</strong> |

## Results

The following values are from the notebook's fixed <strong>80:20 train-test split</strong> (<code>random_state=42</code>). Lower MAE, MSE and RMSE are better; higher R-squared is better.

| Model | MAE | MSE | RMSE | R-squared |
|:--|--:|--:|--:|--:|
| <strong>Random Forest</strong> | <strong>1.173</strong> | <strong>3.868</strong> | <strong>1.967</strong> | <strong>0.811</strong> |
| Linear Regression | 1.647 | 5.657 | 2.378 | 0.724 |

> <strong>Result:</strong> Random Forest performed better on this particular test split. This does not guarantee that it will perform better on every dataset or school.

## Project workflow

1. <strong>Load and inspect</strong> the Mathematics dataset; check dimensions, data types, missing values and duplicates.
2. <strong>Explore the data</strong> with the final-grade distribution, study-time comparison, G2 vs G3 scatter plot, absences box plot and numerical correlation heatmap.
3. <strong>Preprocess</strong> by separating G3, making an 80:20 split, one-hot encoding categorical variables and aligning train/test features.
4. <strong>Scale</strong> the encoded features for Linear Regression using a scaler fitted only on the training data. Random Forest uses the encoded features without scaling.
5. <strong>Train and evaluate</strong> both regressors using MAE, MSE, RMSE and R-squared; inspect actual-versus-predicted values.

## Dataset

We use the Mathematics file, <code>student-mat.csv</code>, from the <a href="https://archive.ics.uci.edu/dataset/320/student+performance">UCI Student Performance dataset</a>.

- <strong>395 student records</strong> and <strong>33 attributes</strong>.
- <strong>Target:</strong> G3, the final Mathematics grade on a 0-20 scale.
- The columns include student, family and school details, study and social information, absences, and period grades.
- No missing values or duplicate rows were found in the saved notebook run.

### Important prediction note

G1 and G2 (the first- and second-period grades) are retained as input features. The model therefore estimates G3 when earlier grades are already available. Predicting final grades before earlier exam results are known would be a different experiment.

## Run the project

### 1. Install dependencies

~~~bash
python -m pip install -r requirements.txt
~~~

### 2. Run the Python script

~~~bash
python student_performance_prediction.py
~~~

The script reads <code>student-mat.csv</code> from the repository folder and saves graphs and result tables in a <code>results/</code> directory.

### 3. Run the notebook

Open <a href="Student_Performance_Prediction.ipynb">Student_Performance_Prediction.ipynb</a> in Jupyter, or <a href="https://colab.research.google.com/github/Tanmayraghav/DATA-SCIENCE-/blob/main/Student_Performance_Prediction.ipynb">open it directly in Google Colab</a>. Run the cells from top to bottom. The notebook can download the CSV from this repository when a local copy is not available.

## Repository contents

~~~text
DATA-SCIENCE- /
├── Student_Performance_Prediction.ipynb
├── student_performance_prediction.py
├── student-mat.csv
├── requirements.txt
├── assets/
│   └── project-banner.svg
└── reports/
    └── Student_Performance_Prediction_Project_Report.md
~~~

## Team

- <strong>Tanmay Raghav</strong>
- <strong>Aditya Sirohi</strong>
- <strong>Ritivik Sharma</strong>

<strong>Submitted to:</strong> Professor Nirbhay<br>
<strong>Institution:</strong> GLA University, Greater Noida

## Limitations and next steps

This is a classroom project based on 395 records and one hold-out split. Results may change with a different split or population. Future work could use cross-validation, test a version without G1 and G2, inspect residuals by grade range, tune model parameters and validate on data from other schools. Predictions should not be used alone for important decisions about students.

## References

1. Cortez, P. (2008). <em>Student Performance</em> [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5TG7T">https://doi.org/10.24432/C5TG7T</a>.
2. <a href="https://pandas.pydata.org/docs/">Pandas documentation</a> · <a href="https://scikit-learn.org/stable/">Scikit-learn documentation</a> · <a href="https://matplotlib.org/stable/">Matplotlib</a> · <a href="https://seaborn.pydata.org/">Seaborn</a>.
