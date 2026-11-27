# Student Performance Prediction
## Machine Learning Project Report

**Submitted to:** Professor Nirbhay  
**Department:** Computer Science and Engineering  
**Institution:** GLA University, Greater Noida  
**Academic Year:** 2026

### Team Members

1. **TANMAY RAGHAV**
2. **ADITYA SIROHI**
3. **RITIVIK SHARMA**

## Abstract

This project uses machine learning to estimate students' final Mathematics grades using the UCI Student Performance dataset. The workflow includes data inspection, exploratory data analysis, preprocessing, an 80:20 train-test split, Linear Regression, Random Forest, and model evaluation. Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² are used to compare predictions. The script saves graphs and result tables in a results folder. The project is intended for academic learning and exploratory analysis, not high-stakes decisions about individual students.

## 1. Introduction

Student performance can be associated with previous grades, study time, attendance, and other demographic or school-related factors. Analysing historical data can help describe patterns in academic outcomes. This project demonstrates an introductory data science workflow and compares two regression models for predicting the final grade, G3.

## 2. Problem Statement

Build and evaluate machine learning models that estimate a student's final Mathematics grade from selected attributes, then compare predicted grades against actual grades in a held-out test set.

## 3. Objectives

- Inspect the dataset's dimensions, data types, missing values, and duplicates.
- Explore the distribution of final grades and relationships among selected variables.
- Encode categorical data and prepare numerical features for modelling.
- Train Linear Regression and Random Forest models.
- Compare the models using MAE, MSE, RMSE, and R².
- Discuss limitations and responsible use of predictions.

## 4. Dataset Description

This project uses the **Mathematics subset** (student-mat.csv) of the Student Performance dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance). The CSV contains **395 student records and 33 attributes** covering demographic, family, school, study-time, attendance, and grade-related information. The target variable is **G3**, the final grade on a 0–20 scale.

The code includes **G1** and **G2** (first- and second-period grades) as model inputs. This is suitable when earlier grades are known at prediction time. To predict final grades without prior grades, those columns should be excluded and that alternative task evaluated separately.

## 5. Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- UCI Student Performance dataset

## 6. Methodology

### 6.1 Data Loading and Inspection

The script loads student-mat.csv using pandas.read_csv with a semicolon delimiter. It prints the dataset shape, columns, data types, missing-value counts, and duplicate count, and removes duplicate rows if present.

### 6.2 Exploratory Data Analysis

The code produces:

- A histogram of G3 final grades.
- A bar plot comparing study-time categories with average final grade.
- A scatter plot of G2 versus G3.
- A box plot for absences.
- A correlation heatmap for numerical variables.

The average G3 for each study-time category is also printed. These plots show associations and do not establish causation. The absences box plot is used to inspect possible outliers; the script does not automatically delete them.

### 6.3 Preprocessing

The target G3 is separated from the features before the raw data is split into training and test subsets using an 80:20 split with random_state=42. Categorical columns are encoded using pandas.get_dummies, and test columns are aligned to the training columns.

StandardScaler is fitted on the training features only and then applied to both sets for Linear Regression. Random Forest uses the encoded features without scaling because tree-based models do not require standardised feature values.

### 6.4 Model Training

Two models are trained using the same training and test split:

- **Linear Regression:** A simple baseline regression model.
- **Random Forest Regressor:** An ensemble model using 200 trees and random_state=42.

### 6.5 Evaluation Metrics

- **MAE:** Average absolute prediction error; lower is better.
- **MSE:** Average squared prediction error; lower is better and larger errors receive more weight.
- **RMSE:** Square root of MSE, in target-grade units; lower is better.
- **R²:** Measures model performance relative to a mean-baseline model; higher is generally better, and it can be negative.

## 7. Results and Discussion

Run student_performance_prediction.py. The script prints the actual test-set metrics for both models and saves the comparison to results/model_comparison.csv.

Enter the values from that generated table below after running the script. Do not estimate or invent metric values.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | __________ | __________ | __________ | __________ |
| Random Forest | __________ | __________ | __________ | __________ |

The script also saves a model comparison graph and an actual-versus-predicted graph in the results folder. Lower MAE and RMSE indicate smaller errors. The results should be interpreted in the context of this dataset and this particular train-test split.

## 8. Limitations

- The dataset contains 395 Mathematics student records, so it is relatively small.
- The models may not generalise to other schools, years, or student populations.
- Linear Regression may not capture complex relationships.
- Results can vary with a different train-test split.
- Including G1 and G2 changes the task because previous grades are useful inputs.
- Historical association does not prove causation.
- Predictions should not be used alone to label students or make consequential decisions.

## 9. Future Scope

- Compare predictions when G1 and G2 are included versus excluded.
- Use cross-validation for a more stable estimate of model performance.
- Test the models on suitable external data before considering real-world use.
- Explore additional models only when there is a clear reason to do so.

## 10. Conclusion

This project demonstrates a basic machine learning workflow for student grade prediction, from data inspection and exploratory analysis to preprocessing, model training, and evaluation. Linear Regression provides a simple baseline, while Random Forest provides a second model for comparison. Final conclusions should be based on the actual test results generated by the script. The project highlights clear task definition, prevention of data leakage, and responsible use of student-related predictions.

## References

1. Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Future Business Technology Conference (FUBUTEC).
2. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
3. Pandas documentation: https://pandas.pydata.org/docs/
4. Scikit-learn documentation: https://scikit-learn.org/stable/
5. Matplotlib documentation: https://matplotlib.org/stable/
6. Seaborn documentation: https://seaborn.pydata.org/
