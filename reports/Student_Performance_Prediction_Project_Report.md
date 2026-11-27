# Student Performance Prediction
## Project Report

**Submitted to:** Professor Nirbhay  
**Department:** Computer Science and Engineering  
**Institution:** GLA University, Greater Noida  
**Academic year:** 2026

### Team members

1. Tanmay Raghav
2. Aditya Sirohi
3. Ritivik Sharma

## Abstract

In this project, we used student data to estimate the final Mathematics grade (G3). We first checked the dataset and made a few graphs to understand it. After preparing the data, we trained two regression models: Linear Regression and Random Forest. We compared their predictions using MAE, MSE, RMSE and R². The Random Forest model had a lower error on the test data used in our notebook. This project helped us practise the basic steps involved in a machine learning project.

## 1. Introduction

A student's final marks can be related to several things, such as previous marks, study time, attendance and family or school background. Looking at this information can help us understand patterns in the data.

The aim of our project is to use the available student information to estimate the final Mathematics grade. We have kept the work focused on basic data analysis and two machine learning models.

## 2. Problem statement

To build a regression model that estimates a student's final Mathematics grade (G3) and compare its predictions with the actual grades.

## 3. Objectives

- Understand the dataset and check its quality.
- Use graphs to explore some of the columns.
- Prepare categorical and numerical data for machine learning.
- Train Linear Regression and Random Forest models.
- Compare the models using common regression metrics.
- Explain the results and limitations of the project.

## 4. Dataset

We used the Mathematics subset of the Student Performance dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

The file used is `student-mat.csv`. It contains 395 records and 33 attributes. The columns include information about students, their family and school, study time, absences and grades.

The column we want to predict is **G3**, which is the final Mathematics grade. Its scale is 0 to 20.

We have also used **G1** and **G2** as input features. These columns contain the first- and second-period grades. This means our model estimates the final grade when the earlier grades are already available.

## 5. Software and libraries

- Python
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 6. Steps followed

### 6.1 Loading and checking the data

We loaded the CSV file using Pandas. The file uses a semicolon as the separator. We checked the shape, column names, data types, missing values and duplicate rows.

In the notebook run, the dataset had 395 rows and 33 columns. No missing values or duplicate rows were found.

### 6.2 Exploratory data analysis

We made the following graphs:

1. **Final grade distribution:** to see how the G3 marks are spread.
2. **Study time and final grade:** to compare the average G3 for different study-time categories.
3. **G2 and G3:** to see how the second-period marks are related to final marks.
4. **Absences box plot:** to inspect the distribution of student absences.
5. **Correlation heatmap:** to view correlations between numerical columns.

We also calculated the average G3 for each study-time category. These graphs help describe the data, but they do not prove that one factor causes a change in marks.

### 6.3 Data preprocessing

First, we separated G3 from the other columns. Then we divided the data into training and testing sets in an 80:20 ratio. We used `random_state=42` so the same split can be reproduced.

The categorical columns were converted into numeric columns using one-hot encoding. We also aligned the test columns with the training columns so both sets had the same features.

For Linear Regression, we used StandardScaler. The scaler was fitted on the training data and then applied to the test data. Random Forest was trained on the encoded data without scaling.

### 6.4 Model training

We trained two models:

**Linear Regression**

This model is used as a basic starting point for predicting a continuous value. It tries to fit a linear relationship between the input features and the final grade.

**Random Forest Regressor**

This model uses multiple decision trees and combines their predictions. In our code, we used 200 trees and `random_state=42`.

Both models used the same training and test split.

## 7. Model evaluation

We used four metrics to compare the predictions:

- **MAE:** the average absolute difference between actual and predicted marks. A lower value is better.
- **MSE:** the average of squared errors. A lower value is better.
- **RMSE:** the square root of MSE. It is expressed in the same units as the grade.
- **R²:** shows how well the model performs compared with predicting the average grade. A higher value is generally better.

### 7.1 Results

These are the values produced by the executed notebook:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 1.173418 | 3.867641 | 1.966632 | 0.811381 |
| Linear Regression | 1.646666 | 5.656643 | 2.378370 | 0.724134 |

Random Forest gave a lower MAE, MSE and RMSE, and a higher R² than Linear Regression on this test split. So, based on this particular run, Random Forest performed better.

The result is based on one 80:20 split. It should not be treated as a guarantee that Random Forest will always be better.

### 7.2 Actual and predicted values

The notebook also displays a small table of actual and predicted marks for ten test records. It creates an actual-versus-predicted graph for both models. These help us see how close the predictions are to the actual grades.

## 8. Limitations

- The dataset has 395 Mathematics records, so it is not very large.
- The data comes from a particular study and may not represent every school or student group.
- The result can change with a different train-test split.
- G1 and G2 are included as input features. A model that does not use earlier grades would be a different experiment.
- The graphs show patterns in the dataset, not proof of cause and effect.
- The model's output is an estimate and may not be correct for an individual student.

## 9. Future improvements

If we continue this project, we can try cross-validation to check whether the results stay similar across different splits. We can also compare a version that does not use G1 and G2, which would answer a different prediction question.

## 10. Conclusion

We completed the main steps of a basic machine learning project: loading data, checking it, exploring it with graphs, preprocessing, training models and comparing their results.

In our notebook run, Random Forest performed better than Linear Regression on the selected test set. The project gave us practical experience with Pandas, data visualisation and Scikit-learn. It also showed us why the input columns and the way we test a model matter when explaining its results.

## References

1. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
2. Pandas documentation: https://pandas.pydata.org/docs/
3. Scikit-learn documentation: https://scikit-learn.org/stable/
4. Matplotlib documentation: https://matplotlib.org/stable/
5. Seaborn documentation: https://seaborn.pydata.org/
