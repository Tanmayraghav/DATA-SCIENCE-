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
This project uses machine learning to estimate students’ final Mathematics grades using the UCI Student Performance dataset. The workflow includes data inspection, exploratory data analysis, preprocessing, a train-test split, feature scaling, Linear Regression, and evaluation. Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² are used to assess predictions. Actual metric values should be recorded after the code is executed; they are not invented in this report. The model is intended for learning and exploratory analysis, not high-stakes decisions about individual students.

## 1. Introduction
Student performance can be associated with previous grades, study time, attendance, and other demographic or school-related factors. Analysing historical data can help describe patterns in academic outcomes. This project demonstrates an introductory data science workflow and uses Linear Regression to predict the final grade, G3.

## 2. Problem Statement
Build and evaluate a machine learning model that estimates a student’s final Mathematics grade from selected attributes, then compare predicted grades against actual grades in a held-out test set.

## 3. Objectives
- Inspect the dataset's dimensions, data types, missing values, and duplicates.
- Explore the distribution of final grades and relationships among selected variables.
- Encode categorical data and scale numerical features for modelling.
- Train a Linear Regression model.
- Evaluate the model using MAE, MSE, RMSE, and R².
- Discuss limitations and responsible use of predictions.

## 4. Dataset Description
This project uses the **Mathematics subset** (student-mat.csv) of the Student Performance dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance). The CSV contains **395 student records and 33 attributes** covering demographic, family, school, study-time, attendance, and grade-related information. The target variable is **G3**, the final grade on a 0–20 scale.

The code currently includes **G1** and **G2** (first- and second-period grades) as model inputs. This is suitable when earlier grades are known at prediction time. To predict final grades without prior grades, those columns should be excluded and that alternative model evaluated separately.

## 5. Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- UCI Student Performance dataset

## 6. Methodology

### 6.1 Data Loading and Inspection
The script loads student-mat.csv with pandas.read_csv using a semicolon delimiter, prints the dataset shape, columns, data types, missing-value counts, and duplicate count, and removes duplicate rows if present.

### 6.2 Exploratory Data Analysis
The code produces a histogram of G3, a bar plot comparing study-time categories with final grade, a scatter plot of G2 versus G3, a box plot for absences, and a correlation heatmap for numerical variables. It also prints the average G3 for each study-time category. These plots show associations and do not establish causation.

### 6.3 Preprocessing
The target G3 is separated from the features before the raw data is split into training and test subsets using an 80:20 split with random_state=42. Categorical columns are encoded separately in both subsets using pandas.get_dummies; test columns are aligned to the training columns. StandardScaler is fitted on training features only and then applied to both sets to reduce data leakage.

### 6.4 Model Training
A Linear Regression model is trained on the scaled training data and used to predict G3 for the held-out test data.

### 6.5 Evaluation Metrics
- **MAE:** average absolute prediction error; lower is better.
- **MSE:** average squared prediction error; lower is better and larger errors receive more weight.
- **RMSE:** square root of MSE, in target-grade units; lower is better.
- **R²:** measures model performance relative to a mean-baseline model; higher is generally better, and it can be negative.

## 7. Results and Discussion
Run student_performance_prediction.py and enter the actual values printed by the evaluation section.

| Metric | Observed result |
|---|---|
| MAE | __________________ |
| MSE | __________________ |
| RMSE | __________________ |
| R² score | __________________ |

The script also prints a sample of actual and predicted grades and produces a scatter plot. Points closer to the ideal y = x line indicate better agreement. Metrics and the plot should be interpreted together.

## 8. Limitations
- Linear Regression may not capture complex relationships.
- Performance depends on the selected features and how representative the dataset is.
- Including G1 and G2 changes the task because previous grades are useful inputs.
- The model may not generalise to other schools, years, or student populations.
- Historical association does not prove causation.
- Predictions should not be used alone to label students or make consequential decisions.

## 9. Future Scope
- Compare Linear Regression with Ridge Regression, Decision Tree, and Random Forest models.
- Use cross-validation for more stable evaluation.
- Compare models with and without G1 and G2.
- Test the model on suitable external data before considering real-world use.

## 10. Conclusion
This project demonstrates a basic machine learning workflow for student grade prediction, from data inspection and exploratory analysis to preprocessing, training, and evaluation. Linear Regression provides a baseline assessed using MAE, MSE, RMSE, and R². Final conclusions should be based on the actual results from the executed script. The project highlights clear task definition, prevention of data leakage, and responsible use of student-related predictions.

## References
1. Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Future Business Technology Conference (FUBUTEC).
2. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
3. Pandas documentation: https://pandas.pydata.org/docs/
4. Scikit-learn documentation: https://scikit-learn.org/stable/
5. Matplotlib documentation: https://matplotlib.org/stable/
6. Seaborn documentation: https://seaborn.pydata.org/
