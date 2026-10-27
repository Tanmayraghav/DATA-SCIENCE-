# Student Performance Prediction — Report Outline

## 1. Introduction
This project uses student information to estimate the final course grade (`G3`) with Python and machine learning.

## 2. Dataset
- **Name:** UCI Student Performance Dataset
- **Source:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Target column:** `G3` (final grade)
- **Important input columns:** study time, absences, and earlier grades (`G1`, `G2`).

## 3. Objectives
- Understand the dataset with EDA and graphs.
- Check data quality and identify possible outliers.
- Encode categorical data and scale numeric data.
- Train a Linear Regression model.
- Evaluate predictions using MAE, RMSE, and R².

## 4. Methodology
1. Load the UCI data using Python.
2. Examine rows, columns, data types, missing values, and duplicates.
3. Create a histogram, bar graph, scatter plot, box plot, and correlation heatmap.
4. Remove duplicates and split the data into training and testing sets.
5. Detect and remove IQR-based outliers in the training data for `absences`.
6. One-hot encode categories and standardize numeric columns.
7. Train Linear Regression and evaluate it on the test set.

## 5. Results
Fill this section using the output printed by the notebook:
- **MAE:** __________
- **RMSE:** __________
- **R² score:** __________

Lower MAE/RMSE indicate smaller errors. An R² score closer to 1 generally indicates a better fit. Explain these values in your own words.

## 6. Conclusion
Write 3–5 sentences about what you learned, what the graphs showed, and how well the model performed based on the actual results.

## 7. Limitation
The model uses `G1` and `G2`, which are earlier-period grades. It predicts `G3` when earlier grades are available; it does not predict a student's result before any grades have been received.

## 8. Dataset citation
Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
