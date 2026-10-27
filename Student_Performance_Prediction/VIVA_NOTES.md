# Viva Notes — Easy Explanation

## Project in two lines
This project uses student details and earlier exam grades to predict the final grade called `G3`. We explore the data, prepare it for machine learning, train Linear Regression, and check the predictions.

## Important terms
- **Dataset:** A table containing information about students.
- **EDA:** Exploring the data with summaries and graphs before building a model.
- **Histogram:** Shows how often different final grades occur.
- **Bar graph:** Compares the average final grade across study-time groups.
- **Scatter plot:** Shows the relationship between two values, such as `G2` and `G3`.
- **Correlation:** Shows how strongly two numeric columns are related. Correlation does not automatically mean one causes the other.
- **Preprocessing:** Preparing data so a machine-learning model can use it.
- **Outlier:** A value unusually far from most values. Here, we check unusually high/low `absences` with the IQR rule.
- **IQR:** Q3 minus Q1. Common outlier limits are Q1 − 1.5 × IQR and Q3 + 1.5 × IQR.
- **Encoding:** Turning text categories into numeric columns. We use one-hot encoding.
- **Min-Max scaling:** Changes values to a 0–1 range.
- **Standardization:** Changes numeric values to an average near 0 and standard deviation near 1.
- **Train-test split:** Uses one part of the data to train and another part to test the model.
- **Linear Regression:** A basic model that learns a relationship between input values and a numeric result.
- **MAE:** Average size of prediction errors. Lower is better.
- **RMSE:** Another error measure that gives larger mistakes more weight. Lower is better.
- **R²:** Measures how much of the variation in final grades the model explains. Closer to 1 is generally better, but interpret it with the error measures.

## Common viva questions

**1. Why did you choose this project?**  
It is a basic real-world machine-learning problem and lets us practise the full data-science process.

**2. What are X and y?**  
`X` contains the input columns. `y` contains the final grade `G3` that we want to predict.

**3. Why encode categorical columns?**  
Most machine-learning models need numeric inputs, so text categories must be converted into numbers.

**4. Why standardize numeric data?**  
It puts numeric columns on a more comparable scale. This can help some models learn more reliably.

**5. Why do we split the dataset?**  
To test the model on examples it did not use during training.

**6. Why use Linear Regression?**  
It is a simple beginner-friendly model for predicting a numeric value such as a grade.

**7. Is the prediction always correct?**  
No. The model gives an estimate. MAE and RMSE show the typical size of errors.

**8. What is one limitation?**  
The model uses `G1` and `G2`, so it assumes earlier-period grades are already known. It is not a prediction made before any exams.

**9. Why might we remove outliers?**  
Unusually extreme values can affect a model. We flag them with the IQR rule, but some unusual values may be genuine, so removal must be explained.

## Before presenting
After running the notebook, remember your actual MAE, RMSE, and R² results. Do not memorise made-up values; explain what your results mean.

## Dataset citation
Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
