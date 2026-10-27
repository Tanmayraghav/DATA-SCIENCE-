# Student Performance Prediction — Project Explanations

**A simple Data Science project using Python and Google Colab**

**Goal:** Use student information and earlier grades to predict the final grade (`G3`).

The notebook is split into small tasks. Run each cell from top to bottom using the play button beside the cell.

Dataset: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

## Part 1 — Setup and understanding the data

### Step 1 — Install the dataset package
Google Colab already has most common data-science libraries. We only need the `ucimlrepo` package to download the dataset from UCI.

### Step 2 — Import libraries
- `pandas` helps us work with table-shaped data.
- `matplotlib` and `seaborn` make graphs.
- `scikit-learn` provides scaling, the model, and evaluation tools.

### Step 3 — Download the dataset
This downloads Student Performance dataset number 320 from the UCI Machine Learning Repository.

### Step 4 — Combine the data into one table
We combine the student information and grade columns into one table called `df`.

### Step 5 — Look at the first five rows
This gives us a quick look at what the dataset contains.

### Step 6 — Check the column names
We use `G3` as the final grade that we want to predict.

### Step 7 — Check data types
This tells us which columns contain numbers and which contain text.

### Step 8 — Check for missing values
A missing value means that a piece of information is not recorded.

### Step 9 — Check for duplicate rows
A duplicate is a row that appears more than once.

### Step 10 — Remove duplicate rows
We remove exact duplicate rows. If there are no duplicates, the data stays the same.

## Part 2 — Exploratory Data Analysis (EDA)

EDA means understanding the data with simple tables and graphs before building a model.

### Step 11 — Graph 1: Distribution of final grades
The histogram shows how often different final grades appear.

### Step 12 — Graph 2: Study time and average final grade
`studytime` is a group code for weekly study time. This graph compares the average final grade in each group.

### Step 13 — Graph 3: Earlier grade vs final grade
`G2` is the second-period grade, and `G3` is the final grade. The graph helps us see how they are related.

### Step 14 — Graph 4: Outliers in absences
A box plot helps us spot unusually high or low values. Here we inspect the number of school absences.

### Step 15 — Graph 5: Correlation heatmap
Correlation shows how number-based columns move together. Values closer to `1` mean a stronger positive relationship; values closer to `-1` mean a stronger negative relationship.

### Step 16 — Simple data analysis table
This table shows how many students are in each study-time group and their average final grade. Use the actual values to describe what you notice.

## Part 3 — Data preprocessing

For the model, `X` means the information used to make a prediction. `y` is the answer we want the model to predict.

### Step 17 — Separate inputs and output
We remove `G3` from the input data because it is the grade we want to predict.

### Step 18 — Split into training and testing data
We use 80% of the rows to train the model and keep 20% aside to test it on data it has not trained on.

### Step 19 — Detect outliers using IQR
We check the `absences` column in the training data. IQR means the third-quarter value minus the first-quarter value.

### Step 20 — Remove the detected training outliers
We remove these rows only from the training data. We leave the test data unchanged so the test remains a fair check of the model.

### Step 21 — Find categorical (text) columns
Columns like `school`, `sex`, or `address` contain categories instead of measurements.

### Step 22 — Encode categorical columns
Machine-learning models need numerical input. One-hot encoding changes text categories into columns containing 0 and 1. We also make sure the test data has exactly the same columns as the training data.

### Step 23 — Demonstrate Min-Max scaling
Min-Max scaling changes numeric values to a range from 0 to 1. This separate example demonstrates the technique on `age` and `absences`. The model uses standardization in the next step.

### Step 24 — Standardize numeric columns
Standardization changes numeric columns so their values are centered around 0 with a standard deviation near 1. We fit the scaler using training data only, then apply it to both training and test data.

## Part 4 — Model building

We use **Linear Regression**, a basic model that predicts a number. Here, the predicted number is the student's final grade.

### Step 25 — Create and train the model
The model learns patterns from the training data.

### Step 26 — Make predictions on the test data
The model predicts final grades for students in the test set.

### Step 27 — Evaluate the model
- **MAE:** average size of the prediction error; lower is better.
- **RMSE:** gives more weight to large errors; lower is better.
- **R²:** how well the model fits the test results; closer to 1 is generally better.

### Step 28 — Compare actual and predicted grades
This table lets us compare some real grades with the model's predictions.

### Step 29 — Graph: actual grade vs predicted grade
Points closer to the dashed line are predictions closer to the actual grade.

### Step 30 — Try one example prediction
This takes one student from the test set and compares the predicted final grade with the actual grade.

## Conclusion

After running the notebook, write a short conclusion using your own actual results:

1. We explored the student dataset using tables and graphs.
2. We prepared the data by removing duplicate rows, checking outliers, encoding categories, and scaling numeric values.
3. We trained a Linear Regression model to predict `G3`.
4. We evaluated the model using MAE, RMSE, and R². Include the values printed in Step 27 in your report.

**Dataset source:** Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

**Limitation:** The model uses `G1` and `G2`, the earlier-period grades, as input features. This can help predict `G3`, but it means the model predicts the final grade after earlier grades are already known.
