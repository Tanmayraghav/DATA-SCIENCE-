# Student Performance Prediction
# Run this file cell-by-cell in Google Colab, or run it as a Python script.

# 1. Install the UCI dataset package
%pip -q install ucimlrepo

# 2. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 3. Download the UCI Student Performance dataset
student_data = fetch_ucirepo(id=320)

# 4. Combine student details and grades into one table
X_data = student_data.data.features
y_data = student_data.data.targets
df = pd.concat([X_data, y_data], axis=1)

# 5. Show the first five rows
display(df.head())

# 6. Show the number of rows and columns
print("Dataset shape:", df.shape)

# 7. Show all column names
print(df.columns.tolist())

# 8. Check column data types
print(df.dtypes)

# 9. Count missing values
print(df.isnull().sum())

# 10. Count duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# 11. Remove duplicate rows
df = df.drop_duplicates().copy()

# 12. Convert the target grade to numeric
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
df = df.dropna(subset=["G3"]).copy()

# 13. Graph 1: Final grade distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="G3", bins=15, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final grade (G3)")
plt.ylabel("Number of students")
plt.show()

# 14. Graph 2: Study time and average final grade
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="studytime", y="G3", errorbar=None)
plt.title("Study Time vs Average Final Grade")
plt.xlabel("Study time group")
plt.ylabel("Average final grade")
plt.show()

# 15. Graph 3: Second-period grade vs final grade
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="G2", y="G3")
plt.title("Second-Period Grade vs Final Grade")
plt.xlabel("Second-period grade (G2)")
plt.ylabel("Final grade (G3)")
plt.show()

# 16. Graph 4: Absences and possible outliers
plt.figure(figsize=(7, 3))
sns.boxplot(data=df, x="absences")
plt.title("Outlier Check for Absences")
plt.xlabel("Absences")
plt.show()

# 17. Graph 5: Correlation between numeric columns
numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 18. Show simple data analysis by study time
display(df.groupby("studytime")["G3"].agg(["count", "mean"]).rename(
    columns={"count": "number_of_students", "mean": "average_final_grade"}
))

# 19. Separate input columns (X) and target grade (y)
X = df.drop(columns=["G3"])
y = df["G3"]

# 20. Split data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 21. Find outlier limits using the training data only
q1 = X_train["absences"].quantile(0.25)
q3 = X_train["absences"].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
print("Lower outlier limit:", lower_limit)
print("Upper outlier limit:", upper_limit)

# 22. Remove outliers from the training data
keep_rows = X_train["absences"].between(lower_limit, upper_limit)
X_train = X_train.loc[keep_rows].copy()
y_train = y_train.loc[keep_rows].copy()
print("Training rows after outlier removal:", len(X_train))

# 23. Find text and numeric input columns
categorical_columns = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
print("Categorical columns:", categorical_columns)
print("Numeric columns:", numeric_columns)

# 24. Demonstrate Min-Max scaling on two numeric columns
minmax_scaler = MinMaxScaler()
example_columns = [column for column in ["age", "absences"] if column in X_train.columns]
if example_columns:
    scaled_example = minmax_scaler.fit_transform(X_train[example_columns])
    scaled_example_df = pd.DataFrame(
        scaled_example, columns=example_columns, index=X_train.index
    )
    display(scaled_example_df.head())

# 25. Set up encoding and standardization
preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", StandardScaler(), numeric_columns),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

# 26. Build the Linear Regression model
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

# 27. Train the model
model.fit(X_train, y_train)
print("Model training complete.")

# 28. Predict final grades for the test data
y_pred = model.predict(X_test)

# 29. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared (R2):", round(r2, 2))

# 30. Compare actual and predicted grades
comparison = pd.DataFrame({
    "Actual grade": y_test,
    "Predicted grade": y_pred.round(2)
})
display(comparison.head(10))

# 31. Graph: Actual grades vs predicted grades
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 20], [0, 20], linestyle="--")
plt.title("Actual vs Predicted Final Grades")
plt.xlabel("Actual grade")
plt.ylabel("Predicted grade")
plt.show()

# 32. Try a sample prediction
sample_student = X_test.iloc[[0]]
sample_actual_grade = y_test.iloc[0]
sample_predicted_grade = model.predict(sample_student)[0]
print("Actual final grade:", sample_actual_grade)
print("Predicted final grade:", round(sample_predicted_grade, 2))
