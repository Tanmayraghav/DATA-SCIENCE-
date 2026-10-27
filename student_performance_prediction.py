# Student Performance Prediction Project
# Run these cells in Google Colab, from top to bottom.

# Install the dataset package
%pip -q install ucimlrepo

# 1. Import basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load the dataset
student_data = fetch_ucirepo(id=320)
X_raw = student_data.data.features
y_raw = student_data.data.targets

# Put everything into one simple dataframe
df = pd.concat([X_raw, y_raw], axis=1)

# 3. Check the data
print("First 5 rows:")
display(df.head())

print("Rows and columns:", df.shape)
print("Column names:", df.columns.tolist())
print("Data types:\n", df.dtypes)

# 4. Check for missing values and duplicates
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

# 5. Make visualizations (graphs)

# Graph 1: Histogram of final grades (G3)
plt.hist(df["G3"], bins=15, edgecolor="black")
plt.title("Final Grades Distribution")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")
plt.show()

# Graph 2: Bar chart of study time vs final grade
sns.barplot(x="studytime", y="G3", data=df, errorbar=None)
plt.title("Study Time vs Average Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Average Final Grade")
plt.show()

# Graph 3: Scatter plot of G2 vs G3
plt.scatter(df["G2"], df["G3"])
plt.title("Second Period Grade (G2) vs Final Grade (G3)")
plt.xlabel("G2")
plt.ylabel("G3")
plt.show()

# Graph 4: Boxplot to check outliers in absences
sns.boxplot(x=df["absences"])
plt.title("Boxplot of Absences")
plt.show()

# Graph 5: Correlation heatmap (numeric columns only)
num_df = df.select_dtypes(include="number")
plt.figure(figsize=(10, 6))
sns.heatmap(num_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Simple grouping: average grade for each study time
print(df.groupby("studytime")["G3"].mean())

# 6. Separate features (X) and target (y)
X = df.drop("G3", axis=1)
y = df["G3"]

# 7. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Remove absences outliers from training data only
Q1 = X_train["absences"].quantile(0.25)
Q3 = X_train["absences"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

keep_rows = X_train["absences"].between(lower, upper)
X_train = X_train.loc[keep_rows].copy()
y_train = y_train.loc[keep_rows].copy()

print("Training rows after removing outliers:", X_train.shape[0])

# 9. Convert text columns to numbers (One-Hot Encoding)
# Make training and testing columns match
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 10. Feature Scaling
# Demonstrate Min-Max Scaling on age and absences
example_columns = ["age", "absences"]
minmax = MinMaxScaler()
print("Sample Min-Max scaled values (first 5 rows):")
print(minmax.fit_transform(X_train[example_columns])[:5])

# Standardize all features; fit only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 12. Make predictions
predictions = model.predict(X_test_scaled)

# 13. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# 14. Compare actual vs predicted grades
results = pd.DataFrame({"Actual": y_test, "Predicted": predictions.round(2)})
display(results.head(10))

# Graph: Actual vs Predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.show()
