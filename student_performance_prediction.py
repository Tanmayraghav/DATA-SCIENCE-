"""
Student Performance Prediction
Predicts final Mathematics grade (G3) using the UCI Student Performance dataset.
Run: python student_performance_prediction.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "student-mat.csv"


def main():
    # Load the Mathematics dataset (CSV uses semicolons as separators).
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep student-mat.csv beside this script."
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    # Inspect data quality.
    print("Dataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Exploratory data analysis.
    plt.figure(figsize=(8, 5))
    plt.hist(df["G3"], bins=15, edgecolor="black")
    plt.title("Final Grade (G3) Distribution")
    plt.xlabel("Final grade")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="studytime", y="G3", errorbar=None)
    plt.title("Study Time vs Final Grade")
    plt.xlabel("Weekly study-time category")
    plt.ylabel("Average final grade (G3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["G2"], df["G3"], alpha=0.7)
    plt.xlabel("Second-period grade (G2)")
    plt.ylabel("Final grade (G3)")
    plt.title("Second-period Grade vs Final Grade")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["absences"])
    plt.title("Absences: Box Plot")
    plt.tight_layout()
    plt.show()

    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numerical Variables)")
    plt.tight_layout()
    plt.show()

    print("\nAverage final grade by study time:")
    print(df.groupby("studytime")["G3"].mean())

    # Separate target and features before splitting.
    # G1 and G2 are included because earlier grades are available as inputs.
    X = df.drop(columns=["G3"])
    y = df["G3"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encode categorical columns after the split and align test columns to train.
    X_train = pd.get_dummies(X_train_raw, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Fit the scaler on training data only to avoid test-data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate Linear Regression.
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    print("\nModel Evaluation")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))
    print("R2 score:", r2_score(y_test, y_pred))

    results = pd.DataFrame(
        {"Actual": y_test.to_numpy(), "Predicted": y_pred},
        index=y_test.index,
    )
    print("\nActual vs predicted grades (first 10 test rows):")
    print(results.head(10))

    # Compare actual and predicted grades with an ideal y = x reference line.
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lower = min(float(y_test.min()), float(np.min(y_pred)))
    upper = max(float(y_test.max()), float(np.max(y_pred)))
    plt.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    plt.xlabel("Actual final grade (G3)")
    plt.ylabel("Predicted final grade (G3)")
    plt.title("Actual vs Predicted Final Grades")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
