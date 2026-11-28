"""
Student Performance Prediction
Predicts final Mathematics grade (G3) using the UCI Student Performance dataset.

Run from the repository root:
    python 1_Code/student_performance_prediction.py

All graphs are displayed directly. No graph or result files are saved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "3_Dataset" / "student-mat.csv"


def show_plot():
    """Display the current graph without saving it."""
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    # Load the Mathematics dataset. The CSV uses semicolons as separators.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep student-mat.csv in the 3_Dataset/ folder."
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    # Check data quality.
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Exploratory data analysis.
    plt.figure(figsize=(8, 5))
    plt.hist(df["G3"], bins=15, edgecolor="black")
    plt.title("Final Grade (G3) Distribution")
    plt.xlabel("Final grade")
    plt.ylabel("Number of students")
    show_plot()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="studytime", y="G3", errorbar=None)
    plt.title("Study Time vs Final Grade")
    plt.xlabel("Weekly study-time category")
    plt.ylabel("Average final grade (G3)")
    show_plot()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["G2"], df["G3"], alpha=0.7)
    plt.xlabel("Second-period grade (G2)")
    plt.ylabel("Final grade (G3)")
    plt.title("Second-period Grade vs Final Grade")
    show_plot()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["absences"])
    plt.title("Absences: Box Plot")
    show_plot()

    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    show_plot()

    print("\nAverage final grade by study time:")
    print(df.groupby("studytime")["G3"].mean())

    # G1 and G2 are included because earlier grades are assumed to be known.
    X = df.drop(columns=["G3"])
    y = df["G3"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert categorical columns into numeric columns.
    X_train = pd.get_dummies(X_train_raw, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale features for Linear Regression only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": (
            LinearRegression(),
            X_train_scaled,
            X_test_scaled,
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, random_state=42),
            X_train,
            X_test,
        ),
    }

    predictions = {}
    metric_rows = []

    for model_name, (model, train_features, test_features) in models.items():
        model.fit(train_features, y_train)
        y_pred = model.predict(test_features)
        predictions[model_name] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        metric_rows.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, y_pred),
        })

    comparison = pd.DataFrame(metric_rows).sort_values("RMSE")
    print("\nModel Comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # Compare model errors.
    error_data = comparison.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=error_data, x="Model", y="Error", hue="Metric")
    plt.title("Model Error Comparison")
    plt.ylabel("Error (lower is better)")
    show_plot()

    # Compare actual and predicted final grades.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.7)
        lower = min(float(y_test.min()), float(np.min(y_pred)))
        upper = max(float(y_test.max()), float(np.max(y_pred)))
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set_xlabel("Actual final grade (G3)")
        ax.set_ylabel("Predicted final grade (G3)")
        ax.set_title(model_name)

    fig.suptitle("Actual vs Predicted Final Grades")
    show_plot()

    # Show sample predictions from the model with the lower test RMSE.
    best_model_name = comparison.iloc[0]["Model"]
    sample_results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions[best_model_name],
    }, index=y_test.index)

    print(f"\nSample predictions from {best_model_name}:")
    print(sample_results.head(10).round(2))
    print(f"\nLowest test RMSE in this comparison: {best_model_name}")


if __name__ == "__main__":
    main()
