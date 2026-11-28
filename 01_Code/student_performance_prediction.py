"""Student Performance Prediction.

Run from the repository root:
    python 01_Code/student_performance_prediction.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "03_Dataset" / "student-mat.csv"
RANDOM_STATE = 42


def build_model(features, estimator, scale_numeric=False):
    """Create a preprocessing + estimator pipeline fitted only on training folds."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessing), ("model", estimator)]
    )


def calculate_metrics(actual, predicted):
    """Return standard regression metrics."""
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


def plot_eda(data):
    """Display the main exploratory plots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="G3", bins=15, ax=ax)
    ax.set_title("Final grade distribution")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="studytime", y="G3", errorbar=None, ax=ax)
    ax.set_title("Study time vs G3")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="G2", y="G3", alpha=0.7, ax=ax)
    ax.set_title("G2 vs G3")
    fig.tight_layout()
    plt.show()


def evaluate_setting(label, train_features, test_features, y_train, y_test):
    """Compare baseline and models for one feature-availability setting."""
    print(f"\n{'=' * 65}\n{label}")
    model_definitions = {
        "Mean baseline": build_model(
            train_features, DummyRegressor(strategy="mean")
        ),
        "Linear Regression": build_model(
            train_features, LinearRegression(), scale_numeric=True
        ),
        "Random Forest (default)": build_model(
            train_features,
            RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
    }

    test_results = []
    cv_results = []
    for model_name, model in model_definitions.items():
        cv_scores = -cross_val_score(
            model,
            train_features,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_results.append(
            {
                "Experiment": label,
                "Model": model_name,
                "CV RMSE mean": cv_scores.mean(),
                "CV RMSE std": cv_scores.std(),
            }
        )

        model.fit(train_features, y_train)
        predictions = model.predict(test_features)
        test_results.append(
            {
                "Experiment": label,
                "Model": model_name,
                **calculate_metrics(y_test, predictions),
            }
        )

    random_forest = build_model(
        train_features,
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    )
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        random_forest,
        parameter_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(train_features, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(test_features)
    test_results.append(
        {
            "Experiment": label,
            "Model": "Random Forest (tuned)",
            **calculate_metrics(y_test, predictions),
        }
    )
    cv_results.append(
        {
            "Experiment": label,
            "Model": "Random Forest (tuned)",
            "CV RMSE mean": -search.best_score_,
            "CV RMSE std": np.nan,
        }
    )
    print("Best parameters:", search.best_params_)

    residuals = y_test - predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, predictions, alpha=0.7)
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    axes[0].plot([lower, upper], [lower, upper], "k--")
    axes[0].set(
        title="Actual vs predicted", xlabel="Actual G3", ylabel="Predicted G3"
    )
    sns.scatterplot(x=predictions, y=residuals, ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(
        title="Residual plot",
        xlabel="Predicted G3",
        ylabel="Actual - predicted",
    )
    fig.suptitle(label)
    fig.tight_layout()
    plt.show()

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(
        best_model.named_steps["model"].feature_importances_,
        index=feature_names,
    ).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Random Forest feature importances")
    fig.tight_layout()
    plt.show()

    positive_grade_mask = y_test > 0
    if positive_grade_mask.any():
        print(
            "Metrics on positive-grade test records:",
            calculate_metrics(
                y_test[positive_grade_mask], predictions[positive_grade_mask]
            ),
        )

    return test_results, cv_results


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, sep=";").drop_duplicates().copy()
    required_columns = {"G1", "G2", "G3", "studytime", "absences"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    print("Dataset shape:", data.shape)
    print("Missing values:", int(data.isnull().sum().sum()))
    zero_grade_records = data[data["G3"] == 0]
    print("G3 = 0 records:", len(zero_grade_records))
    print("Absences summary for G3 = 0 records:")
    print(zero_grade_records["absences"].describe())
    print("The dataset does not explain why these grades are zero.")

    plot_eda(data)

    target = data["G3"]
    features = data.drop(columns=["G3"])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    settings = {
        "With G1/G2 (earlier grades available)": (X_train, X_test),
        "Without G1/G2 (earlier grades unavailable)": (
            X_train.drop(columns=["G1", "G2"]),
            X_test.drop(columns=["G1", "G2"]),
        ),
    }

    all_test_results = []
    all_cv_results = []
    for label, (train_part, test_part) in settings.items():
        test_results, cv_results = evaluate_setting(
            label, train_part, test_part, y_train, y_test
        )
        all_test_results.extend(test_results)
        all_cv_results.extend(cv_results)

    print("\nFive-fold cross-validation results (training data only):")
    print(pd.DataFrame(all_cv_results).round(3).to_string(index=False))
    print("\nFinal hold-out test results:")
    print(pd.DataFrame(all_test_results).round(3).to_string(index=False))
    print(
        "\nFeature importance is not causal evidence. The two feature settings "
        "represent different prediction tasks."
    )


if __name__ == "__main__":
    main()
