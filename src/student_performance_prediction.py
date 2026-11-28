"""Student Performance Prediction: run from repository root with python src/student_performance_prediction.py."""
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
DATA_PATH = ROOT / "data" / "student-mat.csv"

def build_model(X, model, scale=False):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale: num_steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(num_steps), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([("preprocessor", prep), ("model", model)])

def get_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    return {"MAE": mean_absolute_error(actual, pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(actual, pred)}

def main():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=";").drop_duplicates().copy()
    print("Dataset shape:", df.shape, "| Missing values:", int(df.isnull().sum().sum()))
    zeros = df[df.G3 == 0]
    print("G3 = 0 records:", len(zeros), "| Absences summary:")
    print(zeros.absences.describe())
    print("The dataset does not explain why these grades are zero.")

    plt.figure(figsize=(8, 5)); sns.histplot(data=df, x="G3", bins=15); plt.title("Final grade distribution"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(8, 5)); sns.barplot(data=df, x="studytime", y="G3", errorbar=None); plt.title("Study time vs G3"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(8, 5)); sns.scatterplot(data=df, x="G2", y="G3", alpha=.7); plt.title("G2 vs G3"); plt.tight_layout(); plt.show()

    y = df.G3
    X = df.drop(columns=["G3"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    experiments = {
        "With G1/G2 (earlier grades available)": (Xtr, Xte),
        "Without G1/G2 (earlier grades unavailable)": (Xtr.drop(columns=["G1", "G2"]), Xte.drop(columns=["G1", "G2"])),
    }
    test_rows, cv_rows = [], []
    for label, (a, b) in experiments.items():
        print("\\n" + "=" * 60 + "\\n" + label)
        models = {
            "Mean baseline": build_model(a, DummyRegressor(strategy="mean")),
            "Linear Regression": build_model(a, LinearRegression(), scale=True),
            "Random Forest (default)": build_model(a, RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        }
        for name, model in models.items():
            scores = -cross_val_score(model, a, ytr, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
            cv_rows.append({"Experiment": label, "Model": name, "CV RMSE mean": scores.mean(), "CV RMSE std": scores.std()})
            model.fit(a, ytr); pred = model.predict(b)
            test_rows.append({"Experiment": label, "Model": name, **get_metrics(yte, pred)})
        rf = build_model(a, RandomForestRegressor(random_state=42, n_jobs=-1))
        grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 2]}
        search = GridSearchCV(rf, grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(a, ytr); best = search.best_estimator_; pred = best.predict(b)
        test_rows.append({"Experiment": label, "Model": "Random Forest (tuned)", **get_metrics(yte, pred)})
        cv_rows.append({"Experiment": label, "Model": "Random Forest (tuned)", "CV RMSE mean": -search.best_score_, "CV RMSE std": np.nan})
        print("Best parameters:", search.best_params_)

        residuals = yte - pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(yte, pred, alpha=.7); lo=min(float(yte.min()), float(pred.min())); hi=max(float(yte.max()), float(pred.max())); axes[0].plot([lo,hi],[lo,hi],"k--"); axes[0].set(title="Actual vs predicted", xlabel="Actual G3", ylabel="Predicted G3")
        sns.scatterplot(x=pred, y=residuals, ax=axes[1]); axes[1].axhline(0,color="black",linestyle="--"); axes[1].set(title="Residual plot", xlabel="Predicted G3", ylabel="Actual - predicted")
        fig.suptitle(label); plt.tight_layout(); plt.show()
        names = best.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.Series(best.named_steps["model"].feature_importances_, index=names).sort_values().tail(15)
        plt.figure(figsize=(9,6)); importance.plot(kind="barh"); plt.title("Top 15 Random Forest feature importances"); plt.tight_layout(); plt.show()
        mask = yte > 0
        if mask.any(): print("Metrics on positive-grade test records:", get_metrics(yte[mask], pred[mask]))
    print("\\nFive-fold CV results (training data only):"); print(pd.DataFrame(cv_rows).round(3).to_string(index=False))
    print("\\nFinal hold-out test results:"); print(pd.DataFrame(test_rows).round(3).to_string(index=False))
    print("Feature importance is not causal evidence. Compare the two feature settings as different prediction tasks.")

if __name__ == "__main__": main()