from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "gym-churn-prediction" / "data" / "gym_churn.csv"
RESULTS_PATH = BASE_DIR / "gym-churn-prediction" / "results" / "tables"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_PATH / "model_comparison.csv"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = data.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    numeric_features = [
        "Age",
        "Avg_Workout_Duration_Min",
        "Avg_Calories_Burned",
        "Total_Weight_Lifted_kg",
        "Visits_Per_Month",
    ]
    categorical_features = [
        "Gender",
        "Membership_Type",
        "Favorite_Exercise",
    ]

    df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors="coerce")
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    X = df[numeric_features + categorical_features]
    y = df["Churn"].astype(int)
    return X, y


def build_pipeline(model):
    numeric_features = [
        "Age",
        "Avg_Workout_Duration_Min",
        "Avg_Calories_Burned",
        "Total_Weight_Lifted_kg",
        "Visits_Per_Month",
    ]
    categorical_features = [
        "Gender",
        "Membership_Type",
        "Favorite_Exercise",
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict[str, object]:
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def main() -> None:
    data = load_data(DATA_PATH)
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]

    results = []
    for name, model in models:
        print(f"Training and evaluating: {name}")
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nModel comparison saved to:", RESULTS_FILE)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
