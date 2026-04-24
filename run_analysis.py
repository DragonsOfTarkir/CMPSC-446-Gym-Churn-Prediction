from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "gym-churn-prediction" / "data" / "gym_churn.csv"
RESULTS_PATH = BASE_DIR / "gym-churn-prediction" / "results" / "tables"
FIGURES_PATH = BASE_DIR / "gym-churn-prediction" / "results" / "figures"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
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


def plot_churn_distribution(y: pd.Series, output_path: Path) -> None:
    churn_labels = y.map({0: "No", 1: "Yes"}).value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(churn_labels.index, churn_labels.values, color=["skyblue", "salmon"])
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(numeric_df: pd.DataFrame, output_path: Path) -> None:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"], rotation=0)
    ax.set_title("Random Forest Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_accuracy(results_df: pd.DataFrame, output_path: Path) -> None:
    results_sorted = results_df.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(results_sorted["model"], results_sorted["accuracy"], color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Model")
    ax.set_title("Model Accuracy Comparison")
    for i, value in enumerate(results_sorted["accuracy"]):
        ax.annotate(f"{value:.2f}", (value + 0.01, i), va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, feature_names: list[str], output_path: Path) -> None:
    importances = pipeline.named_steps["classifier"].feature_importances_
    indices = np.argsort(importances)
    sorted_feature_names = np.array(feature_names)[indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.35)))
    ax.barh(sorted_feature_names, sorted_importances, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    rf_pipeline = build_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out(X_train.columns)

    plot_churn_distribution(y, FIGURES_PATH / "churn_distribution.png")
    plot_correlation_heatmap(
        pd.concat([X["Age"], X["Avg_Workout_Duration_Min"], X["Avg_Calories_Burned"], X["Total_Weight_Lifted_kg"], X["Visits_Per_Month"], y.rename("Churn")], axis=1),
        FIGURES_PATH / "correlation_heatmap.png",
    )
    plot_confusion_matrix(y_test, y_pred_rf, FIGURES_PATH / "confusion_matrix.png")
    plot_model_accuracy(results_df, FIGURES_PATH / "model_accuracy.png")
    plot_feature_importance(
        rf_pipeline,
        feature_names.tolist(),
        FIGURES_PATH / "feature_importance.png",
    )

    print("\nModel comparison saved to:", RESULTS_FILE)
    print("Figures updated in:", FIGURES_PATH)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
