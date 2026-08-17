"""Train purchase-intent classifiers and persist pipelines for the Streamlit app.

Run from this project folder:
    python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
SESSION_CSV = PROJECT_ROOT / "online_shoppers_intention.csv"
MODEL_DIR = PROJECT_ROOT / "model"
HOLDOUT_CSV = PROJECT_ROOT / "test_data.csv"

NUMERIC_SESSION_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend",
]
CATEGORY_SESSION_COLS = ["Month", "VisitorType"]

INTENT_MODELS = {
    "Logistic Regression": {
        "file": "logistic_regression.pkl",
        "estimator": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    },
    "Decision Tree": {
        "file": "decision_tree.pkl",
        "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    },
    "kNN": {
        "file": "knn.pkl",
        "estimator": KNeighborsClassifier(n_neighbors=5),
    },
    "Naive Bayes": {
        "file": "naive_bayes.pkl",
        "estimator": GaussianNB(),
    },
    "Random Forest (Ensemble)": {
        "file": "random_forest.pkl",
        "estimator": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    },
}


def build_session_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric_scale", StandardScaler(), NUMERIC_SESSION_COLS),
            (
                "category_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORY_SESSION_COLS,
            ),
        ]
    )


def score_holdout(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    shopper_sessions = pd.read_csv(SESSION_CSV)
    print("Dataset shape:", shopper_sessions.shape)
    print("Missing values:", int(shopper_sessions.isna().sum().sum()))
    print("Class balance:\n", shopper_sessions["Revenue"].value_counts(normalize=True))

    session_features = shopper_sessions.drop(columns=["Revenue"]).copy()
    session_features["Weekend"] = session_features["Weekend"].astype(int)
    purchase_label = shopper_sessions["Revenue"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        session_features,
        purchase_label,
        test_size=0.20,
        stratify=purchase_label,
        random_state=42,
    )

    shopper_sessions.loc[X_test.index].to_csv(HOLDOUT_CSV, index=False)
    print(f"Saved hold-out CSV: {HOLDOUT_CSV} ({len(X_test)} rows)")

    holdout_metrics: dict[str, dict[str, float]] = {}
    rf_pipeline = None

    for model_name, spec in INTENT_MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("session_prep", build_session_preprocessor()),
                ("classifier", spec["estimator"]),
            ]
        )
        pipeline.fit(X_train, y_train)

        predicted = pipeline.predict(X_test)
        purchase_proba = pipeline.predict_proba(X_test)[:, 1]
        holdout_metrics[model_name] = score_holdout(y_test, predicted, purchase_proba)

        artifact_path = MODEL_DIR / spec["file"]
        joblib.dump(pipeline, artifact_path)
        print(f"Saved {model_name} -> {artifact_path}")

        if model_name == "Random Forest (Ensemble)":
            rf_pipeline = pipeline

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(holdout_metrics, indent=2))

    metrics_table = pd.DataFrame(holdout_metrics).T
    print("\nHold-out metrics (use this table in README and BITS Lab screenshot):\n")
    print(metrics_table.to_string())
    print(f"\nWrote {metrics_path}")

    if rf_pipeline is not None:
        prepared = rf_pipeline.named_steps["session_prep"]
        forest = rf_pipeline.named_steps["classifier"]
        feature_names = prepared.get_feature_names_out()
        importance = pd.Series(forest.feature_importances_, index=feature_names)
        print("\nRandom Forest — top 8 features:\n")
        print(importance.sort_values(ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
