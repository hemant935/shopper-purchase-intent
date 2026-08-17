"""Streamlit app: compare purchase-intent classifiers on uploaded session CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
REFERENCE_METRICS_PATH = MODEL_DIR / "metrics.json"

REQUIRED_FEATURE_COLS = [
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
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MODEL_ARTIFACTS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

TRUTHY = {True, "True", "TRUE", "true", 1, "1", "Yes", "YES"}
FALSY = {False, "False", "FALSE", "false", 0, "0", "No", "NO"}


def to_binary_series(column: pd.Series) -> pd.Series:
    mapped = column.map(lambda value: 1 if value in TRUTHY else (0 if value in FALSY else pd.NA))
    if mapped.isna().any():
        mapped = pd.to_numeric(column, errors="coerce")
    return mapped.astype("Int64")


def prepare_uploaded_sessions(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [col for col in REQUIRED_FEATURE_COLS if col not in raw_frame.columns]
    if missing_features:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            + ", ".join(missing_features)
        )
    if "Revenue" not in raw_frame.columns:
        raise ValueError(
            "Uploaded CSV must include a Revenue column (True/False) "
            "so evaluation metrics can be computed."
        )

    session_features = raw_frame[REQUIRED_FEATURE_COLS].copy()
    session_features["Weekend"] = to_binary_series(session_features["Weekend"]).astype(int)
    purchase_label = to_binary_series(raw_frame["Revenue"])
    if purchase_label.isna().any():
        raise ValueError("Revenue contains values that could not be mapped to 0/1.")
    return session_features, purchase_label.astype(int)


def score_predictions(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


@st.cache_resource
def load_intent_pipeline(artifact_name: str):
    return joblib.load(MODEL_DIR / artifact_name)


def load_reference_metrics() -> pd.DataFrame | None:
    if not REFERENCE_METRICS_PATH.exists():
        return None
    payload = json.loads(REFERENCE_METRICS_PATH.read_text())
    return pd.DataFrame(payload).T


def render_confusion_heatmap(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=ax,
    )
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(
        page_title="Shopper Purchase Intent",
        page_icon="🛒",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #f4f7f5; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e3db;
            border-radius: 12px;
            padding: 12px 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Shopper purchase intent")
    st.caption(
        "Score an e-commerce session CSV against five classifiers trained on the "
        "UCI Online Shoppers Purchasing Intention dataset. Metrics are computed "
        "against the Revenue label in the file you upload."
    )

    with st.sidebar:
        st.header("Session file")
        uploaded = st.file_uploader(
            "Upload test sessions (CSV)",
            type=["csv"],
            help="Use test_data.csv from this repo, or any CSV with the original 18 columns including Revenue.",
        )
        selected_model = st.selectbox(
            "Classifier",
            list(MODEL_ARTIFACTS.keys()),
            index=4,
        )
        st.markdown(
            "Expected columns match the original dataset: page-visit counts and "
            "durations, BounceRates, ExitRates, PageValues, SpecialDay, Month, "
            "device/traffic fields, VisitorType, Weekend, and **Revenue**."
        )

    reference_table = load_reference_metrics()
    if reference_table is not None:
        with st.expander("Reference scores on the original 20% hold-out set", expanded=True):
            st.dataframe(reference_table, use_container_width=True)
            st.caption(
                "These numbers come from training (random_state=42, stratified 80/20). "
                "Uploading test_data.csv should reproduce them for the selected model."
            )

    if uploaded is None:
        st.info("Upload a CSV in the sidebar to evaluate a model on your sessions.")
        return

    try:
        raw_sessions = pd.read_csv(uploaded)
        session_features, purchase_label = prepare_uploaded_sessions(raw_sessions)
    except Exception as exc:
        st.error(str(exc))
        return

    pipeline = load_intent_pipeline(MODEL_ARTIFACTS[selected_model])
    predicted = pipeline.predict(session_features)
    purchase_proba = pipeline.predict_proba(session_features)[:, 1]
    live_scores = score_predictions(purchase_label, predicted, purchase_proba)

    st.subheader(f"Results on uploaded sessions — {selected_model}")
    st.caption(f"{len(session_features):,} sessions  ·  {int(purchase_label.sum()):,} actual purchases")

    metric_cols = st.columns(6)
    for column, (metric_name, metric_value) in zip(metric_cols, live_scores.items()):
        column.metric(metric_name, f"{metric_value:.4f}")

    chart_col, report_col = st.columns([1.1, 1])
    with chart_col:
        render_confusion_heatmap(purchase_label, predicted)
    with report_col:
        st.markdown("**Classification report**")
        report_text = classification_report(
            purchase_label,
            predicted,
            target_names=["No purchase", "Purchase"],
            zero_division=0,
        )
        st.code(report_text, language=None)


if __name__ == "__main__":
    main()
