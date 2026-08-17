"""Interactive Streamlit dashboard for shopper purchase-intent classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
REFERENCE_METRICS_PATH = MODEL_DIR / "metrics.json"
BUNDLED_HOLDOUT = PROJECT_ROOT / "test_data.csv"

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

MODEL_BLURBS = {
    "Logistic Regression": "Linear baseline with balanced class weights — high recall.",
    "Decision Tree": "Unpruned tree; easy to overfit this mixed tabular mix.",
    "kNN": "Distance votes on scaled features — majority class tends to dominate.",
    "Naive Bayes": "Gaussian assumption struggles with PageValues and one-hot flags.",
    "Random Forest (Ensemble)": "Hold-out winner on MCC, AUC and F1.",
}

TRUTHY = {True, "True", "TRUE", "true", 1, "1", "Yes", "YES"}
FALSY = {False, "False", "FALSE", "false", 0, "0", "No", "NO"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(21,29,46,0.65)",
    font=dict(color="#eef2ff", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#f5b942", "#38bdf8", "#a78bfa", "#34d399", "#fb7185"],
)


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


@st.cache_data
def load_reference_metrics() -> pd.DataFrame | None:
    if not REFERENCE_METRICS_PATH.exists():
        return None
    payload = json.loads(REFERENCE_METRICS_PATH.read_text())
    table = pd.DataFrame(payload).T
    return table[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(120deg, #1e293b 0%, #0f172a 55%, #1d283a 100%);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 1.4rem 1.6rem 1.2rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
        }
        .hero-card p { color: #cbd5e1; margin: 0; }
        .kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.7rem; }
        .kpi {
            background: #151d2e;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 0.85rem 0.7rem 0.75rem 0.7rem;
            text-align: center;
        }
        .kpi .label { color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .kpi .value { color: #f5b942; font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }
        .hint { color: #94a3b8; font-size: 0.9rem; }
        div[data-testid="stMetric"] {
            background: #151d2e;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 10px 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(scores: dict[str, float]) -> None:
    cells = []
    for name, value in scores.items():
        cells.append(
            f'<div class="kpi"><div class="label">{name}</div>'
            f'<div class="value">{value:.4f}</div></div>'
        )
    st.markdown('<div class="kpi-grid">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def comparison_bar_chart(reference_table: pd.DataFrame) -> go.Figure:
    long_table = (
        reference_table.reset_index()
        .rename(columns={"index": "Model"})
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
    )
    fig = px.bar(
        long_table,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Hold-out comparison — click a metric in the legend to isolate it",
    )
    fig.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.22), height=420)
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(range=[0, 1], gridcolor="#334155")
    return fig


def confusion_figure(y_true, y_pred) -> go.Figure:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="YlOrBr",
        labels=dict(x="Predicted intent", y="Actual outcome", color="Sessions"),
        x=["No purchase", "Purchase"],
        y=["No purchase", "Purchase"],
        title="Confusion matrix",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, coloraxis_showscale=False)
    return fig


def roc_figure(y_true, y_proba, auc_value: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            fill="tozeroy",
            name=f"ROC (AUC {auc_value:.3f})",
            line=dict(color="#f5b942", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color="#64748b", dash="dash"),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=420,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", range=[0, 1]),
    )
    return fig


def probability_figure(y_true, y_proba) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Purchase probability": y_proba,
            "Actual": np.where(y_true == 1, "Purchase", "No purchase"),
        }
    )
    fig = px.histogram(
        frame,
        x="Purchase probability",
        color="Actual",
        nbins=28,
        barmode="overlay",
        opacity=0.75,
        title="How confident is the model? (hover a bar)",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, bargap=0.05)
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155", title="Sessions")
    return fig


def month_breakdown_figure(raw_sessions: pd.DataFrame, predicted, purchase_label) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Month": raw_sessions["Month"].values,
            "Actual purchase rate": purchase_label.values,
            "Predicted purchase rate": predicted,
        }
    )
    grouped = frame.groupby("Month", as_index=False).mean(numeric_only=True)
    long_table = grouped.melt(id_vars="Month", var_name="Series", value_name="Rate")
    fig = px.bar(
        long_table,
        x="Month",
        y="Rate",
        color="Series",
        barmode="group",
        title="Purchase rate by month — actual vs predicted",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=400, legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(range=[0, 1], gridcolor="#334155")
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Shopper Purchase Intent",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1>Will this session convert?</h1>
            <p>
                Score live e-commerce sessions against five classifiers trained on
                12,330 UCI shopper visits. Upload a CSV (or load the bundled hold-out)
                and inspect metrics, confusion, ROC and individual predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Session source")
        data_source = st.radio(
            "Choose data",
            ["Upload CSV", "Bundled hold-out (test_data.csv)"],
            label_visibility="collapsed",
        )
        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Test sessions CSV",
                type=["csv"],
                help="Needs the original 18 columns, including Revenue.",
            )
        selected_model = st.selectbox("Classifier", list(MODEL_ARTIFACTS.keys()), index=4)
        st.caption(MODEL_BLURBS[selected_model])
        threshold = st.slider(
            "Purchase decision threshold",
            min_value=0.15,
            max_value=0.85,
            value=0.50,
            step=0.05,
            help="Default 0.50 matches the README table. Lower it to catch more buyers (higher recall).",
        )
        st.markdown("---")
        st.caption(
            "Required columns: page-visit counts/durations, BounceRates, ExitRates, "
            "PageValues, SpecialDay, Month, OS/Browser/Region/TrafficType, VisitorType, "
            "Weekend, Revenue."
        )

    reference_table = load_reference_metrics()

    raw_sessions = None
    if data_source == "Bundled hold-out (test_data.csv)":
        raw_sessions = pd.read_csv(BUNDLED_HOLDOUT)
    elif uploaded is not None:
        raw_sessions = pd.read_csv(uploaded)

    overview_tab, live_tab, compare_tab = st.tabs(
        ["Hold-out overview", "Live evaluation", "Model comparison"]
    )

    with overview_tab:
        if reference_table is not None:
            winner = reference_table["MCC"].idxmax()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sessions in full dataset", "12,330")
            c2.metric("Purchase rate", "15.47%")
            c3.metric("Hold-out sessions", "2,466")
            c4.metric("Winner (MCC)", winner.replace(" (Ensemble)", ""))
            st.plotly_chart(comparison_bar_chart(reference_table), use_container_width=True)
            st.caption(
                "These are the official assignment metrics (stratified 80/20, random_state=42). "
                "PageValues dominates Random Forest importance (~37%)."
            )
        else:
            st.warning("model/metrics.json is missing. Re-run train.py.")

    with compare_tab:
        if reference_table is None:
            st.info("Train models first to populate the comparison table.")
        else:
            st.dataframe(
                reference_table.style.format("{:.4f}").highlight_max(axis=0, color="#7c5a12"),
                use_container_width=True,
            )
            radar = go.Figure()
            theta = list(reference_table.columns)
            for model_name, row in reference_table.iterrows():
                radar.add_trace(
                    go.Scatterpolar(
                        r=list(row.values) + [row.values[0]],
                        theta=theta + [theta[0]],
                        name=model_name,
                        fill="toself",
                        opacity=0.35 if model_name != selected_model else 0.7,
                    )
                )
            radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="rgba(21,29,46,0.65)",
                    radialaxis=dict(range=[0, 1], gridcolor="#334155"),
                    angularaxis=dict(gridcolor="#334155"),
                ),
                title="Radar — selected model is drawn more opaque",
                height=520,
            )
            st.plotly_chart(radar, use_container_width=True)

    with live_tab:
        if raw_sessions is None:
            st.markdown(
                '<p class="hint">Upload a CSV in the sidebar, or switch to the bundled hold-out set to explore the dashboard immediately.</p>',
                unsafe_allow_html=True,
            )
            return

        try:
            session_features, purchase_label = prepare_uploaded_sessions(raw_sessions)
        except Exception as exc:
            st.error(str(exc))
            return

        pipeline = load_intent_pipeline(MODEL_ARTIFACTS[selected_model])
        purchase_proba = pipeline.predict_proba(session_features)[:, 1]
        # sklearn trees vote for predict(); that can differ slightly from proba >= 0.5
        if abs(threshold - 0.50) < 1e-9:
            predicted = pipeline.predict(session_features)
        else:
            predicted = (purchase_proba >= threshold).astype(int)
        live_scores = score_predictions(purchase_label, predicted, purchase_proba)

        n_sessions = len(session_features)
        n_buyers = int(purchase_label.sum())
        n_flagged = int(predicted.sum())

        st.subheader(f"{selected_model}  ·  threshold {threshold:.2f}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sessions scored", f"{n_sessions:,}")
        k2.metric("Actual purchases", f"{n_buyers:,}", f"{100 * n_buyers / n_sessions:.1f}%")
        k3.metric("Predicted purchases", f"{n_flagged:,}", f"{100 * n_flagged / n_sessions:.1f}%")
        k4.metric("MCC", f"{live_scores['MCC']:.4f}")

        st.markdown("#### Evaluation metrics")
        render_kpi_row(live_scores)

        viz1, viz2 = st.columns(2)
        with viz1:
            st.plotly_chart(confusion_figure(purchase_label, predicted), use_container_width=True)
        with viz2:
            st.plotly_chart(roc_figure(purchase_label, purchase_proba, live_scores["AUC"]), use_container_width=True)

        viz3, viz4 = st.columns(2)
        with viz3:
            st.plotly_chart(probability_figure(purchase_label, purchase_proba), use_container_width=True)
        with viz4:
            st.plotly_chart(
                month_breakdown_figure(raw_sessions, predicted, purchase_label),
                use_container_width=True,
            )

        with st.expander("Classification report", expanded=False):
            report_text = classification_report(
                purchase_label,
                predicted,
                target_names=["No purchase", "Purchase"],
                zero_division=0,
            )
            st.code(report_text, language=None)

        st.markdown("#### Session explorer")
        explorer = raw_sessions.copy()
        explorer["actual_purchase"] = purchase_label.values
        explorer["predicted_purchase"] = predicted
        explorer["purchase_probability"] = np.round(purchase_proba, 4)
        explorer["correct"] = explorer["actual_purchase"] == explorer["predicted_purchase"]

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        visitor_filter = filter_col1.multiselect(
            "Visitor type",
            sorted(explorer["VisitorType"].astype(str).unique()),
            default=None,
        )
        outcome_filter = filter_col2.selectbox(
            "Show",
            ["All sessions", "Correct only", "Mistakes only", "Predicted purchases"],
        )
        month_filter = filter_col3.multiselect(
            "Month",
            list(explorer["Month"].astype(str).unique()),
            default=None,
        )

        view = explorer
        if visitor_filter:
            view = view[view["VisitorType"].astype(str).isin(visitor_filter)]
        if month_filter:
            view = view[view["Month"].astype(str).isin(month_filter)]
        if outcome_filter == "Correct only":
            view = view[view["correct"]]
        elif outcome_filter == "Mistakes only":
            view = view[~view["correct"]]
        elif outcome_filter == "Predicted purchases":
            view = view[view["predicted_purchase"] == 1]

        display_cols = [
            "Month",
            "VisitorType",
            "PageValues",
            "ExitRates",
            "ProductRelated",
            "Weekend",
            "actual_purchase",
            "predicted_purchase",
            "purchase_probability",
            "correct",
        ]
        st.dataframe(view[display_cols], use_container_width=True, height=320)
        st.caption(f"Showing {len(view):,} of {len(explorer):,} sessions. Click column headers to sort.")


if __name__ == "__main__":
    main()
