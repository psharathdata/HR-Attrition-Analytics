# ============================================================
# HR Attrition Analytics — PRO PLUS Streamlit App
# Author-ready portfolio dashboard for HR analytics
# Run: streamlit run hr_attrition_app_PRO_PLUS_FIXED.py
# Install: pip install streamlit pandas numpy plotly scikit-learn xgboost imbalanced-learn shap openpyxl
# Dataset file needed in same folder OR upload inside app:
# WA_Fn-UseC_-HR-Employee-Attrition.csv
# ============================================================

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_OK = True
except Exception:
    SMOTE = None
    ImbPipeline = Pipeline
    IMBLEARN_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    shap = None
    SHAP_OK = False


# ============================================================
# STREAMLIT EXECUTION GUARD
# ============================================================
def _ensure_streamlit_run() -> None:
    """Stop cleanly when user accidentally runs: python file.py"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            print("\n❌ This is a Streamlit app. Do not run with python3.")
            print("✅ Run like this:")
            print(f"   streamlit run \"{Path(__file__).resolve()}\"\n")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        # If Streamlit internals change, continue normally under streamlit run.
        pass

_ensure_streamlit_run()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HR Attrition Intelligence | Pro Plus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================
ACCENT = "#ff4d6d"
ACCENT2 = "#7c3aed"
SAFE = "#00d4a6"
AMBER = "#ffb020"
BLUE = "#38bdf8"
BG = "#070A13"
CARD = "rgba(255,255,255,0.075)"
TEXT = "#f8fafc"
MUTED = "#a7b0c0"
GRID = "rgba(255,255,255,0.10)"

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{
        background:
            radial-gradient(circle at 15% 8%, rgba(124,58,237,0.26), transparent 28%),
            radial-gradient(circle at 82% 12%, rgba(255,77,109,0.20), transparent 25%),
            radial-gradient(circle at 50% 96%, rgba(0,212,166,0.14), transparent 30%),
            linear-gradient(135deg, #060713 0%, #0b1020 45%, #090b16 100%);
        color: {TEXT};
    }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1480px; }}

    div[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(8,11,23,0.98), rgba(13,18,35,0.92));
        border-right: 1px solid rgba(255,255,255,0.10);
    }}
    div[data-testid="stSidebar"] * {{ color: #e5e7eb; }}

    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(124,58,237,.22), rgba(255,77,109,.16));
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 28px;
        padding: 34px 34px 30px 34px;
        box-shadow: 0 24px 70px rgba(0,0,0,.34);
        backdrop-filter: blur(18px);
    }}
    .hero::after {{
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -120px;
        top: -120px;
        background: radial-gradient(circle, rgba(255,255,255,.16), transparent 62%);
        pointer-events: none;
    }}
    .eyebrow {{
        display:inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.12);
        color: #dbeafe;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}
    .hero h1 {{
        font-size: clamp(2.1rem, 5vw, 4.6rem);
        line-height: .94;
        letter-spacing: -0.08em;
        margin: 14px 0 12px 0;
        font-weight: 950;
        color: #fff;
    }}
    .hero p {{
        max-width: 980px;
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0;
    }}
    .glass-card {{
        background: {CARD};
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 14px 45px rgba(0,0,0,0.24);
        backdrop-filter: blur(16px);
    }}
    .metric-card {{
        background: linear-gradient(145deg, rgba(255,255,255,.10), rgba(255,255,255,.045));
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 22px;
        padding: 20px 18px;
        min-height: 126px;
        box-shadow: 0 12px 35px rgba(0,0,0,.24);
    }}
    .metric-label {{ color: {MUTED}; font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: .09em; }}
    .metric-value {{ color: #fff; font-size: 2.05rem; font-weight: 950; letter-spacing: -.04em; margin-top: 8px; }}
    .metric-note {{ color: #cbd5e1; font-size: .82rem; margin-top: 4px; }}
    .section-title {{
        font-size: 1.35rem;
        font-weight: 900;
        color: #fff;
        letter-spacing: -.03em;
        margin: 24px 0 10px 0;
    }}
    .insight {{
        background: linear-gradient(135deg, rgba(255,255,255,.09), rgba(255,255,255,.045));
        border: 1px solid rgba(255,255,255,.12);
        border-left: 5px solid {ACCENT};
        border-radius: 18px;
        padding: 15px 17px;
        color: #dbe3ef;
        margin: 8px 0;
        line-height: 1.52;
    }}
    .insight.green {{ border-left-color: {SAFE}; }}
    .insight.amber {{ border-left-color: {AMBER}; }}
    .insight.blue {{ border-left-color: {BLUE}; }}
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        margin: 4px 5px 4px 0;
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.12);
        color: #e2e8f0;
        font-size: .82rem;
        font-weight: 700;
    }}
    .risk-big {{
        text-align: center;
        border-radius: 28px;
        padding: 34px;
        background: radial-gradient(circle at 50% 0%, rgba(255,77,109,.20), rgba(255,255,255,.05));
        border: 1px solid rgba(255,255,255,.13);
        box-shadow: 0 20px 60px rgba(0,0,0,.30);
    }}
    .risk-number {{ font-size: 5.4rem; line-height: 1; font-weight: 950; letter-spacing: -.07em; }}
    div[data-testid="metric-container"] {{
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 18px;
        padding: 14px;
    }}
    .stDataFrame, div[data-testid="stDataFrame"] {{ border-radius: 18px; overflow: hidden; }}
    hr {{ border-color: rgba(255,255,255,.10); }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
DEFAULT_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
BASE_DROP_COLS = ["EmployeeCount", "Over18", "StandardHours"]
TARGET_COLS = ["Attrition", "AttritionBinary"]
LEAKAGE_COLS = ["RiskScore", "RiskCategory", "RiskBand", "PredictedAttrition", "Probability"]
DERIVED_COLS = ["AgeGroup", "SalaryBand", "TenureBand", "IncomeBand"]


def plotly_layout(fig: go.Figure, height: int = 390, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.035)",
        font=dict(color=TEXT, family="Inter"),
        height=height,
        margin=dict(l=18, r=18, t=58, b=25),
        legend=dict(bgcolor="rgba(0,0,0,0)") if legend else None,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def metric_card(label: str, value: str, note: str = "", color: str = "#ffffff") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text: str, tone: str = "") -> None:
    st.markdown(f'<div class="insight {tone}">{text}</div>', unsafe_allow_html=True)


def safe_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)
        .agg(Employees=("AttritionBinary", "size"), Leavers=("AttritionBinary", "sum"))
        .reset_index()
    )
    out["AttritionRate"] = np.where(out["Employees"] > 0, out["Leavers"] / out["Employees"] * 100, 0)
    return out.sort_values("AttritionRate", ascending=False)


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def enrich_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in BASE_DROP_COLS if c in df.columns], errors="ignore")

    if "Attrition" not in df.columns:
        st.error("Dataset-il `Attrition` column kandilla. Correct IBM HR CSV upload cheyyu.")
        st.stop()

    df["AttritionBinary"] = df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if df["AttritionBinary"].isna().any():
        st.error("`Attrition` column Yes/No format-il alla. Please check dataset.")
        st.stop()
    df["AttritionBinary"] = df["AttritionBinary"].astype(int)

    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=[17, 25, 32, 40, 50, 65], labels=["18–25", "26–32", "33–40", "41–50", "51+"], include_lowest=True
        )
    if "MonthlyIncome" in df.columns:
        df["SalaryBand"] = pd.cut(
            df["MonthlyIncome"], bins=[0, 3000, 6000, 10000, 25000], labels=["<3K", "3K–6K", "6K–10K", "10K+"], include_lowest=True
        )
    if "YearsAtCompany" in df.columns:
        df["TenureBand"] = pd.cut(
            df["YearsAtCompany"], bins=[-1, 1, 3, 7, 12, 50], labels=["0–1", "2–3", "4–7", "8–12", "13+"], include_lowest=True
        )
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    # CRITICAL FIX: remove generated risk columns before prediction/training.
    # This prevents: feature_names mismatch: RiskScore, RiskCategory.
    drop_cols = TARGET_COLS + LEAKAGE_COLS + DERIVED_COLS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["AttritionBinary"].astype(int)

    # Remove IDs from modelling? EmployeeNumber is unique ID; keep available in dashboard but not model signal.
    if "EmployeeNumber" in X.columns:
        X = X.drop(columns=["EmployeeNumber"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return X, y, cat_cols, num_cols


@st.cache_resource(show_spinner=False)
def train_attrition_model(df: pd.DataFrame):
    X, y, cat_cols, num_cols = split_features_target(df)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    xgb = XGBClassifier(
        n_estimators=420,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.86,
        colsample_bytree=0.86,
        min_child_weight=2,
        reg_lambda=1.2,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    if IMBLEARN_OK:
        pipe = ImbPipeline(steps=[("prep", preprocessor), ("smote", SMOTE(random_state=42)), ("model", xgb)])
    else:
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])

    pipe.fit(X_train, y_train)

    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.50).astype(int)
    all_prob = pipe.predict_proba(X)[:, 1]

    metrics = {
        "auc": roc_auc_score(y_test, test_prob),
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "cm": confusion_matrix(y_test, test_pred),
        "report": pd.DataFrame(classification_report(y_test, test_pred, target_names=["Stayed", "Left"], output_dict=True)).T,
        "fpr_tpr": roc_curve(y_test, test_prob),
    }

    transformed_names = list(pipe.named_steps["prep"].get_feature_names_out())
    model = pipe.named_steps["model"]
    importances = pd.DataFrame({"Feature": transformed_names, "Importance": model.feature_importances_})
    importances["FeatureClean"] = importances["Feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    importances = importances.sort_values("Importance", ascending=False)

    return pipe, metrics, all_prob, importances, X.columns.tolist(), cat_cols, num_cols


def add_risk_columns(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["RiskScore"] = (probabilities * 100).round(1)
    out["RiskCategory"] = pd.cut(
        out["RiskScore"],
        bins=[-0.01, 30, 60, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    ).astype(str)
    out["RecommendedAction"] = np.select(
        [out["RiskScore"] >= 75, out["RiskScore"] >= 60, out["RiskScore"] >= 30],
        ["Immediate HR intervention", "Manager check-in within 7 days", "Monitor monthly"],
        default="Stable / standard engagement",
    )
    return out


def build_excel_download(df: pd.DataFrame, metrics: Dict, importances: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Employee Risk Scores")
        importances.head(30).to_excel(writer, index=False, sheet_name="Top Drivers")
        pd.DataFrame(
            {
                "Metric": ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"],
                "Value": [metrics["auc"], metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]],
            }
        ).to_excel(writer, index=False, sheet_name="Model KPIs")
    return output.getvalue()


# ============================================================
# SIDEBAR DATA LOADING
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="glass-card" style="padding:16px;">
        <div class="eyebrow">Pro Plus</div>
        <h2 style="margin:10px 0 0 0; line-height:1.1;">HR Attrition<br>Intelligence</h2>
        <p style="color:#a7b0c0; font-size:.86rem;">Executive dashboard + ML + action plan</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload IBM HR CSV", type=["csv"], help="Optional. App will also search same folder for the default CSV.")
    local_options = [DEFAULT_FILE, str(Path(__file__).with_name(DEFAULT_FILE)) if "__file__" in globals() else DEFAULT_FILE]

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "📊 EDA Command Center",
            "🧠 ML Performance",
            "🚨 Risk Intelligence",
            "👤 Employee 360 Lookup",
            "💰 ROI & Cost Simulator",
            "📥 Export Center",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with Python · Streamlit · XGBoost · Plotly · Explainable AI")

# Load raw data
try:
    if uploaded is not None:
        raw_df = read_csv_from_bytes(uploaded.getvalue())
    else:
        path_found = None
        for p in local_options:
            if p and os.path.exists(p):
                path_found = p
                break
        if path_found is None:
            st.markdown('<div class="hero"><span class="eyebrow">Dataset Required</span><h1>Upload the HR CSV to start</h1><p>Upload <b>WA_Fn-UseC_-HR-Employee-Attrition.csv</b> from the sidebar, or keep the CSV in the same folder as this app and run again.</p></div>', unsafe_allow_html=True)
            st.stop()
        raw_df = load_local_csv(path_found)

    base_df = enrich_data(raw_df)
    pipe, metrics, all_probs, importances, model_features, cat_cols, num_cols = train_attrition_model(base_df)
    df = add_risk_columns(base_df, all_probs)
except Exception as e:
    st.error("App loading failed. Error details below:")
    st.exception(e)
    st.stop()

# ============================================================
# GLOBAL FILTERS — must run only AFTER df + RiskScore are created
# ============================================================
if "df" not in globals() or df is None or df.empty:
    st.error("Dataset not loaded. Upload the CSV from the sidebar or keep the CSV in the same folder as this app.")
    st.stop()

# Safety: ensure scoring columns always exist before filters/pages
if "RiskScore" not in df.columns:
    df["RiskScore"] = 0.0
if "RiskCategory" not in df.columns:
    df["RiskCategory"] = pd.cut(
        df["RiskScore"],
        bins=[-1, 30, 60, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

with st.sidebar:
    st.markdown("### Global Filters")
    departments = sorted(df["Department"].dropna().astype(str).unique().tolist()) if "Department" in df.columns else []
    selected_depts = st.multiselect("Department", departments, default=departments)

    available_risks = [str(x) for x in df["RiskCategory"].dropna().unique().tolist()] if "RiskCategory" in df.columns else []
    ordered_risks = [r for r in ["High Risk", "Medium Risk", "Low Risk"] if r in available_risks]
    selected_risks = st.multiselect("Risk category", ordered_risks, default=ordered_risks)

    min_score = st.slider("Minimum risk score", 0, 100, 0)

filtered_df = df.copy()
if selected_depts and "Department" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Department"].astype(str).isin(selected_depts)]
if selected_risks and "RiskCategory" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["RiskCategory"].astype(str).isin(selected_risks)]
if "RiskScore" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["RiskScore"].fillna(0) >= min_score]

# ============================================================
# KPI VALUES
# ============================================================
total = len(df)
left = int(df["AttritionBinary"].sum())
stayed = total - left
attrition_rate = left / total * 100 if total else 0
high_risk = int((df["RiskScore"] >= 60).sum())
very_high = int((df["RiskScore"] >= 75).sum())
avg_income_left = df.loc[df["AttritionBinary"] == 1, "MonthlyIncome"].mean() if "MonthlyIncome" in df.columns else 0
annual_cost = left * avg_income_left * 6

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">AI Powered HR Analytics</span>
        <h1>Employee Attrition<br>Intelligence Dashboard</h1>
        <p>Professional HR decision-support system that identifies attrition drivers, predicts employee-level risk, estimates financial impact, and converts analytics into retention actions.</p>
        <div style="margin-top:16px;">
            <span class="pill">🧠 XGBoost ML</span>
            <span class="pill">📊 Executive KPIs</span>
            <span class="pill">🚨 Risk Scoring</span>
            <span class="pill">💰 ROI Simulator</span>
            <span class="pill">📥 Excel Export</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================
if page == "🏠 Executive Overview":
    st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Employees", f"{total:,}", "IBM HR dataset", BLUE)
    with c2: metric_card("Attrition Rate", f"{attrition_rate:.1f}%", f"{left} employees left", ACCENT)
    with c3: metric_card("Model ROC-AUC", f"{metrics['auc']:.3f}", "Ranking power", SAFE)
    with c4: metric_card("High Risk", f"{high_risk:,}", "RiskScore ≥ 60%", AMBER)
    with c5: metric_card("Est. Cost", f"${annual_cost/1e6:.2f}M", "6 months salary basis", ACCENT2)

    st.markdown('<div class="section-title">Business Story</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.05, 1.65])
    with col1:
        fig = px.pie(
            names=["Stayed", "Left"],
            values=[stayed, left],
            hole=0.62,
            color_discrete_sequence=[SAFE, ACCENT],
            title="Attrition Split",
        )
        fig.update_traces(textinfo="percent+label", pull=[0, 0.04])
        st.plotly_chart(plotly_layout(fig, 410), use_container_width=True)

    with col2:
        dept = safe_rate(df, "Department") if "Department" in df.columns else pd.DataFrame()
        fig = px.bar(
            dept.sort_values("AttritionRate"),
            x="AttritionRate",
            y="Department",
            orientation="h",
            text=dept.sort_values("AttritionRate")["AttritionRate"].round(1).astype(str) + "%",
            color="AttritionRate",
            color_continuous_scale=[SAFE, AMBER, ACCENT],
            title="Attrition Rate by Department",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_layout(fig, 410, False), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        insight("<b>Priority 1: Overtime control</b><br>Employees working overtime show a visibly higher exit pattern. Reduce repeated overtime and add workload balancing.", "")
    with c2:
        insight("<b>Priority 2: Early-tenure retention</b><br>First years are the danger zone. Use structured onboarding, mentorship, and 30/60/90-day check-ins.", "amber")
    with c3:
        insight("<b>Priority 3: Targeted salary action</b><br>Low-income + overtime + low satisfaction employees should receive immediate retention review.", "green")

# ============================================================
# PAGE: EDA
# ============================================================
elif page == "📊 EDA Command Center":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏢 Department & Role", "👥 Demographics", "💵 Pay & Overtime", "⏳ Tenure", "🔗 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            dept = safe_rate(df, "Department")
            fig = px.bar(dept, x="Department", y="AttritionRate", text=dept["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Department Attrition Rate")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
        with c2:
            role = safe_rate(df, "JobRole").head(10)
            fig = px.bar(role.sort_values("AttritionRate"), x="AttritionRate", y="JobRole", orientation="h", text=role.sort_values("AttritionRate")["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Top Job Role Risk")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            age = safe_rate(df, "AgeGroup")
            fig = px.bar(age, x="AgeGroup", y="AttritionRate", text=age["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Attrition by Age Group")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
        with c2:
            if "MaritalStatus" in df.columns:
                ms = safe_rate(df, "MaritalStatus")
                fig = px.bar(ms, x="MaritalStatus", y="AttritionRate", text=ms["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Attrition by Marital Status")
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            sal = safe_rate(df, "SalaryBand")
            fig = px.bar(sal, x="SalaryBand", y="AttritionRate", text=sal["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Salary Band Risk")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
        with c2:
            ot = safe_rate(df, "OverTime")
            fig = px.bar(ot, x="OverTime", y="AttritionRate", text=ot["AttritionRate"].round(1).astype(str)+"%", color="AttritionRate", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Overtime Risk")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with tab4:
        tenure = safe_rate(df, "TenureBand")
        fig = px.area(tenure, x="TenureBand", y="AttritionRate", markers=True, title="Attrition by Tenure Band", color_discrete_sequence=[ACCENT])
        st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)
        insight("<b>Action:</b> Build retention programs around early-tenure employees and promotion-stagnant employees. Combine manager coaching + internal mobility + compensation review.", "blue")

    with tab5:
        num_cols_corr = [c for c in ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "AttritionBinary"] if c in df.columns]
        corr = df[num_cols_corr].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(plotly_layout(fig, 560), use_container_width=True)

# ============================================================
# PAGE: ML PERFORMANCE
# ============================================================
elif page == "🧠 ML Performance":
    st.markdown('<div class="section-title">Machine Learning Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("ROC-AUC", f"{metrics['auc']:.3f}", "Higher is better", SAFE)
    with c2: metric_card("Accuracy", f"{metrics['accuracy']:.3f}", "Overall correctness", BLUE)
    with c3: metric_card("Precision", f"{metrics['precision']:.3f}", "Quality of alerts", AMBER)
    with c4: metric_card("Recall", f"{metrics['recall']:.3f}", "Finds leavers", ACCENT)
    with c5: metric_card("F1 Score", f"{metrics['f1']:.3f}", "Balance metric", ACCENT2)

    tab1, tab2, tab3 = st.tabs(["📈 ROC + Confusion", "🔥 Feature Importance", "🧾 Model Report"])
    with tab1:
        c1, c2 = st.columns(2)
        fpr, tpr, _ = metrics["fpr_tpr"]
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy", name=f"AUC {metrics['auc']:.3f}", line=dict(color=ACCENT, width=4)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="rgba(255,255,255,.35)", dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)
        with c2:
            cm = metrics["cm"]
            fig = px.imshow(cm, text_auto=True, x=["Stayed", "Left"], y=["Stayed", "Left"], color_continuous_scale=["rgba(255,255,255,.06)", ACCENT], title="Confusion Matrix")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(plotly_layout(fig, 430, False), use_container_width=True)

    with tab2:
        top = importances.head(18).sort_values("Importance")
        fig = px.bar(top, x="Importance", y="FeatureClean", orientation="h", color="Importance", color_continuous_scale=[SAFE, AMBER, ACCENT], title="Top ML Drivers")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_layout(fig, 570, False), use_container_width=True)
        insight("<b>Interpretation:</b> This importance chart shows which encoded signals the XGBoost model uses most for ranking attrition risk. Use it as direction, not as legal/HR final decision proof.", "blue")

    with tab3:
        st.dataframe(metrics["report"].round(3), use_container_width=True)
        st.markdown("#### Model Input Columns Used")
        st.code(", ".join(model_features), language="text")
        insight("<b>Feature mismatch fixed:</b> RiskScore and RiskCategory are now always removed from model inputs before training/prediction.", "green")

# ============================================================
# PAGE: RISK INTELLIGENCE
# ============================================================
elif page == "🚨 Risk Intelligence":
    st.markdown('<div class="section-title">Employee Risk Intelligence</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Filtered Employees", f"{len(filtered_df):,}", "After sidebar filters", BLUE)
    with c2: metric_card("High Risk", f"{(filtered_df['RiskScore']>=60).sum():,}", "Score ≥ 60", ACCENT)
    with c3: metric_card("Very High", f"{(filtered_df['RiskScore']>=75).sum():,}", "Score ≥ 75", AMBER)
    with c4: metric_card("Avg Risk", f"{filtered_df['RiskScore'].mean():.1f}%" if len(filtered_df) else "0%", "Filtered average", SAFE)

    c1, c2 = st.columns([1.7, 1])
    with c1:
        fig = px.scatter(
            filtered_df,
            x="MonthlyIncome" if "MonthlyIncome" in filtered_df.columns else filtered_df.index,
            y="RiskScore",
            color="RiskCategory",
            size="Age" if "Age" in filtered_df.columns else None,
            hover_data=[c for c in ["EmployeeNumber", "Department", "JobRole", "Age", "OverTime", "YearsAtCompany", "RecommendedAction"] if c in filtered_df.columns],
            color_discrete_map={"High Risk": ACCENT, "Medium Risk": AMBER, "Low Risk": SAFE},
            title="Risk Score vs Monthly Income",
        )
        fig.add_hline(y=60, line_dash="dash", line_color=AMBER, annotation_text="High Risk Threshold")
        fig.add_hline(y=75, line_dash="dot", line_color=ACCENT, annotation_text="Immediate Action")
        st.plotly_chart(plotly_layout(fig, 500), use_container_width=True)
    with c2:
        fig = px.histogram(filtered_df, x="RiskScore", color="RiskCategory", nbins=24, color_discrete_map={"High Risk": ACCENT, "Medium Risk": AMBER, "Low Risk": SAFE}, title="Risk Distribution")
        st.plotly_chart(plotly_layout(fig, 500), use_container_width=True)

    st.markdown("#### Actionable Employee List")
    show_cols = [c for c in ["EmployeeNumber", "Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "YearsAtCompany", "JobSatisfaction", "WorkLifeBalance", "RiskScore", "RiskCategory", "RecommendedAction", "Attrition"] if c in filtered_df.columns]
    table = filtered_df[show_cols].sort_values("RiskScore", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, height=460)
    st.download_button("⬇️ Download filtered risk list CSV", table.to_csv(index=False).encode("utf-8"), "hr_attrition_filtered_risk_list.csv", "text/csv")

# ============================================================
# PAGE: EMPLOYEE LOOKUP
# ============================================================
elif page == "👤 Employee 360 Lookup":
    st.markdown('<div class="section-title">Employee 360 Risk Lookup</div>', unsafe_allow_html=True)
    emp_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None
    if emp_col:
        emp_id = st.selectbox("Select employee", sorted(df[emp_col].tolist()))
        emp = df[df[emp_col] == emp_id].iloc[0]
    else:
        idx = st.selectbox("Select row", list(range(len(df))))
        emp = df.iloc[idx]
        emp_id = idx

    risk = float(emp["RiskScore"])
    risk_color = ACCENT if risk >= 60 else AMBER if risk >= 30 else SAFE
    risk_label = "HIGH RISK" if risk >= 60 else "MEDIUM RISK" if risk >= 30 else "LOW RISK"

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"""
        <div class="risk-big">
            <div class="metric-label">Employee #{emp_id}</div>
            <div class="risk-number" style="color:{risk_color};">{risk:.1f}%</div>
            <div style="font-size:1.2rem; font-weight:900; color:{risk_color};">{risk_label}</div>
            <div style="color:#a7b0c0; margin-top:8px;">{emp.get('RecommendedAction', '')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        pcols = [c for c in ["Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "YearsAtCompany", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "StockOptionLevel", "MaritalStatus", "DistanceFromHome"] if c in df.columns]
        profile = pd.DataFrame({"Field": pcols, "Value": [emp[c] for c in pcols]})
        st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown("#### Risk Factor Explanation")
    factors = []
    if emp.get("OverTime", "No") == "Yes": factors.append(("Overtime", "Repeated overtime is a strong retention warning signal."))
    if emp.get("MonthlyIncome", 999999) < 6000: factors.append(("Lower pay band", "Compensation review may reduce exit risk."))
    if emp.get("Age", 99) <= 35: factors.append(("Early career", "Needs growth path, mentoring, and internal mobility."))
    if emp.get("JobSatisfaction", 4) <= 2: factors.append(("Low job satisfaction", "Manager check-in and work redesign recommended."))
    if emp.get("WorkLifeBalance", 4) <= 2: factors.append(("Poor work-life balance", "Workload and flexibility intervention recommended."))
    if emp.get("YearsAtCompany", 99) <= 2: factors.append(("Low tenure", "Onboarding and retention touchpoints are important."))

    if factors:
        for name, desc in factors:
            insight(f"<b>{name}</b><br>{desc}", "")
    else:
        insight("<b>No major rule-based risk factor found.</b><br>Continue standard engagement and growth planning.", "green")

# ============================================================
# PAGE: ROI
# ============================================================
elif page == "💰 ROI & Cost Simulator":
    st.markdown('<div class="section-title">Retention ROI & Cost Simulator</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        months_cost = st.slider("Replacement cost as months of salary", 3, 18, 6)
        reduction = st.slider("Target attrition reduction", 5, 60, 20)
        program_cost = st.number_input("Retention program budget ($)", min_value=0, value=250000, step=25000)
    avg_salary = avg_income_left if avg_income_left else 0
    current_cost = left * avg_salary * months_cost
    savings = current_cost * reduction / 100
    net_roi = savings - program_cost
    roi_pct = (net_roi / program_cost * 100) if program_cost else 0
    with c2:
        cc1, cc2, cc3 = st.columns(3)
        with cc1: metric_card("Current Cost", f"${current_cost/1e6:.2f}M", "Annual estimate", ACCENT)
        with cc2: metric_card("Potential Savings", f"${savings/1e6:.2f}M", f"{reduction}% reduction", SAFE)
        with cc3: metric_card("Net ROI", f"{roi_pct:.0f}%" if program_cost else "∞", f"Net ${net_roi:,.0f}", AMBER)

    fig = go.Figure(go.Waterfall(
        x=["Current cost", "Savings", "Program budget", "Net impact"],
        y=[current_cost, -savings, program_cost, net_roi],
        measure=["absolute", "relative", "relative", "total"],
        text=[f"${current_cost/1e6:.2f}M", f"-${savings/1e6:.2f}M", f"${program_cost/1e6:.2f}M", f"${net_roi/1e6:.2f}M"],
        textposition="outside",
        decreasing={"marker": {"color": SAFE}},
        increasing={"marker": {"color": ACCENT}},
        totals={"marker": {"color": AMBER}},
    ))
    fig.update_layout(title="Retention Business Case Waterfall", showlegend=False)
    st.plotly_chart(plotly_layout(fig, 470, False), use_container_width=True)

# ============================================================
# PAGE: EXPORT
# ============================================================
elif page == "📥 Export Center":
    st.markdown('<div class="section-title">Export Center</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Full scored dataset CSV", df.to_csv(index=False).encode("utf-8"), "hr_attrition_scored_full.csv", "text/csv", use_container_width=True)
    with c2:
        high = df[df["RiskScore"] >= 60].sort_values("RiskScore", ascending=False)
        st.download_button("⬇️ High-risk employees CSV", high.to_csv(index=False).encode("utf-8"), "high_risk_employees.csv", "text/csv", use_container_width=True)
    with c3:
        excel_bytes = build_excel_download(df, metrics, importances)
        st.download_button("⬇️ Executive Excel Pack", excel_bytes, "hr_attrition_executive_pack.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("#### What was fixed / upgraded")
    insight("<b>Feature mismatch fixed:</b> Generated columns like RiskScore and RiskCategory are excluded from ML input automatically.", "green")
    insight("<b>No hardcoded Mac path dependency:</b> Upload CSV in sidebar or keep dataset in same folder as app.", "green")
    insight("<b>Professional theme:</b> Glass UI, executive KPIs, interactive Plotly visuals, risk dashboard, employee 360 lookup, ROI simulator, and Excel export.", "blue")
    insight("<b>Portfolio ready:</b> This version is suitable for GitHub, Streamlit Cloud, and interview demonstration.", "amber")

st.caption("HR Attrition Intelligence Pro Plus · For analytics portfolio and HR decision-support demonstration. Predictions should support, not replace, human HR judgement.")
