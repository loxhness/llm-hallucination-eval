"""
Streamlit UI for the hallucination-eval project.

Tab 1 — Live Judge   : paste a question, ground truth, and model response;
                       get an instant verdict and reasoning from the LLM judge.
Tab 2 — Dashboard    : load results/scored.csv and explore interactive charts.

Run with:
    streamlit run app.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
_src = ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
load_dotenv(ROOT / ".env")

from score import call_judge                                          # noqa: E402
from analyze import load_scored, compute_summary, compute_calibration_data  # noqa: E402

RESULTS_DIR = ROOT / "results"
JUDGE_MAX_CALLS_PER_SESSION = 5
_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


# ── Plotly chart helpers ──────────────────────────────────────────────────────

def _metric_bar(summary: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Bar or grouped-bar chart for a single metric column."""
    models = summary["model_name"].unique().tolist()
    if len(models) > 1:
        color_map = {m: _COLORS[i % len(_COLORS)] for i, m in enumerate(models)}
        fig = px.bar(
            summary, x="condition", y=col,
            color="model_name", color_discrete_map=color_map,
            barmode="group", title=title,
        )
    else:
        fig = px.bar(
            summary, x="condition", y=col, title=title,
            color_discrete_sequence=[_COLORS[0]],
        )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis_tickangle=-15,
        yaxis=dict(tickformat=".0%", title=""),
        legend_title_text="Model",
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y:.1%}<extra></extra>")
    return fig


def _calibration_fig(cal_data: dict, models: list) -> go.Figure:
    """Reliability diagram: mean predicted confidence vs actual accuracy."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#aaaaaa", width=1),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    for i, model in enumerate(models):
        data = cal_data.get(model)
        if data is None or data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=data["mean_conf"],
            y=data["accuracy"],
            mode="lines+markers",
            name=model,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            marker=dict(size=7),
            customdata=data["count"],
            hovertemplate=(
                "<b>" + model + "</b><br>"
                "Confidence: %{x:.0%}<br>"
                "Accuracy: %{y:.0%}<br>"
                "n = %{customdata}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Calibration Curve",
        height=380,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(title="Mean predicted confidence", tickformat=".0%", range=[-0.02, 1.05]),
        yaxis=dict(title="Actual accuracy", tickformat=".0%", range=[-0.02, 1.05]),
    )
    return fig


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hallucination Eval",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 LLM Hallucination Eval")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Judge configuration")
    st.caption("Used by the Live Judge tab.")

    _default_provider = os.getenv("JUDGE_PROVIDER", "anthropic")
    judge_provider = st.selectbox(
        "Provider",
        ["anthropic", "openai"],
        index=0 if _default_provider == "anthropic" else 1,
    )
    judge_model = st.text_input(
        "Model",
        value=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
    )

    st.divider()
    st.caption(
        "API keys are read from `.env`. "
        "Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` there."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_judge, tab_dash = st.tabs(["Live Judge", "Results Dashboard"])


# ── Tab 1: Live Judge ─────────────────────────────────────────────────────────

with tab_judge:
    if "judge_call_count" not in st.session_state:
        st.session_state.judge_call_count = 0
    if "judge_last_request_time" not in st.session_state:
        st.session_state.judge_last_request_time = None

    st.subheader("Evaluate a single response")
    st.caption(
        "Paste a question, the correct answer, and a model response. "
        "The judge classifies the response as **correct**, **hallucinated**, or **abstained**."
    )
    st.info(
        "**Live API calls:** Each time you click **Run judge**, this tab sends a **real request** "
        "to the judge provider you chose in the sidebar (may incur usage charges)."
    )

    used = st.session_state.judge_call_count
    remaining = max(0, JUDGE_MAX_CALLS_PER_SESSION - used)
    st.caption(
        f"Session limit: **{used} / {JUDGE_MAX_CALLS_PER_SESSION}** judge calls used "
        f"({remaining} remaining)."
    )
    if st.session_state.judge_last_request_time is not None:
        _last = datetime.fromtimestamp(st.session_state.judge_last_request_time)
        st.caption(f"Last judge API request: **{_last.strftime('%Y-%m-%d %H:%M:%S')}** (local time).")

    if used >= JUDGE_MAX_CALLS_PER_SESSION:
        st.warning(
            "You've reached the **session limit** of "
            f"{JUDGE_MAX_CALLS_PER_SESSION} live judge calls. "
            "Open this app in a new browser session or wait for the session to reset to run more."
        )

    col_l, col_r = st.columns(2)
    with col_l:
        question = st.text_area(
            "Question", height=100,
            placeholder="e.g. What is the capital of France?",
        )
        ground_truth = st.text_input(
            "Ground truth answer",
            placeholder="e.g. Paris",
        )
    with col_r:
        model_response = st.text_area(
            "Model response to evaluate", height=150,
            placeholder="Paste the model's response here…",
        )

    ready = bool(question and ground_truth and model_response)
    under_limit = st.session_state.judge_call_count < JUDGE_MAX_CALLS_PER_SESSION
    if st.button("Run judge", type="primary", disabled=not ready or not under_limit):
        with st.spinner("Calling judge…"):
            v, r = call_judge(
                question=question,
                expected=ground_truth,
                model_answer=model_response,
                provider=judge_provider,
                model=judge_model,
            )
        st.session_state.judge_call_count += 1
        st.session_state.judge_last_request_time = time.time()
        st.session_state["_verdict"] = v
        st.session_state["_reason"] = r

    verdict = st.session_state.get("_verdict")
    reason = st.session_state.get("_reason")
    if verdict is not None:
        st.divider()
        if verdict == "correct":
            st.success("**Verdict: Correct** ✓")
        elif verdict == "hallucinated":
            st.error("**Verdict: Hallucinated** ✗")
        else:
            st.warning("**Verdict: Abstained** ∼")
        st.markdown(f"**Reasoning:** {reason}")


# ── Tab 2: Results Dashboard ──────────────────────────────────────────────────

with tab_dash:
    st.subheader("Results dashboard")

    scored_path = RESULTS_DIR / "scored.csv"

    if not scored_path.exists():
        st.info(
            "No results found. Generate them first:\n\n"
            "```\npython eval.py --all-conditions\n```"
        )
        st.stop()

    # ── load data ─────────────────────────────────────────────────────────────
    mtime = datetime.fromtimestamp(scored_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Loaded `{scored_path.relative_to(ROOT)}` · last modified {mtime}")

    df = load_scored(scored_path)
    summary = compute_summary(df)
    cal_data = compute_calibration_data(df)

    models = summary["model_name"].unique().tolist()

    # ── Brier score metrics ───────────────────────────────────────────────────
    model_brier = (
        summary.groupby("model_name")["brier_score"]
        .mean()
        .dropna()
        .sort_values()
    )
    if not model_brier.empty:
        brier_cols = st.columns(len(model_brier))
        for col, (model, score) in zip(brier_cols, model_brier.items()):
            col.metric(
                label=f"Brier Score — {model}",
                value=f"{score:.4f}",
                help="Mean squared error between self-reported confidence and actual correctness. Lower = better calibrated. Range: 0 (perfect) → 1 (worst).",
            )
        st.divider()

    # ── Three metric charts ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(
        _metric_bar(summary, "accuracy", "Accuracy by Condition"),
        use_container_width=True,
    )
    c2.plotly_chart(
        _metric_bar(summary, "hallucination_rate", "Hallucination Rate by Condition"),
        use_container_width=True,
    )
    c3.plotly_chart(
        _metric_bar(summary, "abstain_rate", "Abstain Rate by Condition"),
        use_container_width=True,
    )

    # ── Calibration curve + summary table ─────────────────────────────────────
    st.divider()
    cal_col, tbl_col = st.columns([1, 1])

    with cal_col:
        st.plotly_chart(_calibration_fig(cal_data, models), use_container_width=True)

    with tbl_col:
        st.caption("Per-condition breakdown")
        _display_cols = {
            "model_name": "Model",
            "condition": "Condition",
            "accuracy": "Accuracy",
            "hallucination_rate": "Halluc.",
            "abstain_rate": "Abstain",
            "brier_score": "Brier",
            "n": "N",
        }
        display = (
            summary[[c for c in _display_cols if c in summary.columns]]
            .rename(columns=_display_cols)
            .reset_index(drop=True)
        )
        for pct_col in ("Accuracy", "Halluc.", "Abstain"):
            if pct_col in display.columns:
                display[pct_col] = display[pct_col].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else ""
                )
        if "Brier" in display.columns:
            display["Brier"] = display["Brier"].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
        st.dataframe(display, use_container_width=True, hide_index=True)
