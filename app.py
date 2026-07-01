"""
RAG Eval Dashboard — Streamlit app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup (allow running from project root) ──────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import EVAL_RESULTS_FILE, TRACES_FILE, EVAL_CSV, INDEX_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Eval Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **Disclaimer:** This project is for RAG evaluation demonstration only "
    "and does **not** provide medical advice. Always consult a qualified healthcare professional."
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_results() -> pd.DataFrame | None:
    if not EVAL_RESULTS_FILE.exists():
        return None
    rows = []
    with EVAL_RESULTS_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else None


@st.cache_data(ttl=30)
def load_traces() -> list[dict]:
    if not TRACES_FILE.exists():
        return []
    traces = []
    with TRACES_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/RAG-Eval%20Dashboard-blueviolet?style=for-the-badge",
            use_column_width=True,
        )
        st.markdown("## Navigation")
        st.markdown(
            """
- [Overview](#overview)
- [Charts](#charts)
- [Failure Analysis](#failure-analysis)
- [Trace Viewer](#trace-viewer)
"""
        )
        st.divider()
        st.markdown("### Quick Setup")
        st.code(
            "python scripts/build_index.py\n"
            "python scripts/sample_eval_set.py --create\n"
            "# Edit data/eval/eval_set.csv\n"
            "python scripts/run_eval.py",
            language="bash",
        )
        st.divider()
        st.info(DISCLAIMER)


# ── Setup check ───────────────────────────────────────────────────────────────

def check_setup() -> bool:
    """Return True if data is available; else show instructions and return False."""
    missing = []
    if not INDEX_PATH.exists():
        missing.append(
            "**FAISS index not found.** Run:\n```bash\npython scripts/build_index.py\n```"
        )
    if not EVAL_RESULTS_FILE.exists():
        missing.append(
            "**Eval results not found.** Run:\n```bash\npython scripts/run_eval.py\n```"
        )

    if missing:
        st.warning("### Setup Required")
        for msg in missing:
            st.markdown(msg)
        st.markdown("---")
        st.markdown(
            "**Full setup steps:**\n"
            "1. Add `.txt` or `.md` guideline files to `data/raw/`\n"
            "2. `python scripts/build_index.py`\n"
            "3. `python scripts/sample_eval_set.py --create`\n"
            "4. Edit `data/eval/eval_set.csv` to fill `relevant_chunk_ids`\n"
            "5. `python scripts/run_eval.py`\n"
            "6. Refresh this page"
        )
        return False
    return True


# ── Overview section ──────────────────────────────────────────────────────────

def section_overview(df: pd.DataFrame) -> None:
    st.header("Overview", anchor="overview")
    k = int(df["recall_at_k"].count())  # number of queries evaluated

    cols = st.columns(6)
    metrics = [
        ("Recall@5", df["recall_at_k"].mean(), "{:.3f}"),
        ("Precision@5", df["precision_at_k"].mean(), "{:.3f}"),
        ("MRR", df["mrr"].mean(), "{:.3f}"),
        ("Faithfulness", df["faithfulness_score"].mean(), "{:.3f}"),
        ("Avg Latency", df["total_ms"].mean(), "{:.0f} ms"),
        ("Total Cost", df["estimated_cost_usd"].sum(), "${:.5f}"),
    ]
    for col, (label, value, fmt) in zip(cols, metrics):
        col.metric(label, fmt.format(value))

    st.caption(f"Evaluated {k} queries — refresh after re-running `run_eval.py`")


# ── Charts section ────────────────────────────────────────────────────────────

def section_charts(df: pd.DataFrame) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("Charts", anchor="charts")

    col1, col2 = st.columns(2)

    # Retrieval metric distribution
    with col1:
        st.subheader("Retrieval Metrics by Query")
        metric_df = df[["query_id", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]].melt(
            id_vars="query_id", var_name="metric", value_name="value"
        )
        fig = px.bar(
            metric_df,
            x="query_id",
            y="value",
            color="metric",
            barmode="group",
            height=350,
            labels={"value": "Score", "query_id": "Query"},
        )
        fig.update_layout(legend_title_text="", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Faithfulness score by query
    with col2:
        st.subheader("Faithfulness Score by Query")
        colors = df["faithfulness_score"].apply(
            lambda s: "#2ecc71" if s >= 0.75 else ("#f39c12" if s >= 0.35 else "#e74c3c")
        )
        fig2 = go.Figure(
            go.Bar(
                x=df["query_id"],
                y=df["faithfulness_score"],
                marker_color=list(colors),
                text=df["verdict"],
                textposition="outside",
            )
        )
        fig2.update_layout(
            yaxis=dict(range=[0, 1.1]),
            height=350,
            margin=dict(t=20),
            xaxis_title="Query",
            yaxis_title="Faithfulness Score",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Latency breakdown
    with col3:
        st.subheader("Latency Breakdown (ms)")
        lat_df = df[["query_id", "retrieval_ms", "generation_ms", "judging_ms"]].melt(
            id_vars="query_id", var_name="stage", value_name="ms"
        )
        fig3 = px.bar(
            lat_df,
            x="query_id",
            y="ms",
            color="stage",
            height=350,
            labels={"ms": "Milliseconds", "query_id": "Query"},
        )
        fig3.update_layout(legend_title_text="", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    # Cost by query
    with col4:
        st.subheader("Estimated Cost per Query (USD)")
        fig4 = px.bar(
            df,
            x="query_id",
            y="estimated_cost_usd",
            height=350,
            labels={"estimated_cost_usd": "Cost (USD)", "query_id": "Query"},
            color_discrete_sequence=["#8e44ad"],
        )
        fig4.update_layout(margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)


# ── Failure analysis section ──────────────────────────────────────────────────

def section_failures(df: pd.DataFrame) -> None:
    st.header("Failure Analysis", anchor="failure-analysis")

    col1, col2, col3 = st.columns(3)

    LOW_RECALL_THRESH = 0.4
    LOW_FAITH_THRESH = 0.5
    HIGH_LAT_THRESH = df["total_ms"].quantile(0.75)

    with col1:
        st.subheader(f"Low Recall (< {LOW_RECALL_THRESH})")
        low_recall = df[df["recall_at_k"] < LOW_RECALL_THRESH][
            ["query_id", "query", "recall_at_k"]
        ].sort_values("recall_at_k")
        if low_recall.empty:
            st.success("No low-recall queries.")
        else:
            st.dataframe(low_recall, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Unsupported / Partial Answers")
        bad_faith = df[df["verdict"].isin(["unsupported", "partially_grounded"])][
            ["query_id", "query", "faithfulness_score", "verdict"]
        ].sort_values("faithfulness_score")
        if bad_faith.empty:
            st.success("All answers are grounded.")
        else:
            st.dataframe(bad_faith, use_container_width=True, hide_index=True)

    with col3:
        st.subheader(f"Slowest Queries (≥ {HIGH_LAT_THRESH:.0f} ms)")
        slow = df[df["total_ms"] >= HIGH_LAT_THRESH][
            ["query_id", "query", "total_ms"]
        ].sort_values("total_ms", ascending=False)
        if slow.empty:
            st.success("No unusually slow queries.")
        else:
            st.dataframe(slow, use_container_width=True, hide_index=True)


# ── Trace viewer section ──────────────────────────────────────────────────────

def section_trace_viewer(traces: list[dict], df: pd.DataFrame) -> None:
    st.header("Trace Viewer", anchor="trace-viewer")

    if not traces:
        st.info("No traces available yet. Run `python scripts/run_eval.py` first.")
        return

    trace_map = {t["query_id"]: t for t in traces}
    query_ids = sorted(trace_map.keys())

    selected_id = st.selectbox("Select Query ID", query_ids)
    trace = trace_map[selected_id]

    # Query row from results for metrics
    row = df[df["query_id"] == selected_id].iloc[0] if selected_id in df["query_id"].values else None

    st.markdown(f"### Query")
    st.info(trace["query"])

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Generated Answer")
        st.markdown(trace["generated_answer"])

        st.markdown("### Faithfulness Judgment")
        fj = trace["faithfulness_judgment"]
        verdict_color = {
            "grounded": "green",
            "partially_grounded": "orange",
            "unsupported": "red",
        }.get(fj.get("verdict", ""), "gray")

        st.markdown(
            f"**Verdict:** :{verdict_color}[{fj.get('verdict', 'N/A').upper()}]  |  "
            f"**Score:** {fj.get('faithfulness_score', 0):.3f}  |  "
            f"**Judge:** `{trace.get('judge_model', 'N/A')}`"
        )
        st.markdown(f"*{fj.get('explanation', '')}*")

        if fj.get("supported_claims"):
            with st.expander(f"Supported claims ({len(fj['supported_claims'])})"):
                for claim in fj["supported_claims"]:
                    st.markdown(f"- ✅ {claim}")

        if fj.get("unsupported_claims"):
            with st.expander(f"Unsupported claims ({len(fj['unsupported_claims'])})", expanded=True):
                for claim in fj["unsupported_claims"]:
                    st.markdown(f"- ❌ {claim}")

    with col_right:
        st.markdown("### Retrieval Metrics")
        rm = trace.get("retrieval_metrics", {})
        for metric, val in [
            ("Recall@K", rm.get("recall_at_k", "—")),
            ("Precision@K", rm.get("precision_at_k", "—")),
            ("MRR", rm.get("mrr", "—")),
            ("nDCG@K", rm.get("ndcg_at_k", "—")),
        ]:
            st.metric(metric, f"{val:.3f}" if isinstance(val, float) else val)

        st.markdown("### Timing & Cost")
        lat = trace.get("latency", {})
        cost = trace.get("cost", {})
        timing_data = {
            "Stage": ["Retrieval", "Reranking", "Generation", "Judging", "**Total**"],
            "ms": [
                lat.get("retrieval_ms", 0),
                lat.get("rerank_ms", 0),
                lat.get("generation_ms", 0),
                lat.get("judging_ms", 0),
                lat.get("total_ms", 0),
            ],
        }
        st.dataframe(
            pd.DataFrame(timing_data),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Tokens: {cost.get('input_tokens', 0)} in / "
            f"{cost.get('output_tokens', 0)} out  |  "
            f"Cost: ${cost.get('estimated_cost_usd', 0):.6f}"
        )

    st.markdown("### Retrieved Chunks")
    chunks_to_show = trace.get("reranked_chunks") or trace.get("retrieved_chunks", [])
    for i, chunk in enumerate(chunks_to_show):
        with st.expander(
            f"Chunk {i+1}: `{chunk['chunk_id']}`  (score={chunk.get('score', 0):.4f})"
        ):
            st.text(chunk.get("text", ""))

    with st.expander("Raw Trace JSON"):
        st.json(trace)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sidebar()

    st.title("🔬 RAG Eval Dashboard")
    st.caption("End-to-end retrieval & faithfulness evaluation over clinical guidelines")
    st.warning(DISCLAIMER)
    st.divider()

    if not check_setup():
        return

    df = load_results()
    traces = load_traces()

    if df is None or df.empty:
        st.warning("No evaluation results yet. Run `python scripts/run_eval.py` first.")
        return

    section_overview(df)
    st.divider()

    try:
        import plotly.express  # noqa: F401
        section_charts(df)
        st.divider()
    except ImportError:
        st.info("Install plotly (`pip install plotly`) for interactive charts.")

    section_failures(df)
    st.divider()
    section_trace_viewer(traces, df)


if __name__ == "__main__":
    main()
