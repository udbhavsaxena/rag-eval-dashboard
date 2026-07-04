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

from src.config import (
    CHUNKER_STRATEGIES,
    EVAL_RESULTS_FILE,
    EVAL_CSV,
    INDEX_PATH,
    TRACES_FILE,
    get_eval_results_file,
    get_index_path,
    get_traces_file,
)

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
def load_results(strategy: str = "word") -> pd.DataFrame | None:
    path = get_eval_results_file(strategy)
    if not path.exists():
        return None
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else None


@st.cache_data(ttl=30)
def load_traces(strategy: str = "word") -> list[dict]:
    path = get_traces_file(strategy)
    if not path.exists():
        return []
    traces = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


@st.cache_data(ttl=30)
def load_all_strategy_results() -> dict[str, pd.DataFrame]:
    """Return {strategy: DataFrame} for every strategy that has results."""
    available: dict[str, pd.DataFrame] = {}
    for strategy in CHUNKER_STRATEGIES:
        df = load_results(strategy)
        if df is not None and not df.empty:
            available[strategy] = df
    return available


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> tuple[str]:
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
- [Chunker Comparison](#chunker-comparison)
- [Failure Analysis](#failure-analysis)
- [Trace Viewer](#trace-viewer)
"""
        )
        st.divider()

        st.markdown("### Active Strategy")
        strategy = st.selectbox(
            "Chunking strategy",
            options=list(CHUNKER_STRATEGIES),
            index=0,
            help="Select which chunking strategy's results to display in Overview / Charts / Traces.",
        )

        st.divider()
        st.markdown("### Quick Setup")
        st.code(
            "# Build index for each strategy\n"
            "python scripts/build_index.py --chunker word\n"
            "python scripts/build_index.py --chunker sentence\n"
            "python scripts/build_index.py --chunker semantic\n\n"
            "# Run eval for each strategy\n"
            "python scripts/run_eval.py --chunker word\n"
            "python scripts/run_eval.py --chunker sentence\n"
            "python scripts/run_eval.py --chunker semantic\n\n"
            "# Compare in terminal\n"
            "python scripts/run_eval.py --compare",
            language="bash",
        )
        st.divider()
        st.info(DISCLAIMER)

    return (strategy,)


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


# ── Chunker comparison section ────────────────────────────────────────────────

def section_chunker_comparison(all_results: dict[str, pd.DataFrame]) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("Chunker Comparison", anchor="chunker-comparison")

    if not all_results:
        st.info(
            "No strategy results found yet.\n\n"
            "Build at least one index and run eval:\n"
            "```bash\n"
            "python scripts/build_index.py --chunker sentence\n"
            "python scripts/run_eval.py --chunker sentence\n"
            "```"
        )
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    metric_cols = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "faithfulness_score"]
    for strategy, df in all_results.items():
        row = {"Strategy": strategy, "Queries": len(df)}
        for col in metric_cols:
            if col in df.columns:
                row[col] = round(df[col].mean(), 4)
        row["Avg Latency (ms)"] = round(df["total_ms"].mean(), 1) if "total_ms" in df.columns else None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Summary Table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if len(all_results) < 2:
        st.caption("Build and run more strategies to see a visual comparison.")
        return

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.subheader("Metric Comparison")
    bar_rows = []
    for strategy, df in all_results.items():
        for col in metric_cols:
            if col in df.columns:
                bar_rows.append({
                    "Strategy": strategy,
                    "Metric": col.replace("_at_k", "").replace("_", " ").title(),
                    "Score": round(df[col].mean(), 4),
                })

    bar_df = pd.DataFrame(bar_rows)
    fig = px.bar(
        bar_df,
        x="Metric",
        y="Score",
        color="Strategy",
        barmode="group",
        height=380,
        color_discrete_map={"word": "#3498db", "sentence": "#2ecc71", "semantic": "#e74c3c"},
    )
    fig.update_layout(yaxis_range=[0, 1.05], margin=dict(t=20), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("Radar Chart")
    radar_metrics = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "faithfulness_score"]
    radar_labels = ["Recall", "Precision", "MRR", "nDCG", "Faithfulness"]
    colors = {"word": "#3498db", "sentence": "#2ecc71", "semantic": "#e74c3c"}

    fig2 = go.Figure()
    for strategy, df in all_results.items():
        values = [
            round(df[m].mean(), 4) if m in df.columns else 0
            for m in radar_metrics
        ]
        values += [values[0]]  # close the polygon
        fig2.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=strategy,
            line_color=colors.get(strategy),
            opacity=0.6,
        ))

    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        margin=dict(t=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Latency comparison ────────────────────────────────────────────────────
    st.subheader("Average Latency per Strategy (ms)")
    lat_rows = [
        {"Strategy": s, "Avg Latency (ms)": round(df["total_ms"].mean(), 1)}
        for s, df in all_results.items()
        if "total_ms" in df.columns
    ]
    if lat_rows:
        lat_df = pd.DataFrame(lat_rows)
        fig3 = px.bar(
            lat_df,
            x="Strategy",
            y="Avg Latency (ms)",
            height=280,
            color="Strategy",
            color_discrete_map=colors,
        )
        fig3.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    (strategy,) = sidebar()

    st.title("🔬 RAG Eval Dashboard")
    st.caption("End-to-end retrieval & faithfulness evaluation over clinical guidelines")
    st.warning(DISCLAIMER)
    st.divider()

    if not check_setup():
        return

    df = load_results(strategy)
    traces = load_traces(strategy)
    all_results = load_all_strategy_results()

    if df is None or df.empty:
        st.warning(
            f"No evaluation results for strategy **'{strategy}'** yet.  \n"
            f"Run: `python scripts/run_eval.py --chunker {strategy}`"
        )
    else:
        section_overview(df)
        st.divider()

        try:
            import plotly.express  # noqa: F401
            section_charts(df)
            st.divider()
        except ImportError:
            st.info("Install plotly (`pip install plotly`) for interactive charts.")

    try:
        import plotly.express  # noqa: F401
        section_chunker_comparison(all_results)
        st.divider()
    except ImportError:
        pass

    if df is not None and not df.empty:
        section_failures(df)
        st.divider()
        section_trace_viewer(traces, df)


if __name__ == "__main__":
    main()
