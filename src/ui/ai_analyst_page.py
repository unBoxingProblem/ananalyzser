from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from uuid import uuid4

from src.ai.analyst_service import AnalystError, analyze_question, configured_model, resolve_api_key
from src.ai.context_builder import sensitive_columns
from src.dataset.dataset_registry import DatasetRegistry
from src.dashboard.chart_service import DashboardChartError, build_dashboard_chart, validate_chart_config
from src.dashboard.dashboard_store import DashboardStore

logger = logging.getLogger(__name__)


def _render_result(message: dict[str, Any], registry: DatasetRegistry) -> None:
    if message.get("error"):
        st.error(message.get("content", "The analysis could not be completed."))
    else:
        st.markdown(message.get("content", ""))
    if message.get("operations"):
        with st.expander("Analysis operations", expanded=False):
            for operation in message["operations"]:
                if operation == "build_dashboard":
                    st.write("✓ Built and saved an interactive dashboard")
                elif operation == "recommend_dashboard":
                    st.write("✓ Measured dataset patterns and ranked chart choices")
                else:
                    st.write(f"✓ Executed with Pandas: {operation}")
    for index, result in enumerate(message.get("results", [])):
        if result.get("operation") == "dataset_overview":
            columns = st.columns(3)
            columns[0].metric("Rows", f"{result['rows']:,}")
            columns[1].metric("Columns", f"{result['columns']:,}")
            columns[2].metric("Duplicate rows", f"{result['duplicate_rows']:,}")
            continue
        if result.get("operation") == "dashboard_created":
            st.success(f"Built and saved **{result['dashboard_name']}** with {result['chart_count']} interactive chart(s).")
            st.caption("The charts are shown here now. Use ‘Edit chart settings’ to open their live controls in Dashboard Builder below.")
            if result.get("planning_notes"):
                with st.expander("Why these charts were selected", expanded=False):
                    for note in result["planning_notes"]:
                        st.write(f"• {note}")
            for chart_config in result.get("charts", []):
                st.markdown(f"#### {chart_config.get('title') or chart_config.get('chart_type', 'Chart')}")
                dataset_id = chart_config.get("dataset_id")
                if dataset_id not in {item.dataset_id for item in registry.list_datasets()}:
                    st.warning("The chart's source dataset is no longer loaded in this session.")
                    continue
                dataframe = registry.get_dataframe(dataset_id)
                issue = validate_chart_config(dataframe, chart_config)
                if issue:
                    st.warning(issue)
                    continue
                try:
                    figure = build_dashboard_chart(dataframe, chart_config)
                    chart_id = chart_config.get("chart_id", "chart")
                    st.plotly_chart(figure, use_container_width=True,
                                    key=f"ai_built_dashboard_{message.get('message_id')}_{chart_id}")
                except DashboardChartError as exc:
                    st.warning(str(exc))
                if st.button("Edit chart settings", key=f"ai_edit_dashboard_chart_{message.get('message_id')}_{chart_config.get('chart_id')}"):
                    st.session_state.active_dashboard_id = result["dashboard_id"]
                    st.session_state.dashboard_editor = {"editor_id": str(uuid4()), "mode": "edit", "config": dict(chart_config)}
                    st.rerun()
            continue
        if result.get("operation") == "dashboard_recommendations":
            with st.expander("Dashboard pattern analysis", expanded=False):
                st.caption(f"Reviewed {result.get('profiled_columns', 0)} columns across {result.get('rows', 0):,} rows.")
                for insight in result.get("insights", []):
                    st.write(f"• {insight}")
                candidates = result.get("suggested_charts", [])
                if candidates:
                    st.dataframe(pd.DataFrame([{"Chart": item["chart"].get("title", item["chart"]["chart_type"]),
                                                "Fit score": round(float(item["score"]), 2),
                                                "Measured reason": item["rationale"]} for item in candidates]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No useful chart could be recommended from the current columns.")
            continue
        with st.expander(result.get("title", "Calculated result"), expanded=True):
            if "records" in result:
                table = pd.DataFrame(result["records"], columns=result.get("columns"))
                st.dataframe(table, use_container_width=True, hide_index=True)
                _render_result_chart(result, table, index)
                if result.get("operation") in {"keyword_frequency_by_group", "distinctive_keywords_by_group", "text_length_analysis"}:
                    st.download_button(
                        "Download calculated table as CSV",
                        data=table.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"datapilot_{result['operation']}.csv",
                        mime="text/csv",
                        key=f"ai_download_{message.get('message_id')}_{index}",
                    )
                if result.get("frequency_definition"):
                    st.caption(result["frequency_definition"])
                if result.get("ranking_note"):
                    st.caption(result["ranking_note"])
            else:
                simple = {key: value for key, value in result.items() if key not in {"operation", "title", "chart"}}
                st.json(simple)
    suggestions = message.get("suggested_questions", [])
    if suggestions:
        st.caption("Suggested follow-up")
        for index, suggestion in enumerate(suggestions[:3]):
            if st.button(suggestion, key=f"ai_followup_{message.get('message_id')}_{index}"):
                st.session_state.ai_pending_prompt = suggestion
                st.rerun()


def _render_result_chart(result: dict[str, Any], table: pd.DataFrame, index: int) -> None:
    chart = result.get("chart") or {}
    kind = chart.get("kind")
    try:
        if kind in {"bar", "line"} and chart.get("x") in table and chart.get("y") in table:
            figure = (px.line if kind == "line" else px.bar)(table, x=chart["x"], y=chart["y"], title=result.get("title"))
            st.plotly_chart(figure, use_container_width=True, key=f"ai_result_chart_{index}_{result['operation']}")
        elif kind == "grouped_bar" and chart.get("x") in table and chart.get("y") in table:
            color = chart.get("color")
            chart_table = table
            top_keywords = chart.get("top_keywords")
            if top_keywords and chart["x"] == "keyword":
                chart_table = table[table[chart["x"]].isin(top_keywords)]
            figure = px.bar(chart_table, x=chart["x"], y=chart["y"], color=color if color in chart_table else None,
                            barmode="group", title=result.get("title"),
                            hover_data=[column for column in chart_table if column not in {chart["x"], chart["y"], color}])
            st.plotly_chart(figure, use_container_width=True, key=f"ai_result_chart_{index}_{result['operation']}")
        elif kind == "heatmap":
            columns = [column for column in chart.get("columns", []) if column in table.columns]
            if columns:
                matrix = table.set_index("column")[columns]
                figure = px.imshow(matrix, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Correlation matrix")
                st.plotly_chart(figure, use_container_width=True, key=f"ai_result_chart_{index}_{result['operation']}")
    except Exception:
        logger.exception("Could not display an AI analysis chart preview.")


def _process_pending(dataframe: pd.DataFrame, dataset_name: str, dataset_id: str, sensitive: list[Any],
                     api_key: str | None, dashboard_store: DashboardStore) -> None:
    prompt = st.session_state.get("ai_pending_prompt")
    if not prompt:
        return
    acknowledged = bool(st.session_state.get(f"ai_sensitive_ack_{dataset_id}", False))
    if api_key and sensitive and not acknowledged:
        st.warning("This dataset has potentially sensitive columns. Review the warning above and confirm before DataPilot sends a compact, redacted context to Gemini.")
        if not st.button("Send sanitized context to Gemini", key=f"ai_confirm_send_{dataset_id}", type="primary"):
            return
        # The confirmation button authorizes this pending request. Do not mutate
        # the checkbox's session state after Streamlit has instantiated it.
    history_key = f"ai_messages_{dataset_id}"
    history = st.session_state.setdefault(history_key, [])
    st.session_state.ai_pending_prompt = None
    history.append({"role": "user", "content": prompt[:2000], "message_id": f"user-{len(history)}"})
    conversation = [{"role": item["role"], "content": item["content"][:600]} for item in history[-8:]]
    try:
        with st.spinner("Profiling the dataset and preparing a Pandas analysis…" if api_key else "Calculating dataset results with Pandas…"):
            plan, results, explanation, used_gemini = analyze_question(
                dataframe, prompt, dataset_name, api_key=api_key, model=configured_model(),
                history=conversation[:-1], is_cleaned=st.session_state.get("ai_dataset_cleaned", False),
                dataset_id=dataset_id,
                dashboard_store=dashboard_store,
            )
        for result in results:
            if result.get("operation") == "dashboard_created":
                st.session_state.active_dashboard_id = result["dashboard_id"]
        history.append({"role": "assistant", "content": explanation.answer,
                        "results": results, "operations": [item.operation for item in plan.operations],
                        "suggested_questions": explanation.suggested_questions,
                        "message_id": f"assistant-{len(history)}", "mode": "Gemini" if used_gemini else "Local Pandas"})
    except AnalystError as exc:
        history.append({"role": "assistant", "content": str(exc), "error": True,
                        "message_id": f"assistant-error-{len(history)}"})
    except Exception:
        logger.exception("Unexpected AI Analyst failure for dataset %s", dataset_id)
        history.append({"role": "assistant", "content": "DataPilot could not finish this analysis. Check the dataset and configuration, then try again.",
                        "error": True, "message_id": f"assistant-error-{len(history)}"})
    st.rerun()


def render_ai_analyst(registry: DatasetRegistry, current_dataset_id: str | None) -> None:
    st.divider()
    st.subheader("Chat with Gemini")
    st.caption("Ask questions about the loaded dataset in plain language. Gemini selects analysis tools; Pandas calculates the results.")
    datasets = registry.list_datasets()
    if not datasets:
        st.info("Load a dataset before starting an analysis conversation.")
        return
    ids = [item.dataset_id for item in datasets]
    if current_dataset_id not in ids:
        current_dataset_id = ids[0]
    if st.session_state.get("ai_dataset_id") not in ids or st.session_state.get("ai_dataset_id") != current_dataset_id:
        st.session_state.ai_dataset_id = current_dataset_id
    dataset_id = st.selectbox("AI Analyst dataset", ids, key="ai_dataset_id",
                              format_func=lambda value: next(item.filename for item in datasets if item.dataset_id == value))
    metadata = registry.get_metadata(dataset_id)
    dataframe = registry.get_dataframe(dataset_id)
    dashboard_store = st.session_state.get("dashboard_store")
    if dashboard_store is None:
        dashboard_store = DashboardStore()
        st.session_state.dashboard_store = dashboard_store
    api_key = resolve_api_key()
    sensitive = sensitive_columns(dataframe)
    if api_key:
        if sensitive:
            st.warning("Potentially sensitive data detected in: " + ", ".join(map(str, sensitive)) + ". Sensitive field values are redacted from samples, but schema names and calculated summaries may still be sent to Gemini.")
            st.checkbox("I understand and want to send this limited, redacted dataset context to Gemini.", key=f"ai_sensitive_ack_{dataset_id}")
        else:
            st.info("On submit, DataPilot sends a compact schema, Pandas-calculated summaries, and at most five sanitized sample rows to Gemini. It does not send the full dataset.")
        st.caption(f"Model: {configured_model()}")
    else:
        st.info("Gemini API key is not configured. AI Analyst is using the visible local, rule-based planner; Pandas still performs all calculations.")

    messages = st.session_state.setdefault(f"ai_messages_{dataset_id}", [])
    if st.button("Clear this dataset chat", key=f"ai_clear_{dataset_id}"):
        st.session_state[f"ai_messages_{dataset_id}"] = []
        st.session_state.ai_pending_prompt = None
        st.rerun()
    for message in messages:
        with st.chat_message(message["role"]):
            _render_result(message, registry)
            if message.get("mode"):
                st.caption(f"Mode: {message['mode']}")

    prompt = st.chat_input("Ask DataPilot about this dataset", key=f"ai_chat_input_{dataset_id}")
    if prompt:
        st.session_state.ai_pending_prompt = prompt
    st.session_state.ai_dataset_cleaned = registry.has_cleaning_changes(dataset_id)
    _process_pending(dataframe, metadata.filename, dataset_id, sensitive, api_key, dashboard_store)
