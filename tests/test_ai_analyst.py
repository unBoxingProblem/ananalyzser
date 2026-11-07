import pandas as pd
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from src.ai.analyst_service import (
    AnalysisPlan, AnalystError, analyze_question, execute_plan, rule_based_plan, validate_plan,
)
from src.ai.context_builder import build_dataset_context, sensitive_columns
from src.dashboard.dashboard_store import DashboardStore
from src.dataset.dataset_registry import DatasetRegistry
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


def test_context_is_compact_and_masks_sensitive_columns_and_values():
    frame = pd.DataFrame({
        "customer_email": ["one@example.com", "two@example.com"],
        "measurement": [10, 20],
        "state": ["AA", "BB"],
    })
    context = build_dataset_context(frame, "records.csv", is_cleaned=True)
    assert context["rows"] == 2
    assert context["cleaning_state"].startswith("current working data")
    assert "customer_email" in sensitive_columns(frame)
    assert context["sample_rows"][0]["customer_email"] == "[redacted: sensitive-looking column]"
    assert "one@example.com" not in str(context)
    assert context["statistics"]["customer_email"]["frequencies"].startswith("omitted")


def test_context_limits_samples_and_schema_size():
    frame = pd.DataFrame({f"field_{i}": range(100) for i in range(50)})
    context = build_dataset_context(frame, "large.csv", sample_rows=100, max_columns=50)
    assert len(context["sample_rows"]) <= 5
    assert len(context["schema"]) <= 30


def test_pandas_executes_whitelisted_operations_with_actual_results():
    frame = pd.DataFrame({"group": ["A", "A", "B"], "amount": [10, 20, 50], "optional": [1, None, 3]})
    plan = AnalysisPlan.model_validate({"operations": [
        {"operation": "group_aggregate", "group_by": "group", "column": "amount", "aggregation": "sum"},
        {"operation": "missing_values", "column": "optional"},
        {"operation": "duplicates"},
    ]})
    results = execute_plan(frame, plan)
    grouped = results[1]["records"]
    assert {row["group"]: row["amount"] for row in grouped} == {"B": 50, "A": 30}
    assert results[2]["records"][0]["missing"] == 1
    assert results[3]["duplicate_rows"] == 0


def test_operation_schema_rejects_arbitrary_code_and_invalid_columns():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate({"operations": [{"operation": "exec", "code": "print(1)"}]})
    frame = pd.DataFrame({"metric": [1, 2, 3]})
    plan = AnalysisPlan.model_validate({"operations": [{"operation": "outliers", "column": "missing_metric"}]})
    with pytest.raises(AnalystError, match="unavailable"):
        validate_plan(plan, frame)


def test_rule_based_analyst_produces_pandas_results_without_api_key():
    frame = pd.DataFrame({"segment": ["A", "A", "B", "B"], "metric": [1, 2, 8, 10], "other": [3, 4, 5, 6]})
    plan = rule_based_plan("Analyze this dataset completely", frame)
    assert {operation.operation for operation in plan.operations} >= {"missing_values", "duplicates", "describe_numeric"}
    _, results, explanation, used_gemini = analyze_question(frame, "Analyze this dataset completely", "sample.csv", api_key=None)
    assert not used_gemini
    assert len(results) > 1
    assert "Pandas" in explanation.answer


def test_keyword_frequency_by_group_uses_pandas_counts_and_frequency_per_thousand():
    frame = pd.DataFrame({
        "passage": ["Blue blue sky, and river.", "Blue leaf in sky.", "Red red red road."],
        "source": ["A", "A", "B"],
    })
    plan = rule_based_plan("Find which keywords occur most frequently in each source", frame)
    assert plan.operations[0].operation == "keyword_frequency_by_group"
    assert plan.operations[0].text_column == "passage"
    assert plan.operations[0].group_by == "source"
    result = execute_plan(frame, plan)[1]
    rows = {(row["source"], row["keyword"]): row for row in result["records"]}
    assert rows[("A", "blue")]["occurrences"] == 3
    assert rows[("A", "blue")]["frequency_per_1000_tokens"] == pytest.approx(3 / 7 * 1000)
    assert rows[("B", "red")]["occurrences"] == 3
    assert "and" not in {row["keyword"] for row in result["records"]}
    assert frame.loc[0, "passage"] == "Blue blue sky, and river."


def test_distinctive_keyword_analysis_ranks_by_group_tfidf():
    frame = pd.DataFrame({
        "document": ["shared shared amber amber", "shared shared amber", "shared shared cobalt", "shared cobalt"],
        "cohort": ["A", "A", "B", "B"],
    })
    plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "distinctive_keywords_by_group", "text_column": "document", "group_by": "cohort", "top_n": 2,
    }]})
    result = execute_plan(frame, plan)[1]
    rows = {(row["cohort"], row["keyword"]): row for row in result["records"]}
    assert rows[("A", "amber")]["distinctiveness_tfidf"] > 0
    assert rows[("A", "shared")]["distinctiveness_tfidf"] < rows[("A", "amber")]["distinctiveness_tfidf"]
    assert "TF-IDF" in result["ranking_note"]


def test_keyword_tool_rejects_missing_columns():
    frame = pd.DataFrame({"content": ["small text"], "label": ["x"]})
    plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "keyword_frequency_by_group", "text_column": "content", "group_by": "missing",
    }]})
    with pytest.raises(AnalystError, match="unavailable"):
        execute_plan(frame, plan)


def test_local_keyword_question_executes_and_returns_actual_results():
    frame = pd.DataFrame({
        "response": ["copper copper bright", "copper bright", "violet violet bright"],
        "candidate": ["one", "one", "two"],
    })
    plan, results, explanation, used_gemini = analyze_question(
        frame, "Which keywords occur most frequently in each candidate?", "tiny.csv", api_key=None,
    )
    assert not used_gemini
    assert plan.operations[0].operation == "keyword_frequency_by_group"
    assert {row["candidate"]: row["keyword"] for row in results[1]["records"] if row["keyword"] in {"copper", "violet"}} == {"one": "copper", "two": "violet"}
    assert "calculated table" in explanation.answer


def test_broad_insight_request_runs_argument_free_pattern_discovery():
    frame = pd.DataFrame({
        "id": [f"i{i}" for i in range(6)],
        "prompt_id": [f"p{i % 2}" for i in range(6)],
        "text": ["amber amber bright river passage", "amber bright river passage", "amber amber calm river passage",
                 "blue blue quiet engine passage", "blue quiet engine passage", "blue blue loud engine passage"],
        "model": ["model-a"] * 3 + ["model-b"] * 3,
    })
    plan = rule_based_plan("What insights and patterns do you see? What makes them different?", frame)
    assert [operation.operation for operation in plan.operations] == ["discover_insights"]
    result = execute_plan(frame, plan)[1]
    text_findings = [row for row in result["records"] if row["pattern"] == "Text pattern"]
    assert {row["columns"] for row in text_findings} == {"text, model"}
    assert any("amber" in row["finding"] for row in text_findings)
    assert any("blue" in row["finding"] for row in text_findings)


def test_general_insight_tool_ignores_incomplete_operation_guesses_and_scans_patterns():
    from src.ai.analyst_service import GeminiAnalyst
    frame = pd.DataFrame({"kind": ["A", "A", "B"], "value": [1, 2, 8]})
    analyst = GeminiAnalyst("unused")
    payload, operations, results = analyst._execute_tool(
        "execute_dataframe_analysis", {"operations": [
            {"operation": "count_categories"},
            {"operation": "group_aggregate", "group_by": "kind", "aggregation": "mean"},
        ]}, frame,
        "sample.csv", "dataset-1", "What are the insights in this data?",
    )
    assert [operation.operation for operation in operations] == ["discover_insights"]
    insight_result = next(item for item in results if item["operation"] == "discover_insights")
    assert insight_result["records"]
    assert payload["operations"][0] == "discover_insights"


def test_gemini_native_tool_result_is_returned_for_followup_explanation(monkeypatch, caplog):
    from types import SimpleNamespace
    from src.ai.analyst_service import GeminiAnalyst
    caplog.set_level("INFO")

    frame = pd.DataFrame({
        "text": ["amber amber glowing passage for this candidate", "amber glowing passage for this candidate",
                 "blue blue glowing passage for this candidate", "blue glowing passage for this candidate"],
        "model": ["A", "A", "B", "B"], "id": ["id1", "id2", "id3", "id4"],
        "prompt_id": ["p1", "p1", "p1", "p1"],
    })

    class Stub:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def from_function_response(cls, *, name, response):
            return cls(function_response=SimpleNamespace(name=name, response=response))

    call = SimpleNamespace(name="execute_dataframe_analysis", args={"operations": [{
        "operation": "keyword_frequency_by_group", "text_column": "text", "grouping_column": "model", "top_n": 5,
    }]})
    first = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=call)]))])
    second = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Pandas results show amber leads model A and blue leads model B.")]))])
    sent_contents = []

    class Models:
        index = 0
        def generate_content(self, **kwargs):
            sent_contents.append(list(kwargs["contents"]))
            self.index += 1
            return [first, second][self.index - 1]

    client = SimpleNamespace(models=Models(), close=lambda: None)
    types = SimpleNamespace(
        HttpOptions=Stub, FunctionDeclaration=Stub, Tool=Stub, AutomaticFunctionCallingConfig=Stub,
        GenerateContentConfig=Stub, Content=Stub, Part=Stub,
    )
    monkeypatch.setattr("src.ai.analyst_service._load_genai_sdk", lambda: (SimpleNamespace(Client=lambda **kwargs: client), types))
    plan, results, explanation = GeminiAnalyst("unused").analyze(
        "Find frequent keywords in each model", frame, {"dataset_name": "train.csv"}, [], dataset_id="dataset-1",
    )
    assert plan.operations[0].operation == "keyword_frequency_by_group"
    assert {(item["model"], item["keyword"], item["occurrences"]) for item in results[1]["records"]} >= {
        ("A", "amber", 3), ("B", "blue", 3),
    }
    assert "amber leads model A" in explanation.answer
    feedback = sent_contents[1][-1].parts[0].function_response.response
    assert feedback["operations"] == ["keyword_frequency_by_group"]
    assert any(row["keyword"] == "amber" for row in feedback["results"][1]["records"])
    assert 'dataset_id=dataset-1' in caplog.text
    assert 'selected_group_column=[\'model\']' in caplog.text
    assert '"grouping_column": "model"' in caplog.text


def test_dataset_overview_request_includes_actual_schema_profile():
    frame = pd.DataFrame({"item": ["pen", "book", None], "units": [2, 5, 8]})
    plan, results, explanation, used_gemini = analyze_question(
        frame, "Tell me what is in this dataset", "inventory.csv", api_key=None,
    )
    assert not used_gemini
    assert plan.operations[0].operation == "dataset_profile"
    profile = results[1]
    assert {row["column"]: row["detected_type"] for row in profile["records"]} == {
        "item": "categorical", "units": "numeric",
    }
    assert results[0]["rows"] == 3
    assert "3 rows and 2 columns" in explanation.answer


def test_gemini_failure_falls_back_to_pandas_results(monkeypatch):
    from src.ai.analyst_service import GeminiAnalyst, GeminiProviderError

    frame = pd.DataFrame({"paragraph": ["amber amber light", "blue blue light"], "family": ["A", "B"]})
    monkeypatch.setattr(GeminiAnalyst, "analyze", lambda *args, **kwargs: (_ for _ in ()).throw(GeminiProviderError("provider unavailable")))
    plan, results, explanation, used_gemini = analyze_question(
        frame, "Find keywords occurring in each family", "text.csv", api_key="configured-key",
    )
    assert used_gemini is False
    assert plan.operations[0].operation == "keyword_frequency_by_group"
    assert {row["keyword"] for row in results[1]["records"]} >= {"amber", "blue"}
    assert "Local Pandas fallback executed: keyword_frequency_by_group" in explanation.answer
    assert "requested analysis was completed" in explanation.answer


def _train_like_frame(two_group_columns=False):
    return pd.DataFrame({
        "id": ["id-1", "id-2", "id-3", "id-4"],
        "prompt_id": ["p-1", "p-1", "p-2", "p-2"],
        "text": ["amber amber bright words in this generated passage", "amber bright words in this generated passage",
                 "violet violet bright words in this generated passage", "violet bright words in this generated passage"],
        "model": ["model-a", "model-a", "model-b", "model-b"],
        **({"source": ["x", "x", "y", "y"]} if two_group_columns else {}),
    })


@pytest.mark.parametrize("alias", ["group_by", "grouping_column", "group_column"])
def test_grouping_column_aliases_normalize_to_canonical_model(alias):
    plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "keyword_frequency_by_group", "text_col": "text", alias: "model",
    }]})
    assert plan.operations[0].text_column == "text"
    assert plan.operations[0].group_by == "model"


def test_legacy_column_alias_maps_text_and_maps_group_when_text_is_explicit():
    text_plan = AnalysisPlan.model_validate({"operations": [{"operation": "keyword_frequency_by_group", "column": "text"}]})
    assert text_plan.operations[0].text_column == "text"
    grouped_plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "keyword_frequency_by_group", "text_column": "text", "column": "model",
    }]})
    assert grouped_plan.operations[0].group_by == "model"


def test_omitted_group_column_infers_model_from_dataset_schema():
    from src.ai.analyst_service import resolve_plan_columns

    frame = _train_like_frame()
    plan = AnalysisPlan.model_validate({"operations": [{"operation": "keyword_frequency_by_group", "text_column": "text"}]})
    resolved = resolve_plan_columns(plan, frame, "Which keywords distinguish each model?")
    assert resolved.operations[0].group_by == "model"
    results = execute_plan(frame, resolved)
    assert any(row["keyword"] == "amber" and row["model"] == "model-a" for row in results[1]["records"])


def test_invalid_group_column_is_not_silently_replaced():
    frame = _train_like_frame()
    plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "keyword_frequency_by_group", "text_column": "text", "group_by": "not_a_column",
    }]})
    with pytest.raises(AnalystError, match="unavailable"):
        execute_plan(frame, plan)


def test_multiple_plausible_group_columns_require_user_clarification():
    from src.ai.analyst_service import AmbiguousColumnError, resolve_plan_columns

    frame = _train_like_frame(two_group_columns=True)
    plan = AnalysisPlan.model_validate({"operations": [{"operation": "keyword_frequency_by_group", "text_column": "text"}]})
    with pytest.raises(AmbiguousColumnError, match="model, source"):
        resolve_plan_columns(plan, frame)
    resolved = resolve_plan_columns(plan, frame, "Group this by model")
    assert resolved.operations[0].group_by == "model"


def test_frequency_and_distinctiveness_returns_counts_rate_and_tfidf():
    frame = _train_like_frame()
    plan = AnalysisPlan.model_validate({"operations": [{
        "operation": "distinctive_keywords_by_group", "text_column": "text", "grouping_column": "model", "top_n": 5,
    }]})
    result = execute_plan(frame, plan)[1]
    assert {"model", "keyword", "count", "occurrences", "frequency_per_1000_tokens", "distinctiveness_tfidf"} <= set(result["columns"])
    assert all(row["distinctiveness_tfidf"] is not None for row in result["records"])


def test_conflicting_group_aliases_are_rejected():
    with pytest.raises(ValidationError, match="Conflicting values supplied for group_by"):
        AnalysisPlan.model_validate({"operations": [{
            "operation": "keyword_frequency_by_group", "text_column": "text",
            "group_by": "model", "grouping_column": "prompt_id",
        }]})


def test_gemini_invalid_group_receives_error_then_one_correction(monkeypatch):
    from types import SimpleNamespace
    from src.ai.analyst_service import GeminiAnalyst

    frame = _train_like_frame()

    class Stub:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def from_function_response(cls, *, name, response):
            return cls(function_response=SimpleNamespace(name=name, response=response))

    def tool_response(group):
        call = SimpleNamespace(name="execute_dataframe_analysis", args={"operations": [{
            "operation": "keyword_frequency_by_group", "text_column": "text", "group_by": group,
        }]})
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=call)]))])

    responses = [tool_response("bad_column"), tool_response("model"),
                 SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Analysis completed after correcting the grouping field.")]))])]

    class Models:
        def __init__(self):
            self.index = 0
            self.inputs = []
        def generate_content(self, **kwargs):
            self.inputs.append(list(kwargs["contents"]))
            result = responses[self.index]
            self.index += 1
            return result

    models = Models()
    client = SimpleNamespace(models=models, close=lambda: None)
    types = SimpleNamespace(HttpOptions=Stub, FunctionDeclaration=Stub, Tool=Stub,
                            AutomaticFunctionCallingConfig=Stub, GenerateContentConfig=Stub,
                            Content=Stub, Part=Stub)
    monkeypatch.setattr("src.ai.analyst_service._load_genai_sdk", lambda: (SimpleNamespace(Client=lambda **kwargs: client), types))
    plan, results, explanation = GeminiAnalyst("unused").analyze(
        "Find frequent keywords in each model", frame, {"dataset_name": "train.csv"}, [], dataset_id="train-id",
    )
    assert models.index == 3
    assert plan.operations[0].group_by == "model"
    assert explanation.answer.startswith("Analysis completed")
    first_tool_feedback = models.inputs[1][-1].parts[0].function_response.response
    assert "unavailable column" in first_tool_feedback["error"]
    assert any(row["keyword"] == "amber" for row in results[1]["records"])


def test_gemini_multiple_tool_calls_in_one_response_are_executed(monkeypatch):
    from types import SimpleNamespace
    from src.ai.analyst_service import GeminiAnalyst

    frame = _train_like_frame()

    class Stub:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        @classmethod
        def from_function_response(cls, *, name, response):
            return cls(function_response=SimpleNamespace(name=name, response=response))

    inspect = SimpleNamespace(name="inspect_dataset", args={})
    analyze = SimpleNamespace(name="execute_dataframe_analysis", args={"operations": [{
        "operation": "keyword_frequency_by_group", "text_column": "text", "grouping_column": "model",
    }]})
    responses = [
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=inspect), SimpleNamespace(function_call=analyze)]))]),
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Inspected the data and calculated grouped keywords.")]))]),
    ]
    class Models:
        def __init__(self): self.index = 0
        def generate_content(self, **kwargs):
            response = responses[self.index]
            self.index += 1
            return response
    client = SimpleNamespace(models=Models(), close=lambda: None)
    types = SimpleNamespace(HttpOptions=Stub, FunctionDeclaration=Stub, Tool=Stub,
                            AutomaticFunctionCallingConfig=Stub, GenerateContentConfig=Stub,
                            Content=Stub, Part=Stub)
    monkeypatch.setattr("src.ai.analyst_service._load_genai_sdk", lambda: (SimpleNamespace(Client=lambda **kwargs: client), types))
    plan, results, _ = GeminiAnalyst("unused").analyze(
        "Find keywords by model", frame, {"dataset_name": "train.csv"}, [], dataset_id="dataset-xyz",
    )
    assert "dataset_profile" in [op.operation for op in plan.operations]
    assert "keyword_frequency_by_group" in [op.operation for op in plan.operations]
    assert [item["operation"] for item in results].count("dataset_profile") == 1
    assert any(item["operation"] == "keyword_frequency_by_group" for item in results)


def test_local_planner_runs_frequency_and_distinctiveness_when_both_requested():
    frame = _train_like_frame()
    plan = rule_based_plan("Find frequently occurring and distinctive keywords for each model", frame)
    assert [op.operation for op in plan.operations] == [
        "keyword_frequency_by_group", "distinctive_keywords_by_group",
    ]
    results = execute_plan(frame, plan)
    assert all({"count", "frequency_per_1000_tokens", "distinctiveness_tfidf"} <= set(item["columns"]) for item in results[1:])


def test_backend_operation_failure_is_not_reported_as_gemini_api_failure(monkeypatch):
    from src.ai.analyst_service import BackendOperationError, GeminiAnalyst

    monkeypatch.setattr(GeminiAnalyst, "analyze", lambda *args, **kwargs: (_ for _ in ()).throw(
        BackendOperationError("Gemini requested an analysis, but the backend operation failed: invalid group.")))
    with pytest.raises(BackendOperationError, match="Gemini requested an analysis, but the backend operation failed"):
        analyze_question(_train_like_frame(), "Find keywords for each model", "train.csv", api_key="configured-key")


def test_filter_and_sort_are_registered_and_execute_with_pandas():
    frame = pd.DataFrame({"amount": [8, 2, 10], "group": ["a", "b", "a"]})
    plan = AnalysisPlan.model_validate({"operations": [
        {"operation": "filter_rows", "column": "amount", "operator": "gt", "value": 5},
        {"operation": "sort_values", "column": "amount", "ascending": False},
    ]})
    results = execute_plan(frame, plan)
    assert results[1]["matched_rows"] == 2
    assert [row["amount"] for row in results[1]["records"]] == [8, 10]
    assert [row["amount"] for row in results[2]["records"]] == [10, 8, 2]


def test_gemini_build_dashboard_tool_saves_charts_in_shared_dashboard_store(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.ai.analyst_service import GeminiAnalyst
    from src.dashboard.ai_dashboard_service import analyze_dashboard_patterns

    frame = pd.DataFrame({"category": ["a", "b", "a"], "amount": [2, 5, 8]})
    store = DashboardStore(tmp_path / "generated.sqlite")

    class Stub:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        @classmethod
        def from_function_response(cls, *, name, response):
            return cls(function_response=SimpleNamespace(name=name, response=response))

    chart_candidates = analyze_dashboard_patterns(frame)["suggested_charts"]
    # Gemini may choose a supported chart that the pattern ranker did not list,
    # when it better matches the user's requested dashboard.
    selected_charts = [{"chart_type": "Pie Chart", "x": "category", "title": "Category mix"}]
    recommend_call = SimpleNamespace(name="recommend_dashboard", args={})
    build_call = SimpleNamespace(name="build_dashboard", args={"dashboard_name": "Category review", "charts": selected_charts})
    responses = [
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=recommend_call)]))]),
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=build_call)]))]),
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="I built and saved the interactive dashboard.")]))]),
    ]
    class Models:
        def __init__(self): self.index = 0
        def generate_content(self, **kwargs):
            response = responses[self.index]
            self.index += 1
            return response
    client = SimpleNamespace(models=Models(), close=lambda: None)
    types = SimpleNamespace(HttpOptions=Stub, FunctionDeclaration=Stub, Tool=Stub,
                            AutomaticFunctionCallingConfig=Stub, GenerateContentConfig=Stub,
                            Content=Stub, Part=Stub)
    monkeypatch.setattr("src.ai.analyst_service._load_genai_sdk", lambda: (SimpleNamespace(Client=lambda **kwargs: client), types))
    plan, results, explanation = GeminiAnalyst("unused").analyze(
        "Build a category dashboard", frame, {"dataset_name": "metrics.csv"}, [],
        dataset_id="data-1", dashboard_store=store,
    )
    assert [operation.operation for operation in plan.operations] == ["recommend_dashboard", "build_dashboard"]
    created = next(result for result in results if result["operation"] == "dashboard_created")
    assert created["operation"] == "dashboard_created"
    assert created["chart_count"] == len(selected_charts) >= 1
    assert len(store.get_dashboard(created["dashboard_id"])["charts"]) == len(selected_charts)
    assert explanation.answer == "I built and saved the interactive dashboard."


def test_dashboard_request_uses_validated_local_builder_if_gemini_only_returns_code(monkeypatch, tmp_path):
    from src.ai.analyst_service import GeminiAnalyst
    from src.dashboard.ai_dashboard_service import create_ai_dashboard

    frame = pd.DataFrame({"category": ["a", "b", "a"], "amount": [2, 5, 8]})
    store = DashboardStore(tmp_path / "fallback.sqlite")
    monkeypatch.setattr(GeminiAnalyst, "analyze", lambda *args, **kwargs: (_ for _ in ()).throw(
        AnalystError("Gemini did not call dashboard builder; code response was not executed.")))
    plan, results, explanation, used_gemini = analyze_question(
        frame, "Build me a dashboard", "metrics.csv", api_key="key", dataset_id="metrics-id", dashboard_store=store,
    )
    assert not used_gemini
    assert [operation.operation for operation in plan.operations] == ["recommend_dashboard", "build_dashboard"]
    created = next(result for result in results if result["operation"] == "dashboard_created")
    assert created["chart_count"] > 0
    assert "created this dashboard" in explanation.answer
    assert len(store.get_dashboard(created["dashboard_id"])["charts"]) == created["chart_count"]


def test_ai_chat_renders_created_dashboard_charts_immediately(tmp_path, monkeypatch):
    from src.dashboard.ai_dashboard_service import create_ai_dashboard
    monkeypatch.setattr("src.ui.ai_analyst_page.resolve_api_key", lambda: None)
    registry = DatasetRegistry()
    ingested = UploadManager().ingest(Upload(b"category,amount\na,2\nb,5\na,8\n", "metrics.csv", "text/csv"), "metrics.csv", "text/csv")
    metadata = registry.register("metrics.csv", ingested)
    store = DashboardStore(tmp_path / "interactive.sqlite")
    built = create_ai_dashboard(store, registry.get_dataframe(metadata.dataset_id), metadata.dataset_id, "metrics.csv", [
        {"chart_type": "Bar Chart", "x": "category", "y_columns": ["amount"], "title": "Average amount"},
    ])
    message = {"role": "assistant", "content": "Built your dashboard.", "message_id": "generated-1",
               "operations": ["build_dashboard"], "results": [built], "mode": "Gemini"}
    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = store
    app.session_state[f"ai_messages_{metadata.dataset_id}"] = [message]
    app.run()
    assert not app.exception
    chart_id = built["charts"][0]["chart_id"]
    assert any(f"ai_built_dashboard_generated-1_{chart_id}" in chart.proto.id for chart in app.get("plotly_chart"))
    assert any(button.label == "Edit chart settings" for button in app.button)


def test_ai_analyst_chat_runs_local_fallback_when_no_key_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.ui.ai_analyst_page.resolve_api_key", lambda: None)
    registry = DatasetRegistry()
    result = UploadManager().ingest(Upload(b"kind,measure\na,1\nb,4\na,8\n", "chat.csv", "text/csv"), "chat.csv", "text/csv")
    metadata = registry.register("chat.csv", result)
    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = DashboardStore(tmp_path / "dashboards.sqlite")
    app.run()
    assert not app.exception
    section_headers = [element.value for element in app.subheader]
    assert "Chat with Gemini" in section_headers
    assert "Manual data tools" not in section_headers
    assert any(button.label == "Open manual data tools" for button in app.button)
    app.button(key=f"open_manual_tools_open_{metadata.dataset_id}").click().run()
    assert not app.exception
    assert "Manual data tools" in [element.value for element in app.subheader]
    app.chat_input(key=f"ai_chat_input_{metadata.dataset_id}").set_value("Analyze this dataset completely").run()
    assert not app.exception
    history = app.session_state[f"ai_messages_{metadata.dataset_id}"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[-1]["mode"] == "Local Pandas"
    assert history[-1]["results"]


def test_sensitive_dataset_confirmation_does_not_mutate_checkbox_state(tmp_path, monkeypatch):
    from src.ai.analyst_service import ExplanationResponse

    monkeypatch.setattr("src.ui.ai_analyst_page.resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.ui.ai_analyst_page.analyze_question",
        lambda *args, **kwargs: (
            AnalysisPlan.model_validate({"operations": [{"operation": "missing_values"}]}),
            [{"operation": "dataset_overview", "rows": 1, "columns": 2, "duplicate_rows": 0}],
            ExplanationResponse(answer="Pandas completed the calculation."),
            True,
        ),
    )
    registry = DatasetRegistry()
    result = UploadManager().ingest(Upload(b"email,value\na@example.com,4\n", "private.csv", "text/csv"), "private.csv", "text/csv")
    metadata = registry.register("private.csv", result)
    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = DashboardStore(tmp_path / "dashboards.sqlite")
    app.run()
    app.chat_input(key=f"ai_chat_input_{metadata.dataset_id}").set_value("Analyze missing values").run()
    assert not app.exception
    assert any(button.label == "Send sanitized context to Gemini" for button in app.button)
    app.button(key=f"ai_confirm_send_{metadata.dataset_id}").click().run()
    assert not app.exception
    history = app.session_state[f"ai_messages_{metadata.dataset_id}"]
    assert history[-1]["mode"] == "Gemini"
    assert history[-1]["content"] == "Pandas completed the calculation."
