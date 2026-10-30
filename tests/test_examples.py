"""Check saved real traces against the current synthetic dataset and business rules."""

import json
from pathlib import Path

import pytest

from src.business import KnowledgeTools, Request, load_documents


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["normal", "conflict", "out-of-scope", "failure", "mixed"])
def test_saved_glm_trace_replays_against_current_data(case):
    request = Request.model_validate_json((ROOT / f"examples/{case}.input.json").read_text())
    recorded = json.loads((ROOT / f"examples/{case}.glm.output.json").read_text())
    service = KnowledgeTools(request, load_documents(ROOT / "data/documents.json"), "glm", 16, "read_excerpt" if case == "failure" else None)
    for step in recorded["trace"]:
        result = service.call(step["tool"], step["arguments"])
        assert result == step["result"]
    current = service.output(recorded["terminal"], service.submitted_decision)
    for field in ("status", "reason", "answer", "citations", "missing_information", "agent_decision"):
        assert current[field] == recorded[field], field


@pytest.mark.parametrize("case,input_case,data", [
    ("conflict-current", "conflict", "data/documents.json"),
    ("updated-current", "conflict", "examples/updated-documents.json"),
    ("rewrite-current", "rewrite", "data/documents.json"),
    ("followup-current", "followup", "data/documents.json"),
    ("duration-current", "duration", "data/documents.json"),
])
def test_new_glm_trace_replays_the_actual_data_snapshot(case, input_case, data):
    request = Request.model_validate_json((ROOT / f"examples/{input_case}.input.json").read_text())
    recorded = json.loads((ROOT / f"examples/{case}.glm.output.json").read_text())
    service = KnowledgeTools(request, load_documents(ROOT / data), "glm", 16, None)
    for step in recorded["trace"]:
        assert service.call(step["tool"], step["arguments"]) == step["result"]
    current = service.output(recorded["terminal"], service.submitted_decision)
    for field in ("status", "reason", "answer", "citations", "missing_information", "agent_decision", "trace"):
        assert current[field] == recorded[field], field


@pytest.mark.parametrize("case", ["conflict-current", "updated-current", "rewrite-current", "followup-current", "duration-current"])
def test_new_glm_trace_matches_sdk_request_and_response_ids(case):
    recorded = json.loads((ROOT / f"examples/{case}.glm.output.json").read_text())
    events = recorded["terminal"]["sdk_events"]
    requests = {event["tool_use_id"]: event for event in events if event["type"] == "tool_request"}
    results = {event["tool_use_id"] for event in events if event["type"] == "tool_result"}
    assert requests and set(requests) == results
    assert len(recorded["trace"]) == len(requests)
    for item in recorded["trace"]:
        sdk_request = requests[item["tool_use_id"]]
        assert sdk_request["message_id"] == item["model_message_id"]
        assert sdk_request["arguments"] == item["arguments"]
