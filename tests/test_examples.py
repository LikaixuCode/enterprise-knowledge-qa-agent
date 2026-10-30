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
