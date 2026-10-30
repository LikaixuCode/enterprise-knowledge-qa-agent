import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case,expected,code", [
    ("normal", "completed", 0),
    ("conflict", "needs_clarification", 0),
    ("out-of-scope", "incomplete", 1),
    ("paraphrase", "completed", 0),
    ("followup", "completed", 0),
    ("missing", "incomplete", 1),
])
def test_mock_cli(case, expected, code):
    process = subprocess.run([sys.executable, "-m", "src.main", "--mode", "mock", "--input", f"examples/{case}.input.json"], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == code, process.stderr
    result = json.loads(process.stdout)
    assert result["status"] == expected


def test_invalid_data_is_reported_without_model(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    process = subprocess.run([sys.executable, "-m", "src.main", "--mode", "mock", "--input", "examples/normal.input.json", "--data", str(path)], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    assert json.loads(process.stdout)["reason"] == "invalid_input_or_data"
