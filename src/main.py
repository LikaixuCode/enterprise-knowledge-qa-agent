import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .agent import run_glm, run_mock
from .business import KnowledgeTools, Request, load_documents


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="企业知识问答 Agent")
    parser.add_argument("--input", type=Path, required=True, help="JSON 问题与可选对话历史")
    parser.add_argument("--data", type=Path, default=ROOT / "data/documents.json", help="合成文档 JSON")
    parser.add_argument("--mode", choices=["glm", "mock"], default="glm")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=16)
    parser.add_argument("--fail-once", choices=["search_documents", "read_excerpt"])
    args = parser.parse_args()
    try:
        if args.max_turns < 1 or args.max_tool_calls < 1:
            raise ValueError("执行上限必须大于零")
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("输入必须为 JSON 对象")
        if not raw.get("question") or not str(raw["question"]).strip():
            result = {"status": "incomplete", "reason": "missing_question", "answer": "", "citations": [], "missing_information": ["question"], "trace": [], "terminal": {"status": "not_started"}}
        else:
            request = Request.model_validate(raw)
            docs = load_documents(args.data)
            service = KnowledgeTools(request, docs, args.mode, args.max_tool_calls, args.fail_once)
            result = run_mock(service) if args.mode == "mock" else asyncio.run(run_glm(service, args.max_turns))
        code = 0 if result["status"] in {"completed", "needs_clarification"} else 1
    except (OSError, ValueError, ValidationError) as exc:
        result = {"status": "error", "reason": "invalid_input_or_data", "message": str(exc)}
        code = 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
