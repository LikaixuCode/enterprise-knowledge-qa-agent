"""Real SDK agent and deliberately scripted offline mock."""

import json
import os

from .business import KnowledgeTools, canonicalize, resolved_question


def run_mock(service: KnowledgeTools) -> dict:
    terminal = {"status": "completed", "source": "scripted_mock"}
    search = service.call("search_documents", {"query": resolved_question(service.request, service.docs), "reason": "依据问题和上下文检索候选文档"})
    if not search["ok"]:
        return service.output(terminal, {"decision": "stop", "document_ids": []})
    canonical = canonicalize(resolved_question(service.request, service.docs))
    if not search["candidates"] and canonical != resolved_question(service.request, service.docs):
        search = service.call("search_documents", {"query": canonical, "reason": "离线 Mock 使用同义词表改写检索"})
        if not search["ok"]:
            return service.output(terminal, {"decision": "stop", "document_ids": []})
    ids = []
    for candidate in search["candidates"]:
        result = service.call("read_excerpt", {"document_id": candidate["id"], "reason": "读取候选文档的当前原文"})
        if not result["ok"]:
            return service.output(terminal, {"decision": "stop", "document_ids": ids})
        ids.append(candidate["id"])
    result = service.call("submit_decision", {"decision": "answer" if ids else "stop", "document_ids": ids, "reason": "汇总已读取的证据或说明资料缺口"})
    return service.output(terminal, service.submitted_decision if result["ok"] else None)


async def run_glm(service: KnowledgeTools, max_turns: int = 12) -> dict:
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ClaudeSDKError,
        ResultError, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage,
        create_sdk_mcp_server, tool,
    )

    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not token:
        return service.output({"status": "incomplete", "reason": "missing_credential"}, None)

    def response(name, args):
        result = service.call(name, args)
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "is_error": not result["ok"]}

    @tool("search_documents", "检索文档候选元信息，不返回原文；query 用自然语言。", {"query": str, "reason": str})
    async def search_documents(args):
        return response("search_documents", args)

    @tool("read_excerpt", "读取已检索到的文档的原文片段与版本。", {"document_id": str, "reason": str})
    async def read_excerpt(args):
        return response("read_excerpt", args)

    @tool("submit_decision", "完成检索和读取后提交决策；只接受 answer、clarify、stop。", {
        "type": "object", "properties": {"decision": {"type": "string", "enum": ["answer", "clarify", "stop"]},
        "document_ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}},
        "required": ["decision", "document_ids", "reason"]})
    async def submit_decision(args):
        return response("submit_decision", args)

    server = create_sdk_mcp_server(name="knowledge", tools=[search_documents, read_excerpt, submit_decision])
    options = ClaudeAgentOptions(
        tools=[],
        allowed_tools=["mcp__knowledge__search_documents", "mcp__knowledge__read_excerpt", "mcp__knowledge__submit_decision"],
        mcp_servers={"knowledge": server}, strict_mcp_config=True, setting_sources=[],
        permission_mode="dontAsk", max_turns=max_turns,
        model=os.environ.get("GLM_MODEL", "glm-5.3-flash"),
        env={"ANTHROPIC_AUTH_TOKEN": token, "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic"),
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_MAX_RETRIES": "0", "API_TIMEOUT_MS": "60000"},
        system_prompt=(
            "你是企业知识问答 Agent。仅使用提供的业务工具，不得使用历史对话中的助手回答充当证据。"
            "先根据当前问题和必要的用户历史检索；本轮已有明确主题时忽略历史中的其他主题，代词追问才使用上一条用户问题。首次尽量按用户原词检索，再根据结果决定是否改写。检索只返回候选元数据，你必须逐一读取所有相关候选原文。"
            "如结果为空，可以改写问题补充检索；若没有支持资料就停止，不得编造。"
            "发现冲突时读取相关文档，要求用户确认适用范围或优先级，不能仅依据版本日期选一个。"
            "工具失败时可以重试或停止，不得称失败为成功。"
            "每次调用工具的 reason 用一句简短公开依据，不输出内部思维。"
            "最后必须调用 submit_decision 工具提交 decision 与所有已读取的相关 document_ids，不要仅在文本中输出 JSON。"
            "有明确答案选 answer，发现冲突选 clarify，资料不足或工具不可用选 stop。"

        ),
    )
    prompt = {"question": service.request.question, "history": [item.model_dump() for item in service.request.history]}
    terminal = {"status": "incomplete", "reason": "no_result", "model": options.model}
    decision = None
    sdk_events = []
    reported_models = set()
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(json.dumps(prompt, ensure_ascii=False))
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    if message.model:
                        reported_models.add(message.model)
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            sdk_events.append({"type": "tool_request", "tool_use_id": block.id, "message_id": message.message_id, "tool": block.name, "arguments": block.input})
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            sdk_events.append({"type": "tool_result", "tool_use_id": block.tool_use_id, "is_error": block.is_error})
                elif isinstance(message, ResultMessage):
                    terminal = {"status": "completed" if not message.is_error and message.subtype == "success" else "incomplete", "reason": message.terminal_reason or message.subtype, "turns": message.num_turns, "model": options.model}
                    if terminal["status"] == "incomplete":
                        terminal["api_error_status"] = message.api_error_status
                        terminal["detail"] = (message.result or "").replace(token, "[REDACTED]")[:500]

    except ResultError as exc:
        terminal = {"status": "incomplete", "reason": exc.terminal_reason or exc.subtype or "sdk_result_error", "model": options.model}
    except ClaudeSDKError as exc:
        terminal = {"status": "incomplete", "reason": type(exc).__name__, "model": options.model}
    except Exception as exc:
        terminal = {"status": "incomplete", "reason": type(exc).__name__, "model": options.model}
    terminal["reported_models"] = sorted(reported_models)
    terminal["sdk_events"] = sdk_events
    return service.output(terminal, service.submitted_decision)
