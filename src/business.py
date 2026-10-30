"""Local synthetic document retrieval and evidence-based answer guard."""

import json
import re
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# These mappings are curated vocabulary for this synthetic FAQ. They anchor a
# model's search rewrite to the user's original topic; they are not embeddings
# or a general claim that arbitrary rewritten questions are equivalent.
SYNONYMS = {
    "报账": "报销",
    "出差": "差旅",
    "费用申报": "费用报销",
    "差旅账单": "差旅报销",
    "出差票据": "差旅报销",
    "账户": "账号",
    "帐号": "账号",
    "年休假": "年假",
    "年度休假": "年假",
}

# Documents use a small, explicit positive-statement grammar. In particular,
# merely containing the annotated value does not prove that the passage says it
# is applicable ("不是30天" must never validate a value of "30天").
FACT_PATTERNS = {
    frozenset({"deadline"}): r"差旅费用(?:报销)?(?:须|应|必须)在(?:费用)?发生后(?P<value>\d+(?:\.\d+)?(?:天|日|个工作日|工作日))内(?:提交报销|提出)[。.]?",
    frozenset({"advance_notice"}): r"员工(?:应|须)(?:至少)?提前(?P<value>\d+(?:\.\d+)?(?:个工作日|工作日|天|日))提交年假申请[。.]?",
    frozenset({"activation_time", "application_method"}): r"(?P<scope>[^，。；]+?)(?:可|可以)在(?P<value>入职(?:当天|当日|前\d+天|后\d+天))通过[^，。；]+?开通协作平台账号[。.]?",
    frozenset({"file_format"}): r"Insight\s*产品支持将报表导出为\s*(?P<value>[A-Z0-9]+)\s*文件[。.]?",
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str = Field(min_length=1)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value):
        if value not in {"user", "assistant"}:
            raise ValueError("history role must be user or assistant")
        return value


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class AgentDecision(BaseModel):
    decision: Literal["answer", "clarify", "stop"]
    document_ids: list[str]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    version_date: date
    title: str = Field(min_length=1)
    key: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    value: str = Field(min_length=1)
    aspects: list[str] = Field(min_length=1)
    scope: str | None = None
    passage: str = Field(min_length=1)


def load_documents(path: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("文档资料必须是数组")
    docs = [Document.model_validate(item) for item in raw]
    if len({doc.id for doc in docs}) != len(docs):
        raise ValueError("文档 ID 必须唯一")
    if any(not re.search(r"(?<![0-9A-Za-z.])" + re.escape(doc.value) + r"(?![0-9A-Za-z.])", doc.passage) for doc in docs):
        raise ValueError("文档的 value 必须出现在原文片段中")
    for doc in docs:
        if doc.scope and (doc.scope not in doc.passage or re.search(r"(?:不是|并非|不属于|不适用|非|排除)" + re.escape(doc.scope), doc.passage) or re.search(re.escape(doc.scope) + r".{0,3}(?:不适用|不允许|不能|不得|不可)", doc.passage)):
            raise ValueError("文档适用范围与原文不一致，需人工核实")
        measures = defaultdict(set)
        for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(个工作日|工作日|天|日|小时|个月|元)", doc.passage):
            measures[unit].add(number)
        if any(len(values) > 1 for values in measures.values()) or re.search(re.escape(doc.value) + r".{0,6}(?:不适用|无效|作废|不得|不应)", doc.passage):
            raise ValueError("文档同单位多值或否定 value，需人工核实")
        pattern = FACT_PATTERNS.get(frozenset(doc.aspects))
        fact = re.fullmatch(pattern, doc.passage) if pattern else None
        if not fact or fact.group("value") != doc.value or fact.groupdict().get("scope") != doc.scope:
            raise ValueError("文档不符合已支持的正向事实格式或标注与原文不一致，需人工核实")
    return docs


def normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())


def canonicalize(text: str) -> str:
    for alias, canonical in sorted(SYNONYMS.items(), key=lambda item: -len(item[0])):
        text = text.replace(alias, canonical)
    return text


def relevant(query: str, docs: list[Document]) -> list[Document]:
    """Use aliases and conservative character bigram overlap; no embeddings are implied."""
    query = normalize(query)
    query_bigrams = {query[i:i + 2] for i in range(len(query) - 1)}
    selected_keys = set()
    selected_aspects = set()
    for doc in docs:
        for alias in doc.aliases:
            term = normalize(alias)
            if not term:
                continue
            if term in query or (len(term) >= 4 and len(query) >= 4 and SequenceMatcher(None, term, query).ratio() >= 0.65 and len(query_bigrams & {term[i:i + 2] for i in range(len(term) - 1)}) >= 2):
                selected_keys.add(doc.key)
                selected_aspects.update(doc.aspects)
                break
    # The supported grammar maps each aspect to one synthetic FAQ topic, so
    # include all records for the matched aspect even if their keys or aliases
    # differ. This avoids silently dropping a conflicting source.
    return [doc for doc in docs if doc.key in selected_keys or set(doc.aspects) & selected_aspects]


# Finite vocabulary for the synthetic FAQ. Unknown words inside a question
# remain unmatched and cause abstention; this is not general semantic inference.
ASPECT_WORDS = {
    "deadline": ["报销", "差旅", "费用", "提交", "发生", "期限", "多久", "多少天", "几天", "截止", "截至", "时间", "内", "日"],
    "activation_time": ["开通", "开设", "申请", "获得", "获取", "领取", "激活", "入职", "什么时候", "何时", "哪天", "几时", "多久", "几天", "时间", "多长时间"],
    "application_method": ["开通", "开设", "申请", "获得", "获取", "领取", "激活", "如何", "怎么", "怎样", "方式", "流程", "通过什么", "在哪里", "到哪里"],
    "advance_notice": ["提前", "多久", "几天", "多少天", "几日", "多少日", "几个工作日", "多少个工作日", "时间", "多长", "申请", "提交", "请", "休"],
    "file_format": ["报表", "导出", "支持", "文件", "什么", "哪些", "哪种", "格式", "类型", "csv", "能够", "可以", "吗"],
}
GENERAL_WORDS = ["请问", "想了解", "我想", "请告诉我", "我", "新员工", "员工", "用户", "公司", "企业", "产品", "制度", "规定", "问题", "一下", "现在", "关于", "根据", "可以", "能够", "应该", "应当", "要", "需要", "是否", "可以", "吗", "呢", "啊", "那", "那么", "的", "是", "有", "为", "在", "后", "前", "想"]
ASPECT_SIGNALS = {
    "deadline": r"期限|截至|截止|(?:多少天|几天|多少日|几日).{0,3}内|多久内",
    "activation_time": r"什么时候|何时|几时|哪天|开通时间",
    "application_method": r"如何|怎么|怎样|方式|流程|通过什么|在哪里|到哪里",
    "advance_notice": r"提前.*(?:多久|几天|多少天|几日|多少日|几个工作日|多少个工作日|时间|多长)|(?:多久|几天|多少天|工作日).*提前",
    "file_format": r"格式|文件类型|哪些类型|什么类型|csv",
}


def unsupported_parts(question: str, docs: list[Document]) -> list[str]:
    """Only accept explicitly modeled question phrases, clause by clause."""
    available = set.intersection(*(set(doc.aspects) for doc in docs)) if docs else set()
    aliases = {normalize(canonicalize(alias)) for doc in docs for alias in doc.aliases}
    pieces = re.findall(r"[^，,；;。？！?!]+[？?]?", question)
    markers = r"什么时候|何时|几时|哪天|多长时间|多久|多少(?:天|日|个)?|几(?:天|日|个|时)|如何|怎么|怎样|什么|哪些|哪个|是否|能否|要不要|吗|呢"
    missing = []
    found_question = False
    for piece in pieces:
        piece = piece.strip()
        normalized = normalize(piece)
        signals = re.findall(markers, piece)
        matched = {aspect for aspect in available if re.search(ASPECT_SIGNALS.get(aspect, r"(?!)"), normalized)}
        if "advance_notice" in matched:
            matched.discard("activation_time")
        is_question = bool(signals or matched or piece.endswith(("？", "?")) or re.search(r"(?:需要|可以|允许|能不能|可否)", piece))
        found_question |= is_question
        vocabulary = aliases | set(GENERAL_WORDS)
        for aspect in (matched if is_question else available):
            vocabulary.update(ASPECT_WORDS[aspect])
        residue = normalized
        for word in sorted(vocabulary, key=len, reverse=True):
            if word:
                residue = residue.replace(normalize(word), "")
        if residue or (is_question and (not matched or len(signals) > 1)) or (
            "activation_time" in matched and re.search(r"需要|必须|应当|应该|须", normalized)
        ):
            missing.append(piece)
    return missing or ([] if found_question else [question])


def resolved_question(request: Request, docs: list[Document]) -> str:
    # An explicit current topic wins over any earlier topic.
    if relevant(canonicalize(request.question), docs):
        return request.question
    # History is context for retrieval only; assistant messages are never evidence.
    if re.search(r"^(那|那么|这个|它|这种|这些|还有|请问它)|呢[？?]?$", request.question):
        previous = next((item.content for item in reversed(request.history) if item.role == "user"), "")
        if previous:
            return f"{previous} {request.question}"
    return request.question


class KnowledgeTools:
    def __init__(self, request: Request, docs: list[Document], mode: str, max_calls: int, fail_once: str | None):
        self.request = request
        self.docs = docs
        self.by_id = {doc.id: doc for doc in docs}
        self.mode = mode
        self.max_calls = max_calls
        self.fail_once = fail_once
        self.failed_injected = False
        self.calls = 0
        self.limit_hit = False
        self.trace = []
        self.searched_ids = set()
        self.reads = {}
        self.unresolved_errors = set()
        self.successful_searches = 0
        self.submitted_decision = None

    def call(self, name: str, args: dict) -> dict:
        if self.calls >= self.max_calls:
            self.limit_hit = True
            result = {"ok": False, "error": "max_tool_calls"}
        else:
            self.calls += 1
            if self.fail_once == name and not self.failed_injected:
                self.failed_injected = True
                self.unresolved_errors.add(name)
                result = {"ok": False, "error": "injected_tool_failure", "message": "模拟工具失败"}
            elif name == "search_documents":
                self.unresolved_errors.discard(name)
                query = args.get("query", "")
                hits = relevant(query, self.docs)
                self.successful_searches += 1
                self.searched_ids.update(doc.id for doc in hits)
                result = {"ok": True, "candidates": [{"id": doc.id, "title": doc.title, "version_date": doc.version_date.isoformat()} for doc in hits]}
            elif name == "read_excerpt":
                doc_id = args.get("document_id", "")
                if doc_id not in self.searched_ids:
                    result = {"ok": False, "error": "document_not_in_search_results"}
                    self.unresolved_errors.add(name)
                else:
                    self.unresolved_errors.discard(name)
                    doc = self.by_id[doc_id]
                    citation = {"document_id": doc.id, "version_date": doc.version_date.isoformat(), "title": doc.title, "excerpt": doc.passage}
                    self.reads[doc_id] = citation
                    result = {"ok": True, "citation": citation}
            elif name == "submit_decision":
                try:
                    self.submitted_decision = AgentDecision.model_validate(args).model_dump()
                    self.unresolved_errors.discard(name)
                    result = {"ok": True, "accepted": True}
                except ValidationError:
                    self.unresolved_errors.add(name)
                    result = {"ok": False, "error": "invalid_model_decision"}
            else:
                result = {"ok": False, "error": "unknown_tool"}
                self.unresolved_errors.add(name)
        event = {"step": len(self.trace) + 1, "tool": name, "arguments": args, "result": result, "decision_basis": args.get("reason", ""), "next_action": "pending"}
        if self.trace:
            self.trace[-1]["next_action"] = name
        self.trace.append(event)
        return result

    def output(self, terminal: dict, decision: dict | None) -> dict:
        try:
            decision = AgentDecision.model_validate(decision).model_dump() if decision is not None else None
        except ValidationError:
            decision = None
        trace = [dict(event) for event in self.trace]
        if trace:
            trace[-1]["next_action"] = "finish" if terminal.get("status") == "completed" and not self.limit_hit else "interrupted"
        sdk_events = terminal.get("sdk_events", [])
        requests = [(index, event) for index, event in enumerate(sdk_events) if event.get("type") == "tool_request"]
        results = {event.get("tool_use_id"): index for index, event in enumerate(sdk_events) if event.get("type") == "tool_result"}
        matched = []
        for event in trace:
            match = next(((index, request) for index, request in requests
                          if request.get("tool", "").rsplit("__", 1)[-1] == event["tool"] and request.get("arguments") == event["arguments"]), None)
            if match:
                requests.remove(match)
                event["tool_use_id"] = match[1]["tool_use_id"]
                event["model_message_id"] = match[1]["message_id"]
            matched.append(match)
        for index, event in enumerate(trace):
            if index + 1 == len(trace):
                event["next_action_relation"] = "terminal"
            elif terminal.get("source") == "scripted_mock":
                event["next_action_relation"] = "scripted_sequence"
            elif matched[index] and matched[index + 1]:
                current, following = matched[index][1], matched[index + 1][1]
                if current["message_id"] == following["message_id"]:
                    event["next_action_relation"] = "same_model_message"
                elif results.get(current["tool_use_id"], len(sdk_events)) < matched[index + 1][0]:
                    event["next_action_relation"] = "after_tool_result"
                else:
                    event["next_action_relation"] = "unknown"
            else:
                event["next_action_relation"] = "unknown"
        resolved = canonicalize(resolved_question(self.request, self.docs))
        expected = {doc.id for doc in relevant(resolved, self.docs)}
        # Only evidence connected to the user's topic through a curated synonym
        # or direct term is eligible; a model's unrelated rewrite cannot answer.
        candidates = expected
        citations = [self.reads[doc.id] for doc in self.docs if doc.id in self.reads and doc.id in candidates]
        result = {"mode": self.mode, "question": self.request.question, "status": "incomplete", "answer": "", "citations": citations, "missing_information": [], "reason": "", "trace": trace, "terminal": {**terminal, "tool_calls": self.calls}, "agent_decision": decision}
        if self.limit_hit or (self.calls >= self.max_calls and not decision):
            result["reason"] = "max_tool_calls"
        elif terminal.get("status") != "completed":
            result["reason"] = terminal.get("reason", "model_error")
        elif self.unresolved_errors:
            result["reason"] = "tool_failure"
        elif decision is None or decision.get("decision") not in {"answer", "clarify", "stop"}:
            result["reason"] = "invalid_model_decision"
        elif self.successful_searches == 0:
            result["reason"] = "search_not_performed"
        elif expected - self.searched_ids:
            result["reason"] = "relevant_documents_not_searched"
            result["missing_information"] = sorted(expected - self.searched_ids)
        elif candidates - set(self.reads):
            result["reason"] = "candidate_documents_not_read"
            result["missing_information"] = sorted(candidates - set(self.reads))
        elif not candidates:
            if self.reads and decision["decision"] != "stop":
                result["status"] = "needs_clarification"
                result["reason"] = "unverified_rewrite"
                result["missing_information"] = ["无法确定改写检索结果与原问题是否相关，请确认主题或补充资料"]
                result["unverified_candidates"] = list(self.reads.values())
            else:
                result["reason"] = "out_of_scope"
                result["missing_information"] = ["未检索到可支持回答的制度或产品资料"]
        elif unsupported_parts(canonicalize(self.request.question), [doc for doc in self.docs if doc.id in candidates]):
            result["reason"] = "unsupported_question_aspect"
            result["missing_information"] = unsupported_parts(canonicalize(self.request.question), [doc for doc in self.docs if doc.id in candidates])
            if citations:
                result["answer"] = "以下原文仅供参考，不能回答完整问题：\n" + "\n".join(f"[{c['document_id']} {c['version_date']}] {c['excerpt']}" for c in citations)
        else:
            selected = [doc for doc in self.docs if doc.id in candidates]
            conflict_ids = set()
            inconsistent_ids = set()
            for index, left in enumerate(selected):
                for right in selected[index + 1:]:
                    if set(left.aspects) & set(right.aspects):
                        if left.value != right.value:
                            conflict_ids.update((left.id, right.id))
                        elif left.passage != right.passage:
                            inconsistent_ids.update((left.id, right.id))
            missing_scope = {doc.scope for doc in self.docs if doc.id in candidates and doc.scope and doc.scope not in resolved_question(self.request, self.docs)}
            if conflict_ids or inconsistent_ids:
                result["status"] = "needs_clarification"
                result["reason"] = "conflicting_sources" if conflict_ids else "source_inconsistent"
                result["missing_information"] = ["各制度的适用范围或明确的优先级"]
                result["answer"] = "相关资料内容不一致，无法确定适用规则，请确认适用范围或制度优先级：\n" + "\n".join(f"[{doc.id} {doc.version_date.isoformat()}] {doc.passage}" for doc in selected if doc.id in conflict_ids | inconsistent_ids)
            elif missing_scope:
                result["status"] = "needs_clarification"
                result["reason"] = "scope_not_specified"
                result["missing_information"] = [f"请确认是否适用于{scope}" for scope in sorted(missing_scope)]
                result["answer"] = "以下原文包含适用范围，请先确认适用对象：\n" + "\n".join(f"[{c['document_id']} {c['version_date']}] {c['excerpt']}" for c in citations)
            elif decision["decision"] != "answer":
                result["status"] = "needs_clarification" if decision["decision"] == "clarify" else "incomplete"
                result["reason"] = "model_requested_clarification" if decision["decision"] == "clarify" else "model_stopped"
                result["missing_information"] = ["请补充问题的适用范围或背景"]
            elif set(decision.get("document_ids", [])) != candidates:
                result["reason"] = "invalid_document_selection"
            else:
                result["status"] = "completed"
                result["reason"] = "supported_by_sources"
                result["answer"] = "\n".join(f"[{doc.id} {doc.version_date.isoformat()}] {doc.passage}" for doc in self.docs if doc.id in candidates)
        return result
