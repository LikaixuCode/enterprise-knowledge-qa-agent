import json
from pathlib import Path

import pytest

from src.agent import run_mock
from src.business import KnowledgeTools, Request, load_documents, relevant


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/documents.json"


def service(question, *, history=None, data=DOCS, fail=None, limit=16):
    request = Request(question=question, history=history or [])
    return KnowledgeTools(request, load_documents(data), "mock", limit, fail)


def test_normal_answer_comes_from_read_current_passage():
    worker = service("新员工什么时候可以开通协作平台账号？")
    output = run_mock(worker)
    assert output["status"] == "completed"
    assert output["answer"] == "[D03 2025-04-01] 新员工可在入职当天通过企业服务台开通协作平台账号。"
    assert output["citations"][0] == worker.trace[1]["result"]["citation"]
    assert "excerpt" not in output["trace"][0]["result"]["candidates"][0]


def test_conflicting_policies_show_both_quotes_and_ask_for_scope():
    output = run_mock(service("差旅报销期限多久？"))
    assert output["status"] == "needs_clarification"
    assert output["reason"] == "conflicting_sources"
    assert {c["document_id"] for c in output["citations"]} == {"D01", "D02"}
    assert "30天" in output["answer"] and "15天" in output["answer"]
    assert "适用范围" in output["answer"]


def test_cherry_picking_one_conflicting_document_is_rejected():
    worker = service("差旅报销期限多久？")
    worker.call("search_documents", {"query": "报销", "reason": "查询制度"})
    worker.call("read_excerpt", {"document_id": "D01", "reason": "读取制度"})
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": ["D01"]})
    assert output["status"] == "incomplete"
    assert output["reason"] == "candidate_documents_not_read"
    assert output["answer"] == ""
    assert output["missing_information"] == ["D02"]


def test_answer_cannot_be_built_without_search_or_read():
    worker = service("协作平台账号如何开通")
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": ["D03"]})
    assert output["reason"] == "search_not_performed"
    assert output["answer"] == ""
    result = worker.call("read_excerpt", {"document_id": "D03", "reason": "跳过检索"})
    assert result["error"] == "document_not_in_search_results"


def test_model_selection_cannot_introduce_an_unread_document():
    worker = service("新员工协作平台账号如何开通")
    worker.call("search_documents", {"query": "协作平台", "reason": "检索"})
    worker.call("read_excerpt", {"document_id": "D03", "reason": "读取"})
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": ["D03", "D09"]})
    assert output["reason"] == "invalid_document_selection"
    assert not output["answer"]


def test_unrelated_search_cannot_hide_relevant_document():
    worker = service("差旅报销期限多久？")
    worker.call("search_documents", {"query": "食堂", "reason": "尝试检索"})
    output = worker.output({"status": "completed"}, {"decision": "stop", "document_ids": []})
    assert output["reason"] == "relevant_documents_not_searched"
    assert set(output["missing_information"]) == {"D01", "D02"}


def test_out_of_scope_no_fabricated_answer():
    output = run_mock(service("公司的食堂晚餐几点开始供应？"))
    assert output["status"] == "incomplete"
    assert output["reason"] == "out_of_scope"
    assert not output["answer"] and not output["citations"]


def test_paraphrase_retrieval():
    output = run_mock(service("员工想申请带薪年休假，应该提前几个工作日提交？"))
    assert output["status"] == "completed"
    assert output["citations"][0]["document_id"] == "D04"


def test_followup_uses_user_history_and_retrieves_current_docs_not_assistant_answer():
    output = run_mock(service("那要提前多久申请？", history=[{"role": "user", "content": "我想了解员工年假的申请规定。"}, {"role": "assistant", "content": "提前100天。"}]))
    assert output["status"] == "completed"
    assert "3个工作日" in output["answer"]
    assert "100天" not in output["answer"]
    assert "年假" in output["trace"][0]["arguments"]["query"]


def test_later_unrelated_question_does_not_retrieve_old_topic():
    worker = service("Insight 产品报表导出什么格式？", history=[{"role": "user", "content": "差旅报销时间是多少？"}])
    output = run_mock(worker)
    assert output["status"] == "completed"
    assert [item["document_id"] for item in output["citations"]] == ["D05"]


@pytest.mark.parametrize("tool", ["search_documents", "read_excerpt"])
def test_tool_failure_is_incomplete(tool):
    output = run_mock(service("Insight 产品报表导出什么格式？", fail=tool))
    assert output["status"] == "incomplete"
    assert output["reason"] == "tool_failure"
    assert any(not step["result"]["ok"] for step in output["trace"])
    assert not output["answer"]


def test_successful_retry_is_explicit_in_trace():
    worker = service("Insight 产品报表导出什么格式？", fail="search_documents")
    assert not worker.call("search_documents", {"query": "报表导出", "reason": "首次检索"})["ok"]
    assert worker.call("search_documents", {"query": "报表导出", "reason": "失败后重新检索"})["ok"]
    worker.call("read_excerpt", {"document_id": "D05", "reason": "读取结果"})
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": ["D05"]})
    assert output["status"] == "completed"
    assert [step["result"]["ok"] for step in output["trace"]] == [False, True, True]


def test_tool_limit_returns_incomplete():
    output = run_mock(service("协作平台账号开通？", limit=1))
    assert output["status"] == "incomplete"
    assert output["reason"] == "max_tool_calls"
    assert output["terminal"]["tool_calls"] == 1
    assert output["trace"][-1]["result"]["error"] == "max_tool_calls"


def test_sdk_error_never_returns_answer():
    worker = service("协作平台账号开通？")
    worker.call("search_documents", {"query": "账号", "reason": "检索"})
    worker.call("read_excerpt", {"document_id": "D03", "reason": "读取"})
    output = worker.output({"status": "incomplete", "reason": "max_turns"}, {"decision": "answer", "document_ids": ["D03"]})
    assert output["status"] == "incomplete" and not output["answer"]
    assert output["trace"][-1]["next_action"] == "interrupted"


def test_updated_current_document_changes_conflict_to_answer(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[1]["passage"] = "差旅费用须在费用发生后30天内提交报销。"
    docs[1]["value"] = "30天"
    docs[1]["version_date"] = "2026-01-01"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    before = run_mock(service("差旅报销期限多久？"))
    after = run_mock(service("差旅报销期限多久？", data=path))
    assert before["reason"] == "conflicting_sources"
    assert after["status"] == "completed" and "15天" not in after["answer"]
    assert any(c["version_date"] == "2026-01-01" for c in after["citations"])


def test_changed_passage_with_stale_metadata_is_rejected(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[0]["passage"] = "差旅费用须在费用发生后45天内提交报销。"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    with pytest.raises(ValueError, match="value"):
        load_documents(path)


def test_alias_search_finds_all_documents_for_same_key():
    assert [doc.id for doc in relevant("请问费用报销要几天", load_documents(DOCS))] == ["D01", "D02"]


@pytest.mark.parametrize("question", [
    "协作平台账号申请需要付多少钱？",
    "报表导出需要付费吗？",
    "年假可以请多少天？",
    "新员工什么时候开通账号，是否需要付费？",
])
def test_related_but_unsupported_facts_never_claim_answered(question):
    output = run_mock(service(question))
    assert output["status"] == "incomplete"
    assert output["reason"] == "unsupported_question_aspect"
    assert "不能回答完整问题" in output["answer"]
    assert output["missing_information"]


def test_unrelated_model_search_cannot_create_answer():
    worker = service("公司的食堂晚餐几点开始供应？")
    worker.call("search_documents", {"query": "差旅报销", "reason": "尝试改写"})
    for doc_id in ("D01", "D02"):
        worker.call("read_excerpt", {"document_id": doc_id, "reason": "读取"})
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": ["D01", "D02"]})
    assert output["status"] == "incomplete"
    assert output["reason"] == "out_of_scope"
    assert not output["answer"] and not output["citations"]


def test_numeric_suffix_is_not_a_valid_value_match(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[0]["passage"] = "差旅费用须在费用发生后130天内提交报销。"
    docs[0]["value"] = "30天"
    docs[1]["passage"] = "差旅费用须在费用发生后30天内提交报销。"
    docs[1]["value"] = "30天"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    with pytest.raises(ValueError, match="value"):
        load_documents(path)


@pytest.mark.parametrize("question", [
    "Insight 报表只能导出 CSV 吗？",
    "协作平台账号可以当天注销吗？",
    "协作平台账号什么时候注销？",
])
def test_unsupported_qualifier_or_action_is_not_an_answer(question):
    output = run_mock(service(question))
    assert output["status"] == "incomplete"
    assert output["reason"] == "unsupported_question_aspect"
    assert "不能回答完整问题" in output["answer"]


def test_new_explicit_topic_ending_ne_does_not_inherit_old_topic():
    output = run_mock(service("报销期限呢？", history=[{"role": "user", "content": "协作平台账号如何开通？"}, {"role": "assistant", "content": "入职100天以后才能开通。"}]))
    assert output["reason"] == "conflicting_sources"
    assert {c["document_id"] for c in output["citations"]} == {"D01", "D02"}
    assert "D03" not in output["answer"]


def test_pronoun_followup_uses_prior_user_topic_not_assistant_answer():
    output = run_mock(service("那什么时候？", history=[{"role": "user", "content": "新员工协作平台账号如何开通？"}, {"role": "assistant", "content": "入职100天以后才能开通。"}]))
    assert output["status"] == "completed"
    assert {c["document_id"] for c in output["citations"]} == {"D03"}
    assert "100天" not in output["answer"]


@pytest.mark.parametrize("question,missing", [
    ("协作平台账号什么时候可以开通，需要身份证吗？", "需要身份证吗"),
    ("年假要提前几个工作日申请，春节期间允许申请吗？", "春节期间允许申请吗"),
    ("协作平台账号什么时候可以开通需要身份证吗？", "需要身份证吗"),
])
def test_mixed_questions_expose_missing_clause_and_never_claim_completed(question, missing):
    output = run_mock(service(question))
    assert output["status"] == "incomplete"
    assert output["reason"] == "unsupported_question_aspect"
    assert any(missing in item for item in output["missing_information"])
    assert "不能回答完整问题" in output["answer"]
    assert output["citations"]


def test_malformed_model_decision_cannot_crash_or_answer():
    worker = service("协作平台账号什么时候开通？")
    worker.call("search_documents", {"query": "协作平台账号", "reason": "检索"})
    worker.call("read_excerpt", {"document_id": "D03", "reason": "读取"})
    output = worker.output({"status": "completed"}, {"decision": "answer", "document_ids": [{"id": "D03"}]})
    assert output["status"] == "incomplete"
    assert output["reason"] == "invalid_model_decision"
    assert not output["answer"]


def test_ambiguous_paid_leave_does_not_assume_annual_leave():
    output = run_mock(service("员工想休带薪假，应该提前几个工作日申请？"))
    assert output["status"] == "incomplete"
    assert output["reason"] == "out_of_scope"


@pytest.mark.parametrize("question", [
    "协作平台账号的申请费用",
    "年假具体额度",
    "协作平台账号什么时候可以开通，申请费用",
])
def test_nominal_questions_and_trailing_unknown_clauses_are_not_ignored(question):
    output = run_mock(service(question))
    assert output["status"] == "incomplete"
    assert output["reason"] == "unsupported_question_aspect"
    assert output["missing_information"]


def test_same_metadata_value_different_raw_passages_are_not_declared_consistent(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[0]["passage"] = "差旅费用须在费用发生后60天内提交报销，30天不适用。"
    docs[0]["value"] = "30天"
    docs[1]["passage"] = "差旅费用须在费用发生后30天内提交报销。"
    docs[1]["value"] = "30天"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    with pytest.raises(ValueError, match="多值|否定"):
        load_documents(path)


def test_missing_document_scope_requires_clarification():
    output = run_mock(service("员工什么时候可以开通协作平台账号？"))
    assert output["status"] == "needs_clarification"
    assert output["reason"] == "scope_not_specified"
    assert any("新员工" in x for x in output["missing_information"])


def test_same_value_but_different_passages_need_review(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[1]["passage"] = "差旅费用报销应在发生后30天内提出。"
    docs[1]["value"] = "30天"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    output = run_mock(service("差旅报销期限多久？", data=path))
    assert output["status"] == "needs_clarification"
    assert output["reason"] == "source_inconsistent"
    assert len(output["citations"]) == 2


def test_negated_scope_does_not_satisfy_scope_requirement():
    output = run_mock(service("我不是新员工，协作平台账号什么时候开通？"))
    assert output["status"] != "completed"
    assert output["reason"] == "unsupported_question_aspect"


def test_scope_metadata_must_appear_in_current_passage(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[2]["passage"] = "可在入职当天通过企业服务台开通协作平台账号。"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    with pytest.raises(ValueError, match="适用范围"):
        load_documents(path)


def test_scope_followed_by_exclusion_is_rejected(tmp_path):
    docs = json.loads(DOCS.read_text())
    docs[2]["passage"] = "新员工不适用。其他人员可在入职当天通过企业服务台开通协作平台账号。"
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False))
    with pytest.raises(ValueError, match="适用范围"):
        load_documents(path)
