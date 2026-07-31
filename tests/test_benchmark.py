import json
from pathlib import Path

from evaluation.benchmark import (
    answer_context_rank,
    is_refusal,
    load_dataset,
    matches_expected_terms,
    summarize_results,
)


def test_dataset_has_expected_coverage():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "dataset.json"
    payload = load_dataset(dataset_path)
    cases = payload["cases"]

    assert len(cases) == 50
    assert sum(case["answerable"] for case in cases) == 40
    assert sum(not case["answerable"] for case in cases) == 10
    assert len({case["id"] for case in cases}) == 50


def test_keyword_groups_accept_alternatives_but_require_every_group():
    groups = [["三个月", "3个月"], ["书面"]]

    assert matches_expected_terms("需要提前3个月并提交书面材料。", groups)
    assert not matches_expected_terms("需要提前三个月。", groups)


def test_refusal_detection():
    assert is_refusal("资料不足，无法根据知识库回答。")
    assert not is_refusal("支持 Apple HomeKit，可以直接配对。")


def test_answer_context_rank_requires_source_and_every_term_group():
    retrieved = [
        {
            "content": "返回后五个工作日内提交材料。",
            "metadata": {"source": "其他制度.txt"},
        },
        {
            "content": "出差返回后五个工作日内提交差旅费报销单。",
            "metadata": {"source": "报销流程规范.txt"},
        },
    ]

    assert answer_context_rank(
        retrieved,
        "报销流程规范.txt",
        [["五个工作日"], ["差旅费报销单"]],
        3,
    ) == 2


def test_summary_calculates_retrieval_and_generation_rates():
    results = [
        {
            "category": "制度",
            "answerable": True,
            "retrieval_hit": True,
            "source_rank": 1,
            "answer_context_hit": True,
            "answer_context_rank": 1,
            "generation_attempted": True,
            "answer_pass": True,
            "source_cited": True,
            "refusal_pass": None,
            "generation_error": None,
            "retrieval_latency_ms": 10,
            "generation_latency_ms": 100,
            "total_latency_ms": 110,
        },
        {
            "category": "知识库外",
            "answerable": False,
            "retrieval_hit": None,
            "source_rank": None,
            "answer_context_hit": None,
            "answer_context_rank": None,
            "generation_attempted": True,
            "answer_pass": None,
            "source_cited": None,
            "refusal_pass": True,
            "generation_error": None,
            "retrieval_latency_ms": 20,
            "generation_latency_ms": 120,
            "total_latency_ms": 140,
        },
    ]

    metrics = summarize_results(results, hit_k=3)

    assert metrics["retrieval_hit_at_3"] == 1.0
    assert metrics["answer_context_hit_at_3"] == 1.0
    assert metrics["answer_context_mrr_at_3"] == 1.0
    assert metrics["answer_keyword_pass_rate"] == 1.0
    assert metrics["source_citation_rate"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
