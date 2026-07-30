import pytest

from operations.store import OperationsStore


@pytest.fixture
def store(tmp_path):
    return OperationsStore(tmp_path / "operations.db")


def test_feedback_round_trip_and_summary(store):
    created = store.add_feedback(
        session_id="session-1",
        question="公司的年假是多少？",
        answer="每年 5 天。",
        rating=1,
        comment="引用清楚",
        intent="knowledge",
    )

    assert created["id"]
    assert store.list_feedback()[0]["comment"] == "引用清楚"
    assert store.feedback_summary() == {
        "total": 1,
        "positive": 1,
        "negative": 0,
        "positive_rate": 1.0,
    }


def test_feedback_rejects_invalid_rating(store):
    with pytest.raises(ValueError, match="rating"):
        store.add_feedback(
            session_id="session-1",
            question="问题",
            answer="答案",
            rating=0,
        )


def test_evaluation_case_and_run_round_trip(store):
    case = store.add_evaluation_case(
        question="如何报销？",
        expected_answer="审批",
        expected_source="报销流程规范.txt",
    )
    run = store.record_evaluation_run(
        case_id=case["id"],
        answer="提交审批后报销。",
        intent="knowledge",
        sources=["报销流程规范.txt"],
        passed=True,
        latency_ms=18.35,
    )

    assert run["passed"] is True
    assert store.list_evaluation_runs()[0]["sources"] == ["报销流程规范.txt"]
    assert store.evaluation_summary() == {
        "case_count": 1,
        "run_count": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "average_latency_ms": 18.35,
    }

    assert store.delete_evaluation_case(case["id"]) is True
    assert store.evaluation_summary()["case_count"] == 0
