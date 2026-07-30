import pytest
from fastapi.testclient import TestClient

from api_server import app, get_service


class FakeService:
    def get_stats(self):
        return {"doc_count": 2, "collection": "test", "persist_dir": "test"}

    def add_files(self, files):
        return {
            "file_count": len(files),
            "chunk_count": len(files),
            "files": [item.filename for item in files],
            "failed_files": [],
            "knowledge_base": self.get_stats(),
        }

    def chat(self, question, session_id):
        return {
            "session_id": session_id or "generated-session",
            "answer": f"回答: {question}",
            "sources": [],
            "thinking": ["测试"],
            "intent": "chat",
            "error": None,
            "trace": [{"step": "classify", "status": "success"}],
        }

    def list_documents(self):
        return [{
            "source": "guide.txt",
            "chunk_count": 2,
            "char_count": 120,
            "file_type": "txt",
            "ingested_at": "2026-07-29T00:00:00+00:00",
        }]

    def document_chunks(self, source, limit=100):
        return [{
            "id": "chunk-1",
            "content": "hello",
            "char_count": 5,
            "chunk_index": 0,
            "metadata": {"source": source},
        }][:limit]

    def delete_document(self, source):
        return 2 if source == "guide.txt" else 0

    def debug_retrieval(self, query, top_k=6):
        return {
            "query": query,
            "top_k": top_k,
            "result_count": 1,
            "latency_ms": 1.2,
            "results": [{
                "content": "报销需审批",
                "metadata": {"source": "guide.txt"},
                "score": 0.91,
            }],
        }

    def add_feedback(self, payload):
        return {"id": "feedback-1", **payload.model_dump()}

    def feedback_dashboard(self):
        return {
            "summary": {
                "total": 1,
                "positive": 1,
                "negative": 0,
                "positive_rate": 1.0,
            },
            "items": [],
        }

    def add_evaluation_case(self, payload):
        return {"id": "case-1", **payload.model_dump()}

    def evaluation_dashboard(self):
        return {
            "summary": {
                "case_count": 1,
                "run_count": 1,
                "passed": 1,
                "pass_rate": 1.0,
                "average_latency_ms": 10.0,
            },
            "cases": [],
            "runs": [],
        }

    def run_evaluation(self, case_ids):
        return {
            "run_count": 1,
            "passed": 1,
            "pass_rate": 1.0,
            "items": [],
        }

    def clear_knowledge_base(self):
        return {"doc_count": 0, "collection": "test", "persist_dir": "test"}

    def reset_session(self, session_id):
        return session_id == "existing"


@pytest.fixture
def client():
    app.dependency_overrides[get_service] = lambda: FakeService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_knowledge_base_stats(client):
    response = client.get("/api/v1/knowledge-base")
    assert response.status_code == 200
    assert response.json()["doc_count"] == 2


def test_upload_documents(client):
    response = client.post(
        "/api/v1/documents",
        files=[("files", ("guide.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 200
    assert response.json()["files"] == ["guide.txt"]


def test_chat(client):
    response = client.post(
        "/api/v1/chat",
        json={"question": "请假是什么？", "session_id": "session-1"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "session-1"
    assert response.json()["intent"] == "chat"
    assert response.json()["trace"][0]["step"] == "classify"


def test_blank_chat_question(client):
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 422


def test_reset_session(client):
    response = client.delete("/api/v1/sessions/existing")
    assert response.status_code == 200
    assert response.json()["cleared"] is True


def test_document_management(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json()["items"][0]["source"] == "guide.txt"

    chunks = client.get("/api/v1/documents/guide.txt/chunks")
    assert chunks.status_code == 200
    assert chunks.json()["items"][0]["content"] == "hello"

    deleted = client.delete("/api/v1/documents/guide.txt")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunks"] == 2

    missing = client.delete("/api/v1/documents/missing.txt")
    assert missing.status_code == 404


def test_retrieval_debug(client):
    response = client.post(
        "/api/v1/retrieval/debug",
        json={"query": "报销流程", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["result_count"] == 1
    assert response.json()["results"][0]["score"] == 0.91


def test_feedback_and_evaluation_endpoints(client):
    feedback = client.post(
        "/api/v1/feedback",
        json={
            "session_id": "session-1",
            "question": "如何报销？",
            "answer": "提交审批。",
            "rating": 1,
            "comment": "有帮助",
            "intent": "knowledge",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["id"] == "feedback-1"

    feedback_summary = client.get("/api/v1/feedback")
    assert feedback_summary.status_code == 200
    assert feedback_summary.json()["summary"]["positive_rate"] == 1.0

    case = client.post(
        "/api/v1/evaluations/cases",
        json={
            "question": "如何报销？",
            "expected_answer": "审批",
            "expected_source": "guide.txt",
        },
    )
    assert case.status_code == 200
    assert case.json()["id"] == "case-1"

    evaluation = client.post(
        "/api/v1/evaluations/run",
        json={"case_ids": ["case-1"]},
    )
    assert evaluation.status_code == 200
    assert evaluation.json()["pass_rate"] == 1.0
