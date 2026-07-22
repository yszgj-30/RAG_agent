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


def test_blank_chat_question(client):
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 422


def test_reset_session(client):
    response = client.delete("/api/v1/sessions/existing")
    assert response.status_code == 200
    assert response.json()["cleared"] is True
