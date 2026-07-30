"""FastAPI 后端入口，复用现有 RAG 业务模块。"""
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from agent.core import RAGAgent
from document.processor import DocumentProcessor
from memory.manager import ChatMemoryManager
from models.tongyi_llm import TongyiLLM
from operations.store import OperationsStore
from vectordb.store import VectorStore


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict]
    thinking: List[str]
    intent: str = "unknown"
    error: Optional[str] = None
    trace: List[Dict] = Field(default_factory=list)


class RetrievalDebugRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("检索问题不能为空")
        return value


class FeedbackRequest(BaseModel):
    session_id: str = Field(default="anonymous", max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=20000)
    rating: int
    comment: str = Field(default="", max_length=2000)
    intent: str = Field(default="unknown", max_length=32)


class EvaluationCaseRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str = Field(default="", max_length=2000)
    expected_source: str = Field(default="", max_length=512)


class EvaluationRunRequest(BaseModel):
    case_ids: List[str] = Field(default_factory=list)


class RAGService:
    """管理共享知识库和按 session_id 隔离的对话记忆。"""

    def __init__(
        self,
        api_key: str,
        *,
        operations_store: Optional[OperationsStore] = None,
    ):
        self.vector_store = VectorStore(api_key=api_key)
        self.processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
        self.llm = TongyiLLM(
            api_key=api_key,
            model_name=os.getenv("QWEN_MODEL", "qwen3.7-plus"),
            temperature=0.1,
        )
        self._sessions: Dict[str, RAGAgent] = {}
        self._lock = threading.RLock()
        self.operations = operations_store or OperationsStore()

    def get_stats(self) -> Dict:
        return self.vector_store.get_stats()

    def add_files(self, files: List[UploadFile]) -> Dict:
        chunks = []
        imported_files = []
        failed_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for upload in files:
                filename = Path(upload.filename or "").name
                extension = Path(filename).suffix.lower()
                if not filename or extension not in self.processor.SUPPORTED_EXTENSIONS:
                    failed_files.append({
                        "filename": filename or "未命名文件",
                        "error": "仅支持 .txt、.pdf 和 .md 文件",
                    })
                    continue

                file_path = Path(tmpdir) / filename
                try:
                    with file_path.open("wb") as target:
                        shutil.copyfileobj(upload.file, target)
                    file_chunks = self.processor.process_file(str(file_path))
                    ingested_at = datetime.now(timezone.utc).isoformat()
                    for index, chunk in enumerate(file_chunks):
                        chunk.metadata["source"] = filename
                        chunk.metadata.pop("file_path", None)
                        chunk.metadata["file_type"] = extension.lstrip(".")
                        chunk.metadata["ingested_at"] = ingested_at
                        chunk.metadata["chunk_index"] = index
                    chunks.extend(file_chunks)
                    imported_files.append(filename)
                except Exception as exc:
                    failed_files.append({"filename": filename, "error": str(exc)})

        if not chunks:
            raise ValueError("未从上传文件中解析到可入库的文本")

        chunk_count = self.vector_store.add_documents(chunks)
        return {
            "file_count": len(imported_files),
            "chunk_count": chunk_count,
            "files": imported_files,
            "failed_files": failed_files,
            "knowledge_base": self.get_stats(),
        }

    def chat(self, question: str, session_id: Optional[str]) -> Dict:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        normalized_session_id = (session_id or "").strip() or uuid.uuid4().hex
        with self._lock:
            agent = self._sessions.get(normalized_session_id)
            if agent is None:
                agent = RAGAgent(
                    llm=self.llm,
                    vector_store=self.vector_store,
                    memory=ChatMemoryManager(max_turns=10),
                )
                self._sessions[normalized_session_id] = agent
            result = agent.query(question)

        return {"session_id": normalized_session_id, **result}

    def list_documents(self) -> List[Dict]:
        return self.vector_store.list_documents()

    def document_chunks(self, source: str, limit: int = 100) -> List[Dict]:
        return self.vector_store.get_document_chunks(source, limit=limit)

    def delete_document(self, source: str) -> int:
        with self._lock:
            return self.vector_store.delete_document(source)

    def debug_retrieval(self, query: str, top_k: int = 6) -> Dict:
        started = time.perf_counter()
        results = self.vector_store.similarity_search(query, k=top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
            "latency_ms": latency_ms,
            "results": results,
        }

    def add_feedback(self, payload: FeedbackRequest) -> Dict:
        return self.operations.add_feedback(**payload.model_dump())

    def feedback_dashboard(self) -> Dict:
        return {
            "summary": self.operations.feedback_summary(),
            "items": self.operations.list_feedback(),
        }

    def add_evaluation_case(self, payload: EvaluationCaseRequest) -> Dict:
        return self.operations.add_evaluation_case(**payload.model_dump())

    def evaluation_dashboard(self) -> Dict:
        return {
            "summary": self.operations.evaluation_summary(),
            "cases": self.operations.list_evaluation_cases(),
            "runs": self.operations.list_evaluation_runs(),
        }

    def run_evaluation(self, case_ids: List[str]) -> Dict:
        cases = self.operations.list_evaluation_cases()
        selected = [
            case for case in cases
            if not case_ids or case["id"] in set(case_ids)
        ]
        if not selected:
            raise ValueError("没有可运行的评测用例")

        run_items = []
        evaluation_session = f"evaluation-{uuid.uuid4().hex}"
        for case in selected:
            started = time.perf_counter()
            result = self.chat(case["question"], evaluation_session)
            latency_ms = (time.perf_counter() - started) * 1000
            source_names = [
                str(source.get("source", ""))
                for source in result.get("sources", [])
            ]
            answer_ok = (
                not case["expected_answer"]
                or case["expected_answer"].lower() in result["answer"].lower()
            )
            source_ok = (
                not case["expected_source"]
                or any(
                    case["expected_source"].lower() in source.lower()
                    for source in source_names
                )
            )
            run_items.append(
                self.operations.record_evaluation_run(
                    case_id=case["id"],
                    answer=result["answer"],
                    intent=result.get("intent", "unknown"),
                    sources=source_names,
                    passed=answer_ok and source_ok and not result.get("error"),
                    latency_ms=latency_ms,
                )
            )
        self.reset_session(evaluation_session)
        passed = sum(1 for item in run_items if item["passed"])
        return {
            "run_count": len(run_items),
            "passed": passed,
            "pass_rate": round(passed / len(run_items), 4),
            "items": run_items,
        }

    def clear_knowledge_base(self) -> Dict:
        with self._lock:
            self.vector_store.clear()
        return self.get_stats()

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


app = FastAPI(
    title="企业知识库 RAG API",
    version="1.0.0",
    description="文档入库、知识库管理与多轮问答接口。",
)

_service: Optional[RAGService] = None
_service_lock = threading.Lock()


def _api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")


def get_service() -> RAGService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                api_key = _api_key()
                if not api_key:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="未配置 DASHSCOPE_API_KEY 环境变量",
                    )
                try:
                    _service = RAGService(api_key)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"RAG 服务初始化失败: {exc}",
                    ) from exc
    return _service


@app.get("/")
def root() -> Dict[str, str]:
    return {"name": app.title, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "api_key_configured": bool(_api_key()),
        "service_initialized": _service is not None,
    }


@app.get("/api/v1/knowledge-base")
def knowledge_base(service: RAGService = Depends(get_service)) -> Dict:
    return service.get_stats()


@app.get("/api/v1/documents")
def list_documents(service: RAGService = Depends(get_service)) -> Dict:
    return {"items": service.list_documents()}


@app.get("/api/v1/documents/{source}/chunks")
def document_chunks(
    source: str,
    limit: int = 100,
    service: RAGService = Depends(get_service),
) -> Dict:
    return {"source": source, "items": service.document_chunks(source, limit)}


@app.delete("/api/v1/documents/{source}")
def delete_document(source: str, service: RAGService = Depends(get_service)) -> Dict:
    deleted = service.delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    return {"source": source, "deleted_chunks": deleted}


@app.post("/api/v1/documents")
def upload_documents(
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_service),
) -> Dict:
    try:
        return service.add_files(files)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"文档入库失败: {exc}",
        ) from exc


@app.delete("/api/v1/knowledge-base")
def clear_knowledge_base(service: RAGService = Depends(get_service)) -> Dict:
    return {"message": "知识库已清空", "knowledge_base": service.clear_knowledge_base()}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, service: RAGService = Depends(get_service)) -> Dict:
    try:
        return service.chat(request.question, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"问答生成失败: {exc}",
        ) from exc


@app.post("/api/v1/retrieval/debug")
def debug_retrieval(
    request: RetrievalDebugRequest,
    service: RAGService = Depends(get_service),
) -> Dict:
    return service.debug_retrieval(request.query, request.top_k)


@app.post("/api/v1/feedback")
def add_feedback(
    request: FeedbackRequest,
    service: RAGService = Depends(get_service),
) -> Dict:
    try:
        return service.add_feedback(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/api/v1/feedback")
def feedback_dashboard(service: RAGService = Depends(get_service)) -> Dict:
    return service.feedback_dashboard()


@app.post("/api/v1/evaluations/cases")
def add_evaluation_case(
    request: EvaluationCaseRequest,
    service: RAGService = Depends(get_service),
) -> Dict:
    try:
        return service.add_evaluation_case(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/api/v1/evaluations")
def evaluation_dashboard(service: RAGService = Depends(get_service)) -> Dict:
    return service.evaluation_dashboard()


@app.post("/api/v1/evaluations/run")
def run_evaluation(
    request: EvaluationRunRequest,
    service: RAGService = Depends(get_service),
) -> Dict:
    try:
        return service.run_evaluation(request.case_ids)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.delete("/api/v1/sessions/{session_id}")
def reset_session(session_id: str, service: RAGService = Depends(get_service)) -> Dict:
    cleared = service.reset_session(session_id)
    return {"session_id": session_id, "cleared": cleared}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
