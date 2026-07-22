"""FastAPI 后端入口，复用现有 RAG 业务模块。"""
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from agent.core import RAGAgent
from document.processor import DocumentProcessor
from memory.manager import ChatMemoryManager
from models.tongyi_llm import TongyiLLM
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


class RAGService:
    """管理共享知识库和按 session_id 隔离的对话记忆。"""

    def __init__(self, api_key: str):
        self.vector_store = VectorStore(api_key=api_key)
        self.processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
        self.llm = TongyiLLM(
            api_key=api_key,
            model_name=os.getenv("QWEN_MODEL", "qwen3.7-plus"),
            temperature=0.1,
        )
        self._sessions: Dict[str, RAGAgent] = {}
        self._lock = threading.RLock()

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
                    for chunk in file_chunks:
                        chunk.metadata["source"] = filename
                        chunk.metadata.pop("file_path", None)
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


@app.delete("/api/v1/sessions/{session_id}")
def reset_session(session_id: str, service: RAGService = Depends(get_service)) -> Dict:
    cleared = service.reset_session(session_id)
    return {"session_id": session_id, "cleared": cleared}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
