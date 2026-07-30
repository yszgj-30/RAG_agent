"""反馈与离线评测数据存储。

使用 SQLite 保存用户反馈、评测用例和评测运行记录，避免把运营数据
混入向量数据库，同时保持本地开发零额外依赖。
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> Path:
    data_dir = os.getenv("RAG_AGENT_DATA_DIR")
    root = Path(data_dir) if data_dir else Path.home() / ".rag_agent"
    return root / "operations.db"


class OperationsStore:
    """线程安全的知识运营数据仓库。"""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    comment TEXT NOT NULL DEFAULT '',
                    intent TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_cases (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_answer TEXT NOT NULL DEFAULT '',
                    expected_source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES evaluation_cases(id)
                );
                """
            )

    def add_feedback(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
        intent: str = "unknown",
    ) -> Dict:
        if rating not in (-1, 1):
            raise ValueError("rating 只能为 1 或 -1")
        item = {
            "id": uuid.uuid4().hex,
            "session_id": session_id.strip() or "anonymous",
            "question": question.strip(),
            "answer": answer.strip(),
            "rating": rating,
            "comment": comment.strip(),
            "intent": intent.strip() or "unknown",
            "created_at": _utc_now(),
        }
        if not item["question"] or not item["answer"]:
            raise ValueError("问题和回答不能为空")

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback
                (id, session_id, question, answer, rating, comment, intent, created_at)
                VALUES (:id, :session_id, :question, :answer, :rating, :comment, :intent, :created_at)
                """,
                item,
            )
        return item

    def list_feedback(self, limit: int = 50) -> List[Dict]:
        safe_limit = max(1, min(limit, 200))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def feedback_summary(self) -> Dict:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS negative
                FROM feedback
                """
            ).fetchone()
        total = int(row["total"] or 0)
        positive = int(row["positive"] or 0)
        negative = int(row["negative"] or 0)
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total, 4) if total else 0.0,
        }

    def add_evaluation_case(
        self,
        *,
        question: str,
        expected_answer: str = "",
        expected_source: str = "",
    ) -> Dict:
        item = {
            "id": uuid.uuid4().hex,
            "question": question.strip(),
            "expected_answer": expected_answer.strip(),
            "expected_source": expected_source.strip(),
            "created_at": _utc_now(),
        }
        if not item["question"]:
            raise ValueError("评测问题不能为空")
        if not item["expected_answer"] and not item["expected_source"]:
            raise ValueError("期望答案关键词和期望来源至少填写一项")

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_cases
                (id, question, expected_answer, expected_source, created_at)
                VALUES (:id, :question, :expected_answer, :expected_source, :created_at)
                """,
                item,
            )
        return item

    def list_evaluation_cases(self) -> List[Dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluation_cases ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_evaluation_case(self, case_id: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM evaluation_runs WHERE case_id = ?",
                (case_id,),
            )
            cursor = connection.execute(
                "DELETE FROM evaluation_cases WHERE id = ?",
                (case_id,),
            )
        return cursor.rowcount > 0

    def record_evaluation_run(
        self,
        *,
        case_id: str,
        answer: str,
        intent: str,
        sources: List[str],
        passed: bool,
        latency_ms: float,
    ) -> Dict:
        item = {
            "id": uuid.uuid4().hex,
            "case_id": case_id,
            "answer": answer,
            "intent": intent,
            "sources_json": json.dumps(sources, ensure_ascii=False),
            "passed": int(passed),
            "latency_ms": round(float(latency_ms), 2),
            "created_at": _utc_now(),
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs
                (id, case_id, answer, intent, sources_json, passed, latency_ms, created_at)
                VALUES (:id, :case_id, :answer, :intent, :sources_json, :passed, :latency_ms, :created_at)
                """,
                item,
            )
        return {**item, "sources": sources, "passed": bool(item["passed"])}

    def list_evaluation_runs(self, limit: int = 100) -> List[Dict]:
        safe_limit = max(1, min(limit, 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, c.question, c.expected_answer, c.expected_source
                FROM evaluation_runs r
                JOIN evaluation_cases c ON c.id = r.case_id
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json"))
            item["passed"] = bool(item["passed"])
            results.append(item)
        return results

    def evaluation_summary(self) -> Dict:
        with self._lock, self._connect() as connection:
            case_count = connection.execute(
                "SELECT COUNT(*) FROM evaluation_cases"
            ).fetchone()[0]
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed,
                    AVG(latency_ms) AS average_latency_ms
                FROM evaluation_runs
                """
            ).fetchone()
        total = int(row["total"] or 0)
        passed = int(row["passed"] or 0)
        return {
            "case_count": int(case_count),
            "run_count": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "average_latency_ms": round(float(row["average_latency_ms"] or 0), 2),
        }
