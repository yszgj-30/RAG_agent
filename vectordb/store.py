"""Chroma 存储与向量 + BM25/RRF 混合检索。"""
import os
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from embeddings.dashscope_embedding import DashScopeEmbeddings


# Chroma 本地持久化目录
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "chroma_db"
)


class VectorStore:
    """Chroma 向量存储管理器

    嵌入层: 默认 DashScopeEmbeddings，也可注入兼容的本地 Embeddings
    存储层: langchain_chroma.Chroma（本地持久化）
    检索层: 向量候选召回 + 字符级 BM25 排名 + RRF 融合
    """

    def __init__(
        self,
        api_key: str,
        collection_name: str = "enterprise_knowledge_base",
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_function: Optional[Embeddings] = None,
    ):
        self.api_key = api_key
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # 初始化嵌入函数 —— 完全基于 DashScope 原生 API
        self._embeddings = embedding_function or DashScopeEmbeddings(
            api_key=api_key,
            model="text-embedding-v2",
            text_type="document",
        )

        self._vectorstore: Chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
        )

    # ── 文档入库 ──────────────────────────────

    def add_documents(self, chunks: List[Document]) -> int:
        """将 LangChain Document 列表批量写入向量库

        每个 Document 的 page_content 必须是纯文本字符串，
        metadata 会随文档一起持久化。

        Args:
            chunks: List[Document] —— 文档对象列表
        Returns:
            成功入库的文档数量
        """
        if not chunks:
            return 0

        # 过滤：只保留 page_content 为非空字符串的 Document
        valid_docs: List[Document] = []
        for doc in chunks:
            content = doc.page_content if hasattr(doc, "page_content") else ""
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content:
                continue
            valid_docs.append(Document(
                page_content=content,
                metadata=doc.metadata if hasattr(doc, "metadata") else {},
            ))

        if not valid_docs:
            return 0

        # langchain_chroma.Chroma.add_documents 会：
        #   1. 从每个 Document 提取 page_content → texts
        #   2. 调用 self._embeddings.embed_documents(texts) → embeddings
        #   3. 调用 chromadb collection.add(ids, embeddings, documents, metadatas)
        self._vectorstore.add_documents(valid_docs)
        return len(valid_docs)

    # ── 语义检索 ──────────────────────────────

    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """语义相似度检索，返回 Top-K 相关文档片段"""
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        formatted: List[Dict[str, Any]] = []
        for doc, distance in results:
            safe_distance = max(float(distance or 0.0), 0.0)
            relevance = 1.0 / (1.0 + safe_distance)
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "score": round(relevance, 4),
            })
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted

    @staticmethod
    def _bm25_tokens(text: str) -> List[str]:
        """为中英混合业务文档生成轻量 BM25 词元。"""
        normalized = re.sub(r"\s+", "", str(text).lower())
        tokens: List[str] = []
        for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
            tokens.extend(run)
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        tokens.extend(re.findall(r"[a-z0-9.%-]+", normalized))
        return tokens

    @classmethod
    def _bm25_order(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[int]:
        """返回候选片段按 BM25 分数从高到低的索引。"""
        document_tokens = [
            cls._bm25_tokens(item["content"])
            for item in candidates
        ]
        query_tokens = set(cls._bm25_tokens(query))
        document_count = len(document_tokens)
        average_length = (
            sum(len(tokens) for tokens in document_tokens)
            / max(document_count, 1)
        )
        document_frequency = Counter(
            token
            for tokens in document_tokens
            for token in set(tokens)
        )

        scored = []
        for index, tokens in enumerate(document_tokens):
            term_frequency = Counter(tokens)
            document_length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = term_frequency[token]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1
                    + (
                        document_count
                        - document_frequency[token]
                        + 0.5
                    )
                    / (document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75
                    + 0.75 * document_length / max(average_length, 1)
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            scored.append((score, index))
        return [
            index
            for _, index in sorted(
                scored,
                key=lambda item: (-item[0], item[1]),
            )
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = 3,
        candidate_k: Optional[int] = None,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """融合向量排名与候选集 BM25 排名，返回精排后的 Top-K。"""
        safe_k = max(1, int(k))
        safe_candidate_k = max(
            safe_k,
            int(candidate_k or max(safe_k * 2, 6)),
        )
        candidates = self.similarity_search(query, k=safe_candidate_k)
        if not candidates:
            return []

        lexical_order = self._bm25_order(query, candidates)
        lexical_ranks = {
            index: rank
            for rank, index in enumerate(lexical_order, 1)
        }
        fused = []
        for index, item in enumerate(candidates):
            vector_rank = index + 1
            lexical_rank = lexical_ranks[index]
            fused_score = (
                1 / (rrf_k + vector_rank)
                + 1 / (rrf_k + lexical_rank)
            )
            fused.append((
                fused_score,
                {
                    **item,
                    "vector_score": item["score"],
                    "vector_rank": vector_rank,
                    "lexical_rank": lexical_rank,
                    "retrieval_strategy": "bm25_rrf",
                },
            ))

        fused.sort(
            key=lambda item: (
                -item[0],
                item[1]["vector_rank"],
            )
        )
        max_score = fused[0][0]
        results = []
        for fused_score, item in fused[:safe_k]:
            item["score"] = round(fused_score / max_score, 4)
            results.append(item)
        return results

    def search_as_context(self, query: str, k: int = 5) -> str:
        """将检索结果格式化为可直接注入 LLM 的参考上下文"""
        results = self.similarity_search(query, k)
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "未知")
            parts.append(
                f"[参考资料{i}] 来源: {source} | 相关度: {r['score']:.2%}\n"
                f"{r['content']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── 集合管理 ──────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """按来源聚合知识库中的文档及片段统计。"""
        try:
            data = self._vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return []

        grouped: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "source": "未知来源",
                "chunk_count": 0,
                "char_count": 0,
                "file_type": "",
                "ingested_at": "",
            }
        )
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        for content, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            source = str(metadata.get("source", "未知来源"))
            item = grouped[source]
            item["source"] = source
            item["chunk_count"] += 1
            item["char_count"] += len(content or "")
            item["file_type"] = str(metadata.get("file_type", ""))
            ingested_at = str(metadata.get("ingested_at", ""))
            if ingested_at > item["ingested_at"]:
                item["ingested_at"] = ingested_at
        return sorted(grouped.values(), key=lambda item: item["source"].lower())

    def get_document_chunks(
        self,
        source: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取指定来源的片段，供知识库预览与排障使用。"""
        safe_limit = max(1, min(limit, 500))
        try:
            data = self._vectorstore.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        chunks = []
        for item_id, content, metadata in zip(
            data.get("ids", []) or [],
            data.get("documents", []) or [],
            data.get("metadatas", []) or [],
        ):
            metadata = metadata or {}
            chunks.append({
                "id": item_id,
                "content": content or "",
                "char_count": len(content or ""),
                "chunk_index": int(metadata.get("chunk_index", len(chunks))),
                "metadata": metadata,
            })
        chunks.sort(key=lambda item: item["chunk_index"])
        return chunks[:safe_limit]

    def delete_document(self, source: str) -> int:
        """按来源删除文档的全部片段。"""
        try:
            data = self._vectorstore.get(where={"source": source})
            ids = data.get("ids", []) or []
            if ids:
                self._vectorstore.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def clear(self) -> None:
        """清空当前集合中的全部文档"""
        try:
            ids = self._vectorstore.get().get("ids", [])
            if ids:
                self._vectorstore.delete(ids=ids)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        try:
            data = self._vectorstore.get()
            doc_count = len(data.get("ids", []))
            return {
                "doc_count": doc_count,
                "collection": self.collection_name,
                "persist_dir": self.persist_dir,
            }
        except Exception:
            return {
                "doc_count": 0,
                "collection": self.collection_name,
                "persist_dir": self.persist_dir,
            }

    def is_empty(self) -> bool:
        """知识库是否为空"""
        return self.get_stats()["doc_count"] == 0

    def close(self) -> None:
        """关闭 Chroma 客户端，释放 Windows 上的持久化文件句柄。"""
        client = getattr(self._vectorstore, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        embedding_close = getattr(self._embeddings, "close", None)
        if callable(embedding_close):
            embedding_close()
