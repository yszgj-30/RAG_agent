"""
向量存储层 —— Chroma 本地向量数据库封装
使用 embeddings.dashscope_embedding.DashScopeEmbeddings 生成嵌入向量
通过 langchain_chroma.Chroma 管理向量库生命周期
"""
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings.dashscope_embedding import DashScopeEmbeddings


# Chroma 本地持久化目录
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "chroma_db"
)


class VectorStore:
    """Chroma 向量存储管理器

    嵌入层: DashScopeEmbeddings（text-embedding-v2 / 768维，中英双语优化）
    存储层: langchain_chroma.Chroma（本地持久化）
    """

    def __init__(
        self,
        api_key: str,
        collection_name: str = "enterprise_knowledge_base",
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ):
        self.api_key = api_key
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # 初始化嵌入函数 —— 完全基于 DashScope 原生 API
        self._embeddings = DashScopeEmbeddings(
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
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )
        formatted: List[Dict[str, Any]] = []
        for doc, score in results:
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "score": round(float(score), 4) if score is not None else 0.0,
            })
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted

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
