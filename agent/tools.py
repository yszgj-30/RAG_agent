"""
Agent 工具层 —— 注册文档检索工具
工具是智能体与外部资源交互的接口，智能体通过调用工具获取知识库信息
"""
from typing import Callable

from langchain_core.tools import tool


def create_retrieval_tool(store) -> Callable:
    """创建绑定指定向量库的检索工具，避免多实例共享全局状态。"""

    @tool("retrieve_knowledge", response_format="content_and_artifact")
    def retrieve_knowledge(query: str):
        """从企业私有知识库检索与问题语义相关的文档片段。

        Args:
            query: 用户的完整问题或需要检索的关键词。
        """
        if store.is_empty():
            return (
                "知识库尚未收录文档，请先上传企业文档并构建知识库。",
                [],
            )

        results = store.hybrid_search(query, k=3, candidate_k=6)
        if not results:
            return (
                "未在知识库中检索到相关内容，请尝试调整问题或补充文档。",
                [],
            )

        parts = []
        for index, result in enumerate(results, 1):
            source = result["metadata"].get("source", "未知")
            parts.append(
                f"【参考资料{index}】来源:{source} "
                f"相关度:{result['score']:.2%}\n{result['content']}"
            )
        return "\n\n---\n\n".join(parts), results

    return retrieve_knowledge
