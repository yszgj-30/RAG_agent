"""
Agent 工具层 —— 注册文档检索工具
工具是智能体与外部资源交互的接口，智能体通过调用工具获取知识库信息
"""
from langchain_core.tools import tool


# ── 全局向量库引用（由 Agent 初始化时注入） ──
_vector_store = None


def set_vector_store(store) -> None:
    """注入全局向量库实例，供检索工具使用"""
    global _vector_store
    _vector_store = store


@tool
def retrieve_knowledge(query: str) -> str:
    """从企业私有知识库中检索与查询语义相关的文档片段。
    适用场景：用户询问企业制度、业务流程、产品说明、员工规范等需要从内部文档查找答案的问题。
    当知识库无相关内容时，返回明确提示信息。

    Args:
        query: 用户的具体问题或搜索关键词（建议保留完整问句以获得更好检索效果）
    Returns:
        检索到的相关文档片段，包含来源文件与内容摘要
    """
    if _vector_store is None or _vector_store.is_empty():
        return "【提示】知识库尚未初始化或暂未收录文档，请先上传企业文档并构建知识库。"

    results = _vector_store.similarity_search(query, k=4)
    if not results:
        return "【提示】未在知识库中检索到与您问题相关的内容。请尝试调整提问方式，或补充上传相关文档。"

    parts = []
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "未知")
        parts.append(
            f"[资料{i}] 来源:《{source}》 相关度:{r['score']:.2%}\n内容:{r['content']}"
        )
    return "\n\n".join(parts)


def create_retrieval_tool(store) -> callable:
    """工厂函数：绑定向量库实例并返回检索工具"""
    set_vector_store(store)
    return retrieve_knowledge
