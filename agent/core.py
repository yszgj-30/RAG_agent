"""
Agent 调度层 —— RAG 智能体核心引擎
简化流程：用户提问 → 检索知识库 → 生成回答 + 多轮记忆
"""
from typing import List, Dict, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from models.tongyi_llm import TongyiLLM
from memory.manager import ChatMemoryManager
from agent.tools import create_retrieval_tool, set_vector_store


# ── 系统提示词 ──────────────────────────────

SYSTEM_PROMPT = """你是企业技术知识库智能问答助手，严格遵循以下规则：

【身份定位】
你是企业内部知识管理助手，专门回答基于已上传企业文档的问题。

【检索结果使用规则】
- 检索结果按语义相关度排序，即使相关度评分不高，也可能包含关键信息
- 优先提炼所有检索片段中的技术事实来回答问题，不要因评分低而忽略
- 中英文混合术语（如Tensor、Variable、PyTorch等）可能因嵌入模型限制导致匹配度偏低，但仍需认真分析片段内容
- 如果多个片段各包含部分答案，请综合各片段信息

【回答规范】
- 回答必须基于检索到的文档内容，不得凭空编造
- 引用文档时标注来源文件名
- 知识库完全无相关内容时，如实告知；若有部分相关但不够完整，先给出已有信息再说明不足
- 回答语言简练专业，条理清晰，使用中文作答
"""


# ── RAG 智能体核心类 ────────────────────────

class RAGAgent:
    """RAG 知识库问答智能体

    简化执行流程：
      接收提问 → 检索知识库 → 结合对话历史 → LLM 生成回答 → 更新记忆
    """

    def __init__(self, llm: TongyiLLM, vector_store, memory: ChatMemoryManager):
        self.llm = llm
        self.vector_store = vector_store
        self.memory = memory

        # 注入向量库到工具层
        set_vector_store(vector_store)
        self.retrieve_tool = create_retrieval_tool(vector_store)

        self.thinking_log: List[str] = []

    # ── 主查询入口 ────────────────────────────

    def query(self, question: str) -> Dict:
        """处理用户提问，返回答案、来源、思考过程

        Returns:
            {"answer": str, "sources": list, "thinking": list}
        """
        self.thinking_log = []
        self._log(f"接收提问: {question}")

        # ── 阶段1: 检索知识库 ──────────────────
        context, sources = self._retrieve(question)
        if context:
            self._log(f"检索到 {len(sources)} 条相关资料")
        else:
            self._log("知识库为空或无匹配内容，将如实告知用户")

        # ── 阶段2: 生成回答 ────────────────────
        answer = self._generate(question, context, sources)
        self._log(f"生成回答 ({len(answer)} 字)")

        # ── 阶段3: 更新对话记忆 ────────────────
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)

        return {
            "answer": answer,
            "sources": sources,
            "thinking": self.thinking_log,
        }

    # ── 检索 ──────────────────────────────────

    def _retrieve(self, question: str) -> Tuple[str, List[Dict]]:
        """语义检索知识库，返回格式化上下文与来源列表"""
        if self.vector_store.is_empty():
            return "", []

        raw_results = self.vector_store.similarity_search(question, k=6)
        # 放宽阈值至 0.0，保留所有检索结果由 LLM 自行判断相关性
        filtered = [r for r in raw_results if r.get("score", 0) >= 0.0]
        context = self._format_context(filtered)
        sources = [
            {"source": r["metadata"].get("source", "未知"), "score": r["score"]}
            for r in filtered
        ]
        return context, sources

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """将检索到的文本块组装为 LLM 可理解的参考上下文"""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "未知")
            parts.append(
                f"【参考资料{i}】来源:{src} 相关度:{r['score']:.2%}\n{r['content']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── 生成 ──────────────────────────────────

    def _generate(self, question: str, context: str, sources: List[Dict]) -> str:
        """基于检索资料与对话历史生成答案"""
        history_text = self.memory.get_history_text()

        if context:
            user_prompt = f"""【对话历史】
{history_text}

【当前问题】
{question}

【知识库检索结果（请严格据此回答）】
{context}

请基于以上检索资料回答问题。重要提示：
- 即使某些片段的相关度评分不高，也可能包含与问题相关的技术关键词或概念定义
- 请仔细阅读每个片段的内容，尝试从中提炼与问题相关的信息
- 对于中英文混合术语（如Tensor、Variable、PyTorch、API等），请在所有片段中搜索相关定义和说明

要求：
1. 答案必须来源于检索资料，不可凭空编造
2. 引用时标注来源文件名和相关度
3. 若资料不足以完全回答问题，先给出已有信息，再说明缺失部分
4. 回答要条理清晰、语言专业"""
        else:
            user_prompt = f"""【对话历史】
{history_text}

【当前问题】
{question}

【重要提示】
知识库中暂未找到与当前问题相关的文档资料。请如实告知用户当前无法从知识库回答该问题，建议补充上传相关文档。不得编造任何信息。"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ── 辅助 ──────────────────────────────────

    def _log(self, message: str) -> None:
        self.thinking_log.append(message)
