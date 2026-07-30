"""
Agent 调度层 —— 基于 LangGraph 的企业知识库智能体。

流程：问题分类 → 普通对话 / 模型请求检索工具 → ToolNode 执行
     → 知识库回答 → 多轮记忆更新。
"""
import operator
import time
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import create_retrieval_tool
from memory.manager import ChatMemoryManager
from models.tongyi_llm import TongyiLLM


CLASSIFIER_PROMPT = """你是企业知识库智能体的问题分类器。
请根据用户当前问题和对话历史，只输出以下一个英文标签：
- knowledge：问题需要查询企业制度、业务流程、产品手册、技术文档或已上传资料
- chat：问候、闲聊、自我介绍、感谢等不需要查询知识库的普通对话

如果用户在追问上一轮知识库内容，输出 knowledge；无法确定时优先输出 knowledge。
不要输出解释、标点或其他内容。"""

KNOWLEDGE_PROMPT = """你是企业内部知识管理助手。
请严格基于知识库检索工具返回的资料回答当前问题：
- 不得补充检索资料之外的事实
- 引用资料时标注来源文件名
- 资料不足时先说明已知内容，再指出缺失信息
- 使用简练、专业的中文作答"""

CHAT_PROMPT = """你是友好、专业的企业智能助手。
当前问题被分类为普通对话，不需要检索知识库。请结合对话历史自然回答，
但不要声称已经查询企业资料，也不要编造企业内部信息。"""


class AgentState(TypedDict, total=False):
    question: str
    history: str
    intent: Literal["chat", "knowledge"]
    messages: Annotated[List[BaseMessage], add_messages]
    answer: str
    sources: List[Dict]
    error: Optional[str]
    thinking: Annotated[List[str], operator.add]
    trace: Annotated[List[Dict], operator.add]


class RAGAgent:
    """具备问题分类、工具调用、知识库问答与多轮记忆的 LangGraph 智能体。"""

    def __init__(self, llm: TongyiLLM, vector_store, memory: ChatMemoryManager):
        self.llm = llm
        self.vector_store = vector_store
        self.memory = memory
        self.retrieve_tool = create_retrieval_tool(vector_store)
        self.tool_llm = llm.bind_tools(
            [self.retrieve_tool],
            tool_choice="auto",
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("classify", self._classify)
        builder.add_node("chat", self._chat)
        builder.add_node("request_tool", self._request_tool)
        builder.add_node(
            "tools",
            ToolNode(
                [self.retrieve_tool],
                handle_tool_errors="知识库检索工具执行失败，请稍后重试。",
            ),
        )
        builder.add_node("knowledge_answer", self._knowledge_answer)
        builder.add_node("error", self._error)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_intent,
            {"chat": "chat", "knowledge": "request_tool"},
        )
        builder.add_conditional_edges(
            "chat",
            self._route_after_generation,
            {"done": END, "error": "error"},
        )
        builder.add_conditional_edges(
            "request_tool",
            self._route_after_tool_request,
            {"tools": "tools", "error": "error"},
        )
        builder.add_edge("tools", "knowledge_answer")
        builder.add_conditional_edges(
            "knowledge_answer",
            self._route_after_generation,
            {"done": END, "error": "error"},
        )
        builder.add_edge("error", END)
        return builder.compile()

    def query(self, question: str) -> Dict:
        """运行 LangGraph，并返回回答、来源、分类和执行轨迹。"""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        result = self.graph.invoke({
            "question": question,
            "history": self.memory.get_history_text(),
            "messages": [HumanMessage(content=question)],
            "sources": [],
            "thinking": [f"接收提问: {question}"],
            "trace": [{
                "step": "receive",
                "label": "接收问题",
                "status": "success",
                "detail": question,
                "duration_ms": 0.0,
            }],
            "error": None,
        })

        answer = result.get("answer", "系统暂时无法处理该问题，请稍后重试。")
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "thinking": result.get("thinking", []),
            "intent": result.get("intent", "knowledge"),
            "error": result.get("error"),
            "trace": result.get("trace", []),
        }

    def _classify(self, state: AgentState) -> Dict:
        started = time.perf_counter()
        try:
            response = self.llm.invoke([
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(
                    content=(
                        f"【对话历史】\n{state['history']}\n\n"
                        f"【当前问题】\n{state['question']}"
                    )
                ),
            ])
            label = str(response.content).strip().lower()
            intent: Literal["chat", "knowledge"] = (
                "chat" if label.startswith("chat") else "knowledge"
            )
            return {
                "intent": intent,
                "thinking": [
                    "问题分类: 普通对话" if intent == "chat"
                    else "问题分类: 知识库问答"
                ],
                "trace": [self._trace_event(
                    "classify",
                    "问题分类",
                    "success",
                    "普通对话" if intent == "chat" else "知识库问答",
                    started,
                )],
            }
        except Exception as exc:
            intent = self._fallback_intent(state["question"])
            return {
                "intent": intent,
                "thinking": ["分类模型不可用，已使用规则完成降级分类"],
                "trace": [self._trace_event(
                    "classify",
                    "问题分类",
                    "fallback",
                    f"规则降级：{exc}",
                    started,
                )],
            }

    @staticmethod
    def _fallback_intent(question: str) -> Literal["chat", "knowledge"]:
        chat_terms = (
            "你好", "您好", "早上好", "下午好", "晚上好",
            "谢谢", "感谢", "再见", "你是谁", "介绍一下你自己",
        )
        normalized = question.strip()
        if len(normalized) <= 30 and any(term in normalized for term in chat_terms):
            return "chat"
        return "knowledge"

    @staticmethod
    def _route_intent(state: AgentState) -> Literal["chat", "knowledge"]:
        return state.get("intent", "knowledge")

    @staticmethod
    def _route_after_generation(state: AgentState) -> Literal["done", "error"]:
        return "error" if state.get("error") else "done"

    @staticmethod
    def _route_after_tool_request(
        state: AgentState,
    ) -> Literal["tools", "error"]:
        if state.get("error"):
            return "error"
        return "tools" if tools_condition(state) == "tools" else "error"

    def _chat(self, state: AgentState) -> Dict:
        started = time.perf_counter()
        try:
            response = self.llm.invoke([
                SystemMessage(content=CHAT_PROMPT),
                HumanMessage(
                    content=(
                        f"【对话历史】\n{state['history']}\n\n"
                        f"【当前问题】\n{state['question']}"
                    )
                ),
            ])
            answer = str(response.content).strip()
            return {
                "answer": answer,
                "sources": [],
                "thinking": [f"普通对话回答生成完成 ({len(answer)} 字)"],
                "trace": [self._trace_event(
                    "generate",
                    "生成普通对话回答",
                    "success",
                    f"{len(answer)} 字",
                    started,
                )],
            }
        except Exception as exc:
            return {
                "error": f"普通对话生成失败: {exc}",
                "trace": [self._trace_event(
                    "generate",
                    "生成普通对话回答",
                    "error",
                    str(exc),
                    started,
                )],
            }

    def _request_tool(self, state: AgentState) -> Dict:
        started = time.perf_counter()
        try:
            response = self.tool_llm.invoke([
                SystemMessage(
                    content=(
                        "请调用 retrieve_knowledge 工具检索知识库。"
                        "工具参数 query 必须保留用户完整问题。"
                    )
                ),
                HumanMessage(content=state["question"]),
            ])
            if not isinstance(response, AIMessage) or not response.tool_calls:
                return {
                    "error": "模型未生成知识库检索工具调用",
                    "trace": [self._trace_event(
                        "tool_select",
                        "选择检索工具",
                        "error",
                        "模型未返回结构化 tool_calls",
                        started,
                    )],
                }
            tool_name = response.tool_calls[0]["name"]
            return {
                "messages": [response],
                "thinking": [
                    f"模型选择工具: {tool_name}"
                ],
                "trace": [self._trace_event(
                    "tool_select",
                    "选择检索工具",
                    "success",
                    tool_name,
                    started,
                )],
            }
        except Exception as exc:
            return {
                "error": f"工具调用生成失败: {exc}",
                "trace": [self._trace_event(
                    "tool_select",
                    "选择检索工具",
                    "error",
                    str(exc),
                    started,
                )],
            }

    def _knowledge_answer(self, state: AgentState) -> Dict:
        started = time.perf_counter()
        tool_message = next(
            (
                message for message in reversed(state.get("messages", []))
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        if tool_message is None:
            return {
                "error": "未收到知识库检索工具结果",
                "trace": [self._trace_event(
                    "retrieve",
                    "检索知识库",
                    "error",
                    "ToolNode 未返回工具消息",
                    started,
                )],
            }
        if tool_message.status == "error":
            return {
                "error": str(tool_message.content),
                "trace": [self._trace_event(
                    "retrieve",
                    "检索知识库",
                    "error",
                    str(tool_message.content),
                    started,
                )],
            }

        results = tool_message.artifact if isinstance(tool_message.artifact, list) else []
        sources = [
            {
                "source": item.get("metadata", {}).get("source", "未知"),
                "score": item.get("score", 0.0),
            }
            for item in results
        ]
        if not results:
            return {
                "answer": str(tool_message.content),
                "sources": [],
                "thinking": ["检索工具未返回相关文档"],
                "trace": [self._trace_event(
                    "retrieve",
                    "检索知识库",
                    "empty",
                    "未召回相关片段",
                    started,
                )],
            }

        generation_started = time.perf_counter()
        try:
            response = self.llm.invoke([
                SystemMessage(content=KNOWLEDGE_PROMPT),
                HumanMessage(
                    content=(
                        f"【对话历史】\n{state['history']}\n\n"
                        f"【当前问题】\n{state['question']}\n\n"
                        f"【知识库检索结果】\n{tool_message.content}"
                    )
                ),
            ])
            answer = str(response.content).strip()
            return {
                "answer": answer,
                "sources": sources,
                "thinking": [
                    f"检索工具返回 {len(results)} 条资料",
                    f"知识库回答生成完成 ({len(answer)} 字)",
                ],
                "trace": [
                    self._trace_event(
                        "retrieve",
                        "检索知识库",
                        "success",
                        f"召回 {len(results)} 个片段",
                        started,
                    ),
                    self._trace_event(
                        "generate",
                        "生成知识库回答",
                        "success",
                        f"{len(answer)} 字，引用 {len(sources)} 个来源",
                        generation_started,
                    ),
                ],
            }
        except Exception as exc:
            return {
                "error": f"知识库回答生成失败: {exc}",
                "trace": [self._trace_event(
                    "generate",
                    "生成知识库回答",
                    "error",
                    str(exc),
                    generation_started,
                )],
            }

    @classmethod
    def _error(cls, state: AgentState) -> Dict:
        return {
            "answer": "系统处理请求时出现异常，请稍后重试。",
            "sources": [],
            "thinking": [f"异常降级: {state.get('error', '未知错误')}"],
            "trace": [{
                "step": "fallback",
                "label": "异常降级",
                "status": "error",
                "detail": state.get("error", "未知错误"),
                "duration_ms": 0.0,
            }],
        }

    @staticmethod
    def _trace_event(
        step: str,
        label: str,
        status: str,
        detail: str,
        started: float,
    ) -> Dict:
        return {
            "step": step,
            "label": label,
            "status": status,
            "detail": detail,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
