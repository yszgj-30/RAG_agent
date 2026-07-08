"""
对话记忆管理层 —— 多轮对话上下文维护
基于 langchain_core 原生消息类型实现，零废弃模块依赖
"""
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ChatMemoryManager:
    """对话记忆管理器
    使用 langchain_core 原生 HumanMessage / AIMessage 存储对话历史，
    无需依赖 langchain.memory 等已废弃模块，兼容最新版 LangChain。
    """

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 最大保留对话轮数（超出后自动截断最早记录）
        """
        self.max_turns = max_turns
        self._messages: List[BaseMessage] = []

    # ── 添加消息 ──────────────────────────────
    def add_user_message(self, content: str) -> None:
        """记录用户提问"""
        self._messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        """记录 AI 回答"""
        self._messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self) -> None:
        """超出最大轮数时自动截断最早的对话记录"""
        max_count = self.max_turns * 2  # 每轮 = 用户消息 + AI 回答
        if len(self._messages) > max_count:
            self._messages = self._messages[-max_count:]

    # ── 获取历史 ──────────────────────────────
    def get_history(self) -> List[Dict[str, str]]:
        """获取当前会话的完整对话历史，返回字典列表"""
        history = []
        for msg in self._messages:
            role = "user" if msg.type == "human" else "assistant"
            history.append({"role": role, "content": str(msg.content)})
        return history

    def get_history_text(self) -> str:
        """获取格式化的对话历史文本，用于注入 Agent 提示词"""
        history = self.get_history()
        if not history:
            return "（当前为第一轮对话，无历史记录）"
        lines = []
        for i, turn in enumerate(history, 1):
            who = "用户" if turn["role"] == "user" else "助手"
            lines.append(f"[第{i}轮] {who}: {turn['content']}")
        return "\n".join(lines)

    # ── 管理操作 ──────────────────────────────
    def clear(self) -> None:
        """清空全部对话记忆"""
        self._messages.clear()

    def get_summary(self) -> str:
        """返回当前记忆状态摘要"""
        count = len(self._messages)
        turns = count // 2
        return f"当前会话: {turns} 轮对话, 共 {count} 条消息"

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0
