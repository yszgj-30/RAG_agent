"""
通义千问自定义LLM类 —— 模型层
适配LangChain框架BaseChatModel接口，封装DashScope OpenAI兼容API调用
使用Pydantic v2 PrivateAttr管理非序列化私有属性
"""
import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

from pydantic import Field, PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI


class TongyiLLM(BaseChatModel):
    """通义千问大模型 LangChain 适配封装
    通过 DashScope OpenAI 兼容接口调用 qwen 系列模型，
    支持同步生成与流式输出。
    """
    # ── 公开字段（Pydantic v2 Field）───────────
    api_key: str = Field(default="", description="DashScope API密钥")
    model_name: str = Field(default="qwen3.7-plus", description="模型名称")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    request_timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # ── 私有属性（不参与Pydantic序列化）───────
    _client: Optional[OpenAI] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "tongyi_qwen"

    @property
    def client(self) -> OpenAI:
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
        return self._client

    # ── 消息格式转换 ──────────────────────────
    @staticmethod
    def _convert_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """将 LangChain 消息列表转换为 OpenAI 兼容格式"""
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        converted = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                converted.append({
                    "role": "tool",
                    "content": str(msg.content),
                    "tool_call_id": msg.tool_call_id,
                })
                continue

            role = role_map.get(msg.type, "user")
            item: Dict[str, Any] = {
                "role": role,
                "content": str(msg.content),
            }
            if isinstance(msg, AIMessage) and msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call["args"], ensure_ascii=False
                            ),
                        },
                    }
                    for call in msg.tool_calls
                ]
            converted.append(item)
        return converted

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | type | BaseTool],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable:
        """将 LangChain 工具转换为 OpenAI 兼容工具定义并绑定到模型。"""
        formatted_tools = [convert_to_openai_tool(item) for item in tools]
        if tool_choice and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {
                "type": "function",
                "function": {"name": tool_choice},
            }
        return self.bind(
            tools=formatted_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # ── 核心生成方法 ──────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步调用通义千问，返回完整生成结果"""
        formatted = self._convert_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            **kwargs,
        )
        response_message = response.choices[0].message
        content = response_message.content or ""
        tool_calls = []
        for call in response_message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"query": call.function.arguments or ""}
            tool_calls.append({
                "name": call.function.name,
                "args": arguments,
                "id": call.id,
                "type": "tool_call",
            })
        generation = ChatGeneration(
            message=AIMessage(content=content, tool_calls=tool_calls)
        )
        return ChatResult(generations=[generation])

    # ── 流式生成 ──────────────────────────────
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用通义千问，逐步返回生成片段"""
        formatted = self._convert_messages(messages)
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
