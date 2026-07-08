"""
通义千问自定义LLM类 —— 模型层
适配LangChain框架BaseChatModel接口，封装DashScope OpenAI兼容API调用
使用Pydantic v2 PrivateAttr管理非序列化私有属性
"""
from typing import Any, List, Optional, Iterator, Dict
from pydantic import Field, PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
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
            )
        return self._client

    # ── 消息格式转换 ──────────────────────────
    @staticmethod
    def _convert_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将 LangChain 消息列表转换为 OpenAI 兼容格式"""
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        converted = []
        for msg in messages:
            role = role_map.get(msg.type, "user")
            converted.append({"role": role, "content": str(msg.content)})
        return converted

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
        content = response.choices[0].message.content or ""
        generation = ChatGeneration(message=AIMessage(content=content))
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
