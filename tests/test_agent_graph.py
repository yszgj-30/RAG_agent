import pytest
from langchain_core.messages import AIMessage

from agent.core import (
    CHAT_PROMPT,
    CLASSIFIER_PROMPT,
    KNOWLEDGE_PROMPT,
    RAGAgent,
)
from memory.manager import ChatMemoryManager


class FakeBoundModel:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages):
        self.llm.tool_requests += 1
        question = str(messages[-1].content)
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "retrieve_knowledge",
                "args": {"query": question},
                "id": "call-1",
                "type": "tool_call",
            }],
        )


class FakeLLM:
    def __init__(self, intent="knowledge", classification_error=False):
        self.intent = intent
        self.classification_error = classification_error
        self.tool_requests = 0

    def bind_tools(self, tools, tool_choice=None):
        assert tools[0].name == "retrieve_knowledge"
        assert tool_choice == "auto"
        return FakeBoundModel(self)

    def invoke(self, messages):
        system_prompt = str(messages[0].content)
        if system_prompt == CLASSIFIER_PROMPT:
            if self.classification_error:
                raise RuntimeError("classifier unavailable")
            return AIMessage(content=self.intent)
        if system_prompt == CHAT_PROMPT:
            return AIMessage(content="你好，我可以帮助你查询企业知识。")
        if system_prompt == KNOWLEDGE_PROMPT:
            return AIMessage(content="根据员工手册，员工每年享有5天年假。")
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class FakeVectorStore:
    def __init__(self, empty=False, fail=False):
        self.empty = empty
        self.fail = fail
        self.queries = []

    def is_empty(self):
        return self.empty

    def hybrid_search(self, query, k=3, candidate_k=6):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("vector database unavailable")
        return [{
            "content": "员工每年享有5天年假。",
            "metadata": {"source": "员工手册.txt"},
            "score": 0.92,
        }]


def build_agent(llm=None, store=None):
    return RAGAgent(
        llm=llm or FakeLLM(),
        vector_store=store or FakeVectorStore(),
        memory=ChatMemoryManager(max_turns=10),
    )


def test_chat_route_skips_retrieval_tool():
    llm = FakeLLM(intent="chat")
    store = FakeVectorStore()
    agent = build_agent(llm, store)

    result = agent.query("你好，请介绍一下自己")

    assert result["intent"] == "chat"
    assert result["sources"] == []
    assert store.queries == []
    assert llm.tool_requests == 0


def test_knowledge_route_calls_tool_and_returns_sources():
    llm = FakeLLM(intent="knowledge")
    store = FakeVectorStore()
    agent = build_agent(llm, store)

    result = agent.query("员工每年有多少天年假？")

    assert result["intent"] == "knowledge"
    assert result["sources"] == [{"source": "员工手册.txt", "score": 0.92}]
    assert store.queries == ["员工每年有多少天年假？"]
    assert llm.tool_requests == 1
    assert "retrieve_knowledge" in " ".join(result["thinking"])
    assert [item["step"] for item in result["trace"]] == [
        "receive",
        "classify",
        "tool_select",
        "retrieve",
        "generate",
    ]
    assert len(agent.memory.get_history()) == 2


def test_empty_knowledge_base_returns_actionable_message():
    agent = build_agent(store=FakeVectorStore(empty=True))

    result = agent.query("公司的报销流程是什么？")

    assert result["sources"] == []
    assert "上传企业文档" in result["answer"]
    assert result["error"] is None


def test_classifier_failure_uses_rule_fallback():
    llm = FakeLLM(classification_error=True)
    agent = build_agent(llm=llm)

    result = agent.query("你好")

    assert result["intent"] == "chat"
    assert "降级分类" in " ".join(result["thinking"])


def test_tool_failure_enters_error_fallback():
    agent = build_agent(store=FakeVectorStore(fail=True))

    result = agent.query("公司的年假规定是什么？")

    assert result["error"]
    assert result["sources"] == []
    assert "稍后重试" in result["answer"]
