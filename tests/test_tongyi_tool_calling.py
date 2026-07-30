from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from models.tongyi_llm import TongyiLLM


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="lookup",
                    arguments='{"query":"annual leave"}',
                ),
            )],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_bind_tools_sends_schema_and_parses_tool_calls():
    @tool
    def lookup(query: str) -> str:
        """Look up enterprise knowledge."""
        return query

    completions = FakeCompletions()
    llm = TongyiLLM(api_key="test-key")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    bound_model = llm.bind_tools([lookup], tool_choice="lookup")
    response = bound_model.invoke([HumanMessage(content="annual leave")])

    assert response.tool_calls == [{
        "name": "lookup",
        "args": {"query": "annual leave"},
        "id": "call-1",
        "type": "tool_call",
    }]
    assert completions.request["tools"][0]["function"]["name"] == "lookup"
    assert completions.request["tool_choice"]["function"]["name"] == "lookup"


def test_tool_messages_convert_to_openai_format():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup",
                "args": {"query": "leave"},
                "id": "call-1",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content="five days", tool_call_id="call-1"),
    ]

    converted = TongyiLLM._convert_messages(messages)

    assert converted[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert converted[1] == {
        "role": "tool",
        "content": "five days",
        "tool_call_id": "call-1",
    }
