"""Test doubles shared by the test modules. No test touches the network or loads a model."""

import json

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.models import ToolCallResult


def fake_llm(*answers):
    """A LangChain chat model that replays the given answers in order, then starts over.

    Dicts are sent as JSON text, the way the real model answers; None stands for an unparsable answer.
    """
    return FakeListChatModel(responses=[a if isinstance(a, str) else json.dumps(a) for a in answers] or ["not json"])


class _ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted_llm(*turns):
    """A tool-calling chat model that replays the given turns in order.

    A turn is text (an answer), or a list of (subagent, arguments) pairs (tool calls).
    """
    messages = [
        AIMessage(turn) if isinstance(turn, str)
        else AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{i}_{j}"} for j, (name, args) in enumerate(turn)])
        for i, turn in enumerate(turns)
    ]
    return _ScriptedChatModel(messages=iter(messages))


class StubTools:
    """Stands in for ToolRegistry: serves canned data, or an error for tools listed in `failing`."""

    def __init__(self, data=None, failing=()):
        self.data, self.failing, self.log = data or {}, set(failing), []

    def call(self, name, **arguments):
        if name in self.failing or name not in self.data:
            result = ToolCallResult(tool_name=name, arguments=arguments, status="error", duration_seconds=0, error="DataToolError: unavailable")
        else:
            result = ToolCallResult(tool_name=name, arguments=arguments, status="success", duration_seconds=0, data=self.data[name])
        self.log.append(result)
        return result


def price_frame(closes, start="2026-01-01"):
    index = pd.bdate_range(start, periods=len(closes), name="date")
    frame = pd.DataFrame({"close": closes}, index=index, dtype="float64")
    frame.attrs = {"source": "test", "retrieved_at": "2026-01-01T00:00:00+00:00"}
    return frame
