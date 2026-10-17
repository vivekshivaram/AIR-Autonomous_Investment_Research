"""Test doubles for the specialist tests. No test touches the network or loads a model.

Ported from Engineer 2's branch: the LangChain fake chat models are replaced by a plain fake with the
project's ``chat`` / ``chat_json`` interface, and ``StubTools`` follows the integrated ``ToolCallResult``.
"""

import json

import pandas as pd

from src.models import Status, ToolCallResult


class FakeChatLLM:
    """Replays the given answers in order, then starts over, through ``chat_json`` and ``chat``.

    Dicts are what a schema-constrained reply parses to; strings are free text; None stands for a reply
    that cannot be parsed, which the news pipeline must treat as unusable.
    """

    provider, model, available = "fake", "fake-model", True

    def __init__(self, answers):
        self.answers = list(answers) or ["not json"]
        self.position = 0

    def _next(self):
        answer = self.answers[self.position % len(self.answers)]
        self.position += 1
        return answer

    def chat_json(self, messages, json_schema, **kwargs):
        answer = self._next()
        if isinstance(answer, dict):
            return answer
        try:
            parsed = json.loads(answer) if isinstance(answer, str) else None
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            raise ValueError("model reply is not a JSON object")
        return parsed

    def chat(self, messages, **kwargs):
        answer = self._next()
        return answer if isinstance(answer, str) else json.dumps(answer)


def fake_llm(*answers):
    return FakeChatLLM(answers)


class StubTools:
    """Stands in for a ToolSession: serves canned data, or an error for tools listed in ``failing``.

    ``allowed`` mirrors the tools the plan selected for the step (the earnings agent calls SEC filings
    only when they are allowed); by default it is every tool that has data or is listed as failing.
    """

    def __init__(self, data=None, failing=(), allowed=None):
        self.data, self.failing, self.log = data or {}, set(failing), []
        self.allowed = set(allowed) if allowed is not None else set(self.data) | self.failing

    def call(self, name, **arguments):
        if name in self.failing or name not in self.data:
            result = ToolCallResult(name, arguments, Status.FAILED, 0.0, error="DataToolError: unavailable")
        else:
            result = ToolCallResult(name, arguments, Status.SUCCESS, 0.0, data=self.data[name])
        self.log.append(result)
        return result


def price_frame(closes, start="2026-01-01"):
    index = pd.bdate_range(start, periods=len(closes), name="date")
    frame = pd.DataFrame({"close": closes}, index=index, dtype="float64")
    frame.attrs = {"source": "test", "retrieved_at": "2026-01-01T00:00:00+00:00"}
    return frame
