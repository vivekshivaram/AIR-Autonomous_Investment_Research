import json
from types import SimpleNamespace

import pytest

from src.llm_client import GeminiClient, GroqClient, LLMError, create_llm_client

LLM_VARS = (
    "LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
    "GEMINI_BASE_URL", "GEMINI_REASONING_EFFORT", "HF_TOKEN", "HUGGINGFACE_API_KEY", "HF_MODEL", "HF_BASE_URL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """A local .env may hold real keys and settings; every test starts from the documented defaults."""
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class RecordingCompletions:
    """Stands in for ``client.chat.completions``; replies in order and records each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def attach(client, completions):
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client


def test_groq_provider_uses_requested_model(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")
    clean_env.setenv("GROQ_API_KEY", "test-key")

    client = create_llm_client()

    assert isinstance(client, GroqClient)
    assert client.available
    assert client.model == "openai/gpt-oss-120b"
    assert client.config.base_url == "https://api.groq.com/openai/v1"


def test_unknown_provider_is_rejected(clean_env):
    clean_env.setenv("LLM_PROVIDER", "unsupported-provider")

    with pytest.raises(ValueError, match="choose gemini, groq or huggingface"):
        create_llm_client()


def test_gemini_provider_defaults(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GEMINI_API_KEY", "test-key")

    client = create_llm_client()

    assert isinstance(client, GeminiClient)
    assert client.available
    assert client.model == "gemini-3.8-flash"
    assert client.config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert client.config.reasoning_effort == "low"
    assert "test-key" not in repr(client.config)


def test_auto_detect_picks_gemini_when_it_is_the_only_key(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(create_llm_client(), GeminiClient)


def test_gemini_never_sends_temperature_but_sends_reasoning_effort(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "test-key")
    completions = RecordingCompletions(["hello"])
    client = attach(GeminiClient(), completions)

    assert client.chat([{"role": "user", "content": "hi"}], temperature=0.0) == "hello"
    request = completions.requests[0]
    assert "temperature" not in request
    assert request["reasoning_effort"] == "low"
    assert request["model"] == "gemini-3.8-flash"


def test_gemini_retries_with_the_schema_in_the_prompt(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "test-key")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    completions = RecordingCompletions([RuntimeError("Unknown name additionalProperties"), "```json\n{\"ok\": true}\n```"])
    client = attach(GeminiClient(), completions)

    assert client.chat_json([{"role": "user", "content": "check"}], schema, schema_name="probe") == {"ok": True}
    first, second = completions.requests
    assert first["response_format"]["json_schema"]["name"] == "probe"
    assert "response_format" not in second
    assert json.dumps(schema) in second["messages"][-1]["content"]


def test_groq_does_not_retry_failed_schema_calls(clean_env):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    completions = RecordingCompletions([RuntimeError("bad request")])
    client = attach(GroqClient(), completions)

    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "check"}], {"type": "object"})
    assert len(completions.requests) == 1
