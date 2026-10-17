import pytest

from src.llm_cache import CachedLLM, with_cache


class CountingLLM:
    provider = "fake"
    model = "fake-model"
    available = True

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return f"reply {self.calls}"

    def chat_json(self, messages, json_schema, **kwargs):
        self.calls += 1
        if messages[-1]["content"] == "boom":
            raise RuntimeError("provider down")
        return {"n": self.calls}


def test_identical_calls_are_served_from_disk(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    assert llm.chat(messages) == "reply 1"
    assert llm.chat(messages) == "reply 1"
    assert inner.calls == 1 and llm.hits == 1 and llm.misses == 1
    # A fresh wrapper over the same folder still hits: the cache survives the process.
    assert CachedLLM(CountingLLM(), tmp_path).chat(messages) == "reply 1"


def test_any_change_to_the_prompt_or_schema_is_a_miss(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    llm.chat_json(messages, {"type": "object"})
    llm.chat_json(messages, {"type": "object", "required": ["n"]})
    llm.chat_json([{"role": "user", "content": "other"}], {"type": "object"})
    assert inner.calls == 3


def test_errors_are_not_cached(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            llm.chat_json([{"role": "user", "content": "boom"}], {"type": "object"})
    assert inner.calls == 2


def test_with_cache_respects_switches(tmp_path, monkeypatch):
    inner = CountingLLM()
    assert with_cache(inner, tmp_path, ttl_hours=0) is inner

    monkeypatch.setenv("LLM_CACHE_HOURS", "0")
    assert with_cache(inner, tmp_path) is inner

    offline = CountingLLM()
    offline.available = False
    assert with_cache(offline, tmp_path, ttl_hours=24) is offline

    wrapped = with_cache(inner, tmp_path, ttl_hours=24)
    assert isinstance(wrapped, CachedLLM)
    assert (wrapped.provider, wrapped.model, wrapped.available) == ("fake", "fake-model", True)
    assert with_cache(wrapped, tmp_path, ttl_hours=24) is wrapped
