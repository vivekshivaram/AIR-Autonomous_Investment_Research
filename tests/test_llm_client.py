from src.llm_client import GroqClient, create_llm_client


def test_groq_provider_uses_requested_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    client = create_llm_client()

    assert isinstance(client, GroqClient)
    assert client.available
    assert client.model == "openai/gpt-oss-120b"
    assert client.config.base_url == "https://api.groq.com/openai/v1"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

    try:
        create_llm_client()
    except ValueError as exc:
        assert "choose groq or huggingface" in str(exc)
    else:
        raise AssertionError("An unsupported provider must not be selected")