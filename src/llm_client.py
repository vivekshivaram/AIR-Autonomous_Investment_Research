"""Thin wrapper around OpenAI-compatible chat APIs with JSON-schema outputs."""

from __future__ import annotations

import json
import os
from typing import Any

from .config import LLMConfig


class LLMError(RuntimeError):
    """Raised for missing credentials, API failures, or unparsable model output."""


class OpenAICompatClient:
    """Lazy client for OpenAI-compatible chat endpoints: constructing it never touches the network or needs a key."""

    provider = "openai-compatible"
    key_name = "API key"
    # Gemini 3 and later reject sampling parameters, so a subclass can opt out of sending temperature.
    sends_temperature = True
    # Strict schema mode; providers that rarely honour it switch it off.
    strict_schema = True
    # Retry once with the schema written into the prompt when schema-constrained output fails.
    retry_json_in_prompt = False

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or self._default_config()
        self._client: Any = None

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.groq_from_env()

    def _response_format(self, json_schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema, "strict": self.strict_schema},
        }

    @property
    def available(self) -> bool:
        """True when credentials are configured; checking it never calls the API."""
        return self.config.is_configured

    @property
    def model(self) -> str:
        """Model name sent with every request."""
        return self.config.model

    def _get_client(self) -> Any:
        if not self.available:
            raise LLMError(f"{self.key_name} is not set; add it to your environment or .env file.")
        if self._client is None:
            import ssl

            import httpx
            from openai import OpenAI  # imported lazily so the module loads without network setup

            verify: Any = True
            try:
                import truststore  # uses the OS certificate store; needed behind TLS-inspecting proxies

                verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                pass

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                max_retries=self.config.max_retries,
                http_client=httpx.Client(verify=verify, timeout=self.config.timeout_s),
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str:
        """Return the assistant text. With ``json_schema`` the output is schema-constrained JSON."""
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if temperature is not None and self.sends_temperature:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if json_schema is not None:
            kwargs["response_format"] = self._response_format(json_schema, schema_name)
        try:
            response = self._get_client().chat.completions.create(**kwargs)
        except LLMError:
            raise
        except Exception as exc:  # SDK raises many exception types; callers only need one
            raise LLMError(f"{self.provider} API call failed: {type(exc).__name__}: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"{self.provider} returned an empty response.")
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        schema_name: str = "response",
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the reply parsed as a JSON object that follows ``json_schema``."""
        try:
            text = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
                schema_name=schema_name,
            )
            return _parse_json_object(text, self.provider)
        except LLMError:
            if not self.retry_json_in_prompt:
                raise
        # Some providers reject response_format or return fenced/loose JSON: retry with the schema in the prompt.
        hint = (
            "Reply with a single JSON object only, no prose or code fences, matching this JSON schema:\n"
            + json.dumps(json_schema)
        )
        text = self.chat([*messages, {"role": "user", "content": hint}], temperature=temperature, max_tokens=max_tokens)
        return _parse_json_object(text, self.provider)


class GroqClient(OpenAICompatClient):
    """Groq's OpenAI-compatible endpoint with strict JSON-schema output."""

    provider = "Groq"
    key_name = "GROQ_API_KEY"

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.groq_from_env()


class HuggingFaceClient(OpenAICompatClient):
    """Hugging Face Inference Providers router; needs a free HF token with 'Make calls to Inference Providers'."""

    provider = "HuggingFace"
    key_name = "HF_TOKEN"
    # Open models served by HF providers rarely honour strict mode.
    strict_schema = False
    retry_json_in_prompt = True

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.hf_from_env()


class GeminiClient(OpenAICompatClient):
    """Gemini through Google AI Studio's OpenAI-compatible endpoint; needs GEMINI_API_KEY.

    Gemini 3 models removed temperature and the other sampling parameters, so none are sent; the
    thinking depth comes from ``reasoning_effort`` instead. The compatibility layer is in beta, so a
    rejected schema request falls back to the schema written into the prompt.
    """

    provider = "Gemini"
    key_name = "GEMINI_API_KEY"
    sends_temperature = False
    retry_json_in_prompt = True

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.gemini_from_env()


def _parse_json_object(text: str, provider: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        try:
            parsed = json.loads(text[start : end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError as exc:
            raise LLMError(f"{provider} returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"{provider} returned JSON that is not an object.")
    return parsed


def create_llm_client() -> Any:
    """Pick an API client (Groq, then Gemini, then Hugging Face) unless LLM_PROVIDER names one.

    The local backend loads only when explicitly selected, because it downloads a model checkpoint.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "local":
        from .local_llm import LocalQwenClient

        return LocalQwenClient()
    if provider in {"hf", "huggingface"}:
        return HuggingFaceClient()
    if provider == "groq":
        return GroqClient()
    if provider in {"gemini", "google"}:
        return GeminiClient()
    if provider:
        raise ValueError(
            f"Unsupported LLM_PROVIDER {provider!r}; choose gemini, groq or huggingface, or explicitly select local."
        )
    groq = GroqClient()
    if groq.available:
        return groq
    for client in (GeminiClient(), HuggingFaceClient()):
        if client.available:
            return client
    return groq
