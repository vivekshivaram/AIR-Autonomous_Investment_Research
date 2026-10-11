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

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or self._default_config()
        self._client: Any = None

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.groq_from_env()

    @staticmethod
    def _response_format(json_schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
        }

    @property
    def available(self) -> bool:
        return self.config.is_configured

    @property
    def model(self) -> str:
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
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
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
        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            schema_name=schema_name,
        )
        return _parse_json_object(text, self.provider)


class GroqClient(OpenAICompatClient):
    provider = "Groq"
    key_name = "GROQ_API_KEY"

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.groq_from_env()


class HuggingFaceClient(OpenAICompatClient):
    """Hugging Face Inference Providers router; needs a free HF token with 'Make calls to Inference Providers'."""

    provider = "HuggingFace"
    key_name = "HF_TOKEN"

    @staticmethod
    def _default_config() -> LLMConfig:
        return LLMConfig.hf_from_env()

    @staticmethod
    def _response_format(json_schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        # Open models served by HF providers rarely honour strict mode.
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema, "strict": False},
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        schema_name: str = "response",
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        try:
            return super().chat_json(
                messages, json_schema, schema_name=schema_name, temperature=temperature, max_tokens=max_tokens
            )
        except LLMError:
            # Some providers reject response_format or return fenced/loose JSON: retry with the schema in the prompt.
            hint = (
                "Reply with a single JSON object only, no prose or code fences, matching this JSON schema:\n"
                + json.dumps(json_schema)
            )
            text = self.chat(
                [*messages, {"role": "user", "content": hint}], temperature=temperature, max_tokens=max_tokens
            )
            return _parse_json_object(text, self.provider)


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


def create_llm_client() -> OpenAICompatClient:
    """Pick from LLM_PROVIDER (groq | huggingface); default prefers configured Groq, then HF."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in {"hf", "huggingface"}:
        return HuggingFaceClient()
    if provider == "groq":
        return GroqClient()
    if provider:
        raise ValueError(f"Unsupported LLM_PROVIDER {provider!r}; choose groq or huggingface.")
    groq = GroqClient()
    if groq.available:
        return groq
    hf = HuggingFaceClient()
    return hf if hf.available else groq
