"""Optional local Transformers client implementing the shared chat interface."""

from __future__ import annotations

import json
from typing import Any

from .config import settings
from .llm_client import LLMError
from .utils import extract_json


class LocalQwenClient:
    provider = "local-transformers"
    available = True

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.llm_model
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def model(self) -> str:
        return self.model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMError(
                "The local backend requires PyTorch and transformers; install the optional local-model dependencies."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype).to(device).eval()
        self._model.generation_config.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        self._torch = torch

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str:
        self._load()
        prompt_messages = [dict(message) for message in messages]
        if json_schema is not None:
            schema_hint = f"Return one JSON object matching this schema ({schema_name}):\n{json.dumps(json_schema)}"
            if prompt_messages and prompt_messages[-1].get("role") == "user":
                prompt_messages[-1]["content"] += "\n\n" + schema_hint
            else:
                prompt_messages.append({"role": "user", "content": schema_hint})

        inputs = self._tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=max_tokens or 512)
        prompt_length = inputs["input_ids"].shape[1]
        return self._tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

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
        parsed = extract_json(text)
        if not isinstance(parsed, dict):
            raise LLMError("The local model returned invalid JSON.")
        return parsed
