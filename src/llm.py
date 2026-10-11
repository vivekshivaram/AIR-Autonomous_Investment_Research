"""Local Qwen chat model for LangChain / LangGraph, with tool calling.

The model is loaded once with Hugging Face Transformers (CUDA, Apple MPS, or CPU) and wrapped
as a LangChain chat model. LangChain's stock Hugging Face wrapper cannot do tool calling for
a locally loaded model, so this wrapper passes the tools through the model's own chat template
and parses the ``<tool_call>`` blocks it answers with. Generation is greedy.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, convert_to_openai_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from .config import settings
from .utils import extract_json

TOOL_CALL_TAG = re.compile(r"</?tool_call>")


class LocalChatModel(BaseChatModel):
    """A local Transformers model as a LangChain chat model that supports ``bind_tools``."""

    model_id: str
    tokenizer: Any = Field(exclude=True, repr=False)
    model: Any = Field(exclude=True, repr=False)
    max_new_tokens: int = 320  # upper bound per answer; the model stops earlier at the end of its answer

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    def bind_tools(self, tools, **kwargs) -> Runnable:
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        import torch

        # Chat templates take OpenAI-style message dicts, which LangChain can produce from its own messages.
        # enable_thinking=False: Qwen3 can "think" before answering; switching that off gives direct, faster answers.
        inputs = self.tokenizer.apply_chat_template(
            convert_to_openai_messages(messages), tools=kwargs.get("tools"), add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Text before the first <tool_call> tag is the answer; every JSON object after a tag is one call.
        # Splitting on the tags (rather than matching pairs) also accepts the small model's occasional
        # habit of opening the next call without closing the previous one.
        content, *blocks = TOOL_CALL_TAG.split(text)
        turn = sum(isinstance(m, AIMessage) for m in messages)  # ids follow the position in the conversation,
        tool_calls = []  # so a repeated run produces identical messages
        for block in blocks:
            call = extract_json(block)
            if isinstance(call, dict) and call.get("name"):
                arguments = call.get("arguments")
                tool_calls.append(
                    {"name": call["name"], "args": arguments if isinstance(arguments, dict) else {}, "id": f"call_{turn}_{len(tool_calls)}"}
                )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content.strip(), tool_calls=tool_calls))])


def build_chat_model() -> LocalChatModel:
    """Load the Qwen model and return it as a LangChain chat model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()  # keeps progress widgets out of saved notebook output
    transformers_logging.set_verbosity_error()
    model_id = settings.llm_model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # Half precision halves memory and is faster on GPUs; CPUs need full precision.
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    # Greedy decoding: the same prompt always produces the same answer.
    model.generation_config.update(do_sample=False, temperature=None, top_p=None, top_k=None, max_length=None)
    return LocalChatModel(model_id=model_id, tokenizer=AutoTokenizer.from_pretrained(model_id), model=model)


def load_prompt(name: str) -> str:
    """Read a system prompt from the prompts/ directory."""
    return (settings.prompts_dir / f"{name}.txt").read_text(encoding="utf-8").strip()


def _json_object(text: str) -> dict:
    parsed = extract_json(text)
    return parsed if isinstance(parsed, dict) else {}


def text_chain(llm: BaseChatModel, system_prompt: str) -> Runnable:
    """user text -> model answer as plain text."""
    # The system prompt goes in as a message, not a template: the prompts contain literal JSON braces.
    prompt = ChatPromptTemplate.from_messages([SystemMessage(system_prompt), ("human", "{user}")])
    # A batch runs one prompt at a time: a single local model gains nothing from threads.
    return prompt | llm.with_config(max_concurrency=1) | StrOutputParser() | str.strip


def json_chain(llm: BaseChatModel, system_prompt: str) -> Runnable:
    """user text -> model answer parsed as a JSON object ({} when the answer is not valid JSON)."""
    return text_chain(llm, system_prompt) | _json_object
