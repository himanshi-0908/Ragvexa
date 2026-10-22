"""
llm_handler.py — NVIDIA NIM chat models with token/cost tracking, per-user
API keys and automatic model fallback.

get_llm(model, api_key=...) returns an NvidiaChat wrapper around ChatNVIDIA.
When a call fails because the key cannot use that model (retired, not
enabled, forbidden), the wrapper asks the model registry for a working
replacement, switches to it once, and records the switch so the UI and the
benchmark report can show it. invoke_tracked() returns text plus token usage
so usage and cost are reported without guessing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import NVIDIA_API_KEY, LLM_MODEL, JUDGE_MODEL, MODEL_PRICES
from llm import model_registry

logger = logging.getLogger("ragvexa")

DEFAULT_MAX_TOKENS = 2048


class ModelUnavailable(RuntimeError):
    """The NVIDIA API rejected the configured model id for this key and no replacement worked."""


@dataclass
class LLMCall:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Optional[float]            # None when no price is configured for the model
    switched_from: Optional[str] = None  # set when the model was auto-replaced for this key


def explain_model_error(model: str, exc: Exception) -> Optional[str]:
    """Turn NVIDIA 401/403/404/410 responses into an actionable message, else None."""
    text = str(exc)
    if "[410]" in text or "end of life" in text:
        return (f"Model '{model}' has been retired by NVIDIA. Set LLM_MODEL / JUDGE_MODEL / "
                "EVAL_GENERATOR_MODELS in .env to a current id (run: python scripts/check_models.py --probe).")
    if "[404]" in text and "account" in text:
        return (f"Model '{model}' is not enabled for your NVIDIA key. Pick one your key can use "
                "(run: python scripts/check_models.py --probe) and set it in .env.")
    if "[403]" in text or "Authorization failed" in text:
        return (f"NVIDIA refused model '{model}' for this key (403). The key may lack access to this model; "
                "run: python scripts/check_models.py --probe and set a working id in .env.")
    if "[401]" in text:
        return "NVIDIA rejected the API key (401). Check NVIDIA_API_KEY in .env or the key you entered."
    return None


def _status_from_error(exc: Exception) -> object:
    text = str(exc)
    for code in (401, 403, 404, 410, 429):
        if f"[{code}]" in text:
            return code
    return "ERR"


class NvidiaChat:
    """
    Thin wrapper over ChatNVIDIA with one job beyond delegation: if the model
    is unusable for this key, switch once to a suggested working model.

    Exposes .model, .invoke(), .switched_from so existing call sites (which
    only use those) keep working. The API key is held privately and never
    appears in repr/str.
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.2,
                 max_tokens: int = DEFAULT_MAX_TOKENS, allow_fallback: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.allow_fallback = allow_fallback
        self.switched_from: Optional[str] = None
        self.__api_key = api_key
        self._client = self._build(model)

    def _build(self, model: str) -> ChatNVIDIA:
        return ChatNVIDIA(
            nvidia_api_key=self.__api_key,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @property
    def key_fingerprint(self) -> str:
        return model_registry.key_fingerprint(self.__api_key)

    def __repr__(self) -> str:  # never leak the key
        return f"NvidiaChat(model={self.model!r}, switched_from={self.switched_from!r})"

    def invoke(self, messages, **kwargs):
        try:
            return self._client.invoke(messages, **kwargs)
        except Exception as exc:
            friendly = explain_model_error(self.model, exc)
            if friendly is None:
                raise
            status = _status_from_error(exc)
            if status == 401 or not self.allow_fallback:
                raise ModelUnavailable(friendly) from exc
            model_registry.mark_failed(self.__api_key, self.model, status, friendly)
            replacement = model_registry.suggest_model(self.__api_key, exclude=[self.model])
            if not replacement:
                raise ModelUnavailable(friendly + " No working replacement model was found for this key.") from exc
            logger.warning("Model %s unusable for key %s (%s); switching to %s",
                           self.model, self.key_fingerprint, status, replacement)
            self.switched_from = self.switched_from or self.model
            self.model = replacement
            self._client = self._build(replacement)
            return self._client.invoke(messages, **kwargs)


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """User-supplied key if given, else the server key. Raises if neither exists."""
    key = (api_key or "").strip() or (NVIDIA_API_KEY or "")
    if not key:
        raise ValueError(
            "No NVIDIA API key available. Add NVIDIA_API_KEY to .env (demo key) or enter your own key in the sidebar."
        )
    return key


def get_llm(model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = DEFAULT_MAX_TOKENS,
            api_key: Optional[str] = None, allow_fallback: bool = True) -> NvidiaChat:
    """
    Chat model for any NIM id. max_tokens is raised above the library's 1024
    default because reasoning models spend hundreds of tokens thinking before
    the visible answer; if the cap is hit the visible answer comes back empty.
    """
    return NvidiaChat(model or LLM_MODEL, resolve_api_key(api_key), temperature, max_tokens, allow_fallback)


def get_judge_llm(api_key: Optional[str] = None, model: Optional[str] = None) -> NvidiaChat:
    """Independent judge. Temperature 0 for repeatable scoring."""
    return get_llm(model or JUDGE_MODEL, temperature=0.0, api_key=api_key)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD cost from MODEL_PRICES (per 1M tokens). None when the model is unpriced."""
    price = MODEL_PRICES.get(model) if model else None
    if not isinstance(price, dict):
        return None
    try:
        return round(
            (input_tokens / 1_000_000) * float(price.get("input", 0))
            + (output_tokens / 1_000_000) * float(price.get("output", 0)),
            6,
        )
    except (TypeError, ValueError):
        return None


def _model_name(llm) -> str:
    return getattr(llm, "model", None) or LLM_MODEL


def extract_usage(response) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a LangChain AIMessage, else (0, 0)."""
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
    meta = getattr(response, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    return int(tu.get("prompt_tokens", 0) or 0), int(tu.get("completion_tokens", 0) or 0)


def invoke_tracked(llm, prompt: str) -> LLMCall:
    """Invoke the model and return text + usage + latency + estimated cost."""
    t0 = time.time()
    response = llm.invoke(prompt)
    latency_ms = int((time.time() - t0) * 1000)
    text = (getattr(response, "content", None) or str(response)).strip()
    in_tok, out_tok = extract_usage(response)
    model = _model_name(llm)
    return LLMCall(
        text=text,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=estimate_cost(model, in_tok, out_tok),
        switched_from=getattr(llm, "switched_from", None),
    )


def _answer_prompt(context, question, history_string=""):
    return f"""Use the following pieces of context and previous conversation history to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Previous Conversation History:
{history_string}

Context: 
{context}

Question: {question}
Answer:"""


def generate_response(llm, context, question, history_string=""):
    return invoke_tracked(llm, _answer_prompt(context, question, history_string)).text


def generate_response_tracked(llm, context, question, history_string="") -> LLMCall:
    return invoke_tracked(llm, _answer_prompt(context, question, history_string))


def generate_study_guide(llm, context):
    prompt = f"""Based on the provided context from the student's study materials, generate a comprehensive study guide.
Include:
1. Key Concepts (bulleted list)
2. Important Definitions
3. 5 Flashcard-style Questions and Answers for review.

Context:
{context}

Study Guide:"""
    return invoke_tracked(llm, prompt).text
