"""
llm_handler.py — NVIDIA NIM chat models with token/cost tracking.

get_llm(model) returns a ChatNVIDIA for any NIM model id. invoke_tracked()
wraps a call and returns the text plus token usage so the evaluator and
benchmark can report usage and (when prices are configured) cost without
guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import NVIDIA_API_KEY, LLM_MODEL, JUDGE_MODEL, MODEL_PRICES


@dataclass
class LLMCall:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Optional[float]   # None when no price is configured for the model


def get_llm(model: Optional[str] = None, temperature: float = 0.2):
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is not set in .env")
    return ChatNVIDIA(
        nvidia_api_key=NVIDIA_API_KEY,
        model=model or LLM_MODEL,
        temperature=temperature,
    )


def get_judge_llm():
    """Independent judge. Temperature 0 for repeatable scoring."""
    return get_llm(JUDGE_MODEL, temperature=0.0)


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
    )


def generate_response(llm, context, question, history_string=""):
    prompt = f"""Use the following pieces of context and previous conversation history to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Previous Conversation History:
{history_string}

Context:
{context}

Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    return response.content


def generate_response_tracked(llm, context, question, history_string="") -> LLMCall:
    prompt = f"""Use the following pieces of context and previous conversation history to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Previous Conversation History:
{history_string}

Context:
{context}

Question: {question}
Answer:"""
    return invoke_tracked(llm, prompt)


def generate_study_guide(llm, context):
    prompt = f"""Based on the provided context from the student's study materials, generate a comprehensive study guide.
Include:
1. Key Concepts (bulleted list)
2. Important Definitions
3. 5 Flashcard-style Questions and Answers for review.

Context:
{context}

Study Guide:"""
    response = llm.invoke(prompt)
    return response.content
