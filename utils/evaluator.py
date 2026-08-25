"""
evaluator.py — Proper RAG evaluation metrics.

Retrieval metrics require ground-truth relevant chunk IDs.
Generation metrics use LLM-as-a-judge with structured prompts, scored 0.0–1.0.
No hardcoded fallback scores are used anywhere.
"""

from __future__ import annotations

import re
import math
import time
from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# Retrieval Metrics
# ─────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant chunks found in top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = sum(1 for rid in relevant_ids if rid in retrieved_top_k)
    return round(hits / len(relevant_ids), 4)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if not retrieved_ids or k == 0:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_top_k if rid in set(relevant_ids))
    return round(hits / k, 4)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank — rank of the first relevant result."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance: 1 or 0)."""
    relevant_set = set(relevant_ids)
    retrieved_top_k = retrieved_ids[:k]

    def _dcg(ids):
        return sum(
            (1.0 / math.log2(rank + 1)) if rid in relevant_set else 0.0
            for rank, rid in enumerate(ids, start=1)
        )

    actual_dcg = _dcg(retrieved_top_k)
    # Ideal: relevant docs ranked first
    ideal_ids = [rid for rid in relevant_ids if rid in relevant_set][:k]
    ideal_dcg = _dcg(ideal_ids)
    if ideal_dcg == 0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


# ─────────────────────────────────────────────
# LLM-as-a-Judge Helpers
# ─────────────────────────────────────────────

def _parse_score(text: str, label: str = "score") -> Optional[float]:
    """Extract a float from an LLM response in format 'Score: X.X'."""
    # Try labeled format first
    match = re.search(rf"{label}\s*[:\-]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)), 1.0)
    # Fallback: any float in [0,1]
    match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", text)
    if match:
        return float(match.group(1))
    return None


def _llm_judge(llm, prompt: str, label: str = "Score") -> Optional[float]:
    try:
        res = llm.invoke(prompt)
        return _parse_score(res.content.strip(), label)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Generation Metrics (LLM-as-a-judge)
# ─────────────────────────────────────────────

def context_relevance(llm, query: str, context: str) -> Optional[float]:
    """
    Is the retrieved context relevant to the user's query?
    Score: 0.0 (irrelevant) – 1.0 (highly relevant)
    """
    if not context.strip():
        return 0.0
    prompt = (
        "You are a strict RAG evaluator.\n"
        "Rate how relevant the Retrieved Context is to the User Query.\n"
        "Return ONLY this line: Score: X.XX where X.XX is between 0.00 and 1.00.\n"
        "0.00 = completely irrelevant, 1.00 = perfectly on-topic.\n\n"
        f"User Query: {query}\n\n"
        f"Retrieved Context: {context[:2000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt)


def answer_faithfulness(llm, context: str, answer: str) -> Optional[float]:
    """
    Is the answer grounded in the context (no hallucination)?
    Score: 0.0 (fully hallucinated) – 1.0 (fully grounded)
    """
    if not context.strip() or not answer.strip():
        return 0.0
    prompt = (
        "You are a strict RAG evaluator.\n"
        "Rate how faithfully the Generated Answer is supported by the Context.\n"
        "Return ONLY this line: Score: X.XX where X.XX is between 0.00 and 1.00.\n"
        "0.00 = completely hallucinated, 1.00 = every claim is in the context.\n\n"
        f"Context: {context[:2000]}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt)


def answer_relevance(llm, query: str, answer: str) -> Optional[float]:
    """
    Does the answer address the user's question?
    Score: 0.0 (off-topic) – 1.0 (directly answers)
    """
    if not answer.strip():
        return 0.0
    prompt = (
        "You are a strict RAG evaluator.\n"
        "Rate how well the Generated Answer addresses the User Query.\n"
        "Return ONLY this line: Score: X.XX where X.XX is between 0.00 and 1.00.\n"
        "0.00 = completely off-topic, 1.00 = directly and fully answers the query.\n\n"
        f"User Query: {query}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt)


# ─────────────────────────────────────────────
# Per-query evaluation (used by benchmark runner)
# ─────────────────────────────────────────────

def evaluate_query(
    query: str,
    pipeline_result,       # PipelineResult from retriver.retriver
    relevant_chunk_ids: List[str],
    k: int,
    llm=None,
    include_llm_judges: bool = True,
) -> Dict:
    """
    Compute all metrics for a single query result from one pipeline.
    Returns a flat dict suitable for tabular display.
    """
    retrieved_ids = [c.chunk_id for c in pipeline_result.chunks]

    rec = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
    prec = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
    m = mrr(retrieved_ids, relevant_chunk_ids)
    n = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)

    result = {
        "pipeline": pipeline_result.pipeline,
        "recall_at_k": rec,
        "precision_at_k": prec,
        "mrr": m,
        "ndcg_at_k": n,
        "retrieval_latency_ms": pipeline_result.retrieval_latency_ms,
        "reranking_latency_ms": pipeline_result.reranking_latency_ms,
        "rewritten_query": pipeline_result.rewritten_query,
        "error": pipeline_result.error,
        # Generation metrics populated below
        "context_relevance": None,
        "answer_faithfulness": None,
        "answer_relevance": None,
        "generation_latency_ms": None,
        "answer": None,
    }

    if include_llm_judges and llm:
        context = "\n\n".join(c.text for c in pipeline_result.chunks)
        # Generate answer
        t0 = time.time()
        try:
            from llm.llm_handler import get_llm
            answer_llm = llm
            ans_prompt = (
                f"Answer the following question using only the provided context.\n"
                f"Question: {query}\n\n"
                f"Context:\n{context[:3000]}\n\n"
                "Answer:"
            )
            answer = answer_llm.invoke(ans_prompt).content.strip()
        except Exception:
            answer = ""
        gen_latency = int((time.time() - t0) * 1000)

        result["answer"] = answer
        result["generation_latency_ms"] = gen_latency
        result["context_relevance"] = context_relevance(llm, query, context)
        result["answer_faithfulness"] = answer_faithfulness(llm, context, answer)
        result["answer_relevance"] = answer_relevance(llm, query, answer)

    return result
