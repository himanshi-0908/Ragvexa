"""
evaluator.py — RAG evaluation metrics.

Retrieval metrics require ground-truth relevant chunk IDs.
Generation metrics use an independent LLM judge with structured prompts,
scored 0.0–1.0. Token usage is captured for every call so the benchmark can
report usage and cost. No hardcoded fallback scores are used anywhere: a
judge failure yields None, never a made-up number.
"""

from __future__ import annotations

import math
import re
import time
from typing import Dict, List, Optional

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
# Usage accumulator
# ─────────────────────────────────────────────

class UsageTracker:
    """Accumulates token usage and cost across generation and judge calls."""

    def __init__(self):
        self.gen_input = 0
        self.gen_output = 0
        self.judge_input = 0
        self.judge_output = 0
        self.cost_usd: Optional[float] = 0.0
        self._priced_calls = 0
        self._calls = 0

    def add(self, call, role: str) -> None:
        if call is None:
            return
        self._calls += 1
        if role == "gen":
            self.gen_input += call.input_tokens
            self.gen_output += call.output_tokens
        else:
            self.judge_input += call.input_tokens
            self.judge_output += call.output_tokens
        if call.cost_usd is None:
            self.cost_usd = None  # any unpriced call → cost unknown, do not guess
        elif self.cost_usd is not None:
            self.cost_usd = round(self.cost_usd + call.cost_usd, 6)
            self._priced_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.gen_input + self.gen_output + self.judge_input + self.judge_output

    def as_dict(self) -> Dict:
        return {
            "gen_input_tokens": self.gen_input,
            "gen_output_tokens": self.gen_output,
            "judge_input_tokens": self.judge_input,
            "judge_output_tokens": self.judge_output,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd if self._calls else None,
        }


# ─────────────────────────────────────────────
# LLM-as-a-Judge Helpers
# ─────────────────────────────────────────────

def _parse_score(text: str, label: str = "score") -> Optional[float]:
    """Extract a float from an LLM response in format 'Score: X.X'."""
    match = re.search(rf"{label}\s*[:\-]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)), 1.0)
    match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", text)
    if match:
        return float(match.group(1))
    return None


def _llm_judge(llm, prompt: str, label: str = "Score", usage: Optional[UsageTracker] = None) -> Optional[float]:
    try:
        from llm.llm_handler import invoke_tracked
        call = invoke_tracked(llm, prompt)
        if usage is not None:
            usage.add(call, "judge")
        return _parse_score(call.text, label)
    except Exception:
        return None


_JUDGE_HEADER = (
    "You are a strict, impartial RAG evaluator. You did not write the answer. "
    "Return ONLY this line: Score: X.XX where X.XX is between 0.00 and 1.00.\n"
)


# ─────────────────────────────────────────────
# Generation Metrics (LLM-as-a-judge)
# ─────────────────────────────────────────────

def context_relevance(llm, query: str, context: str, usage: Optional[UsageTracker] = None) -> Optional[float]:
    """Is the retrieved context relevant to the user's query? 0.0–1.0"""
    if not context.strip():
        return 0.0
    prompt = (
        _JUDGE_HEADER
        + "Rate how relevant the Retrieved Context is to the User Query.\n"
        "0.00 = completely irrelevant, 1.00 = perfectly on-topic.\n\n"
        f"User Query: {query}\n\n"
        f"Retrieved Context: {context[:2000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt, usage=usage)


def answer_faithfulness(llm, context: str, answer: str, usage: Optional[UsageTracker] = None) -> Optional[float]:
    """Does the answer avoid contradicting or inventing beyond the context? 0.0–1.0"""
    if not context.strip() or not answer.strip():
        return 0.0
    prompt = (
        _JUDGE_HEADER
        + "Rate how faithfully the Generated Answer is supported by the Context. "
        "Penalise any claim that contradicts the context or is invented.\n"
        "0.00 = completely hallucinated, 1.00 = every claim is in the context.\n\n"
        f"Context: {context[:2000]}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt, usage=usage)


def answer_groundedness(llm, context: str, answer: str, usage: Optional[UsageTracker] = None) -> Optional[float]:
    """
    What fraction of the answer's factual claims can be traced to a specific
    sentence in the context? Differs from faithfulness: an answer can be
    non-contradictory (faithful) yet mostly unsupported (ungrounded).
    """
    if not context.strip() or not answer.strip():
        return 0.0
    prompt = (
        _JUDGE_HEADER
        + "Split the Generated Answer into its factual claims. For each claim decide "
        "whether a specific sentence in the Context directly supports it. "
        "Score = supported claims / total claims.\n\n"
        f"Context: {context[:2000]}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt, usage=usage)


def answer_relevance(llm, query: str, answer: str, usage: Optional[UsageTracker] = None) -> Optional[float]:
    """Does the answer address the user's question? 0.0–1.0"""
    if not answer.strip():
        return 0.0
    prompt = (
        _JUDGE_HEADER
        + "Rate how well the Generated Answer addresses the User Query.\n"
        "0.00 = completely off-topic, 1.00 = directly and fully answers the query.\n\n"
        f"User Query: {query}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt, usage=usage)


def answer_correctness(
    llm, query: str, reference_answer: Optional[str], answer: str, usage: Optional[UsageTracker] = None
) -> Optional[float]:
    """
    Does the answer agree with the reference answer? Requires a reference;
    returns None (not scored) when there is none.
    """
    if not reference_answer or not reference_answer.strip():
        return None
    if not answer.strip():
        return 0.0
    prompt = (
        _JUDGE_HEADER
        + "Compare the Generated Answer with the Reference Answer for the Query. "
        "Score semantic agreement on the facts, ignoring wording.\n"
        "0.00 = contradicts or misses the reference, 1.00 = fully matches it.\n\n"
        f"Query: {query}\n\n"
        f"Reference Answer: {reference_answer[:1000]}\n\n"
        f"Generated Answer: {answer[:1000]}\n\n"
        "Score:"
    )
    return _llm_judge(llm, prompt, usage=usage)


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
    judge_llm=None,
    reference_answer: Optional[str] = None,
) -> Dict:
    """
    Compute all metrics for a single query result from one pipeline.

    `llm` generates the answer; `judge_llm` scores it. When judge_llm is not
    given the generator is used (and the report flags that the judge is not
    independent). Returns a flat dict suitable for tabular display.
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
        "answer_groundedness": None,
        "answer_relevance": None,
        "answer_correctness": None,
        "generation_latency_ms": None,
        "answer": None,
        "generator_model": None,
        "judge_model": None,
        "judge_independent": None,
        "gen_input_tokens": 0,
        "gen_output_tokens": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
    }

    if include_llm_judges and llm:
        from llm.llm_handler import invoke_tracked

        judge = judge_llm or llm
        usage = UsageTracker()
        context = "\n\n".join(c.text for c in pipeline_result.chunks)

        t0 = time.time()
        try:
            ans_prompt = (
                "Answer the following question using only the provided context. "
                "If the context does not contain the answer, say you don't know.\n"
                f"Question: {query}\n\n"
                f"Context:\n{context[:3000]}\n\n"
                "Answer:"
            )
            call = invoke_tracked(llm, ans_prompt)
            usage.add(call, "gen")
            answer = call.text
            result["generator_model"] = call.model
        except Exception:
            answer = ""
        gen_latency = int((time.time() - t0) * 1000)

        result["answer"] = answer
        result["generation_latency_ms"] = gen_latency
        result["judge_model"] = getattr(judge, "model", None)
        result["judge_independent"] = bool(
            judge_llm is not None and getattr(judge, "model", None) != getattr(llm, "model", None)
        )
        result["context_relevance"] = context_relevance(judge, query, context, usage)
        result["answer_faithfulness"] = answer_faithfulness(judge, context, answer, usage)
        result["answer_groundedness"] = answer_groundedness(judge, context, answer, usage)
        result["answer_relevance"] = answer_relevance(judge, query, answer, usage)
        result["answer_correctness"] = answer_correctness(judge, query, reference_answer, answer, usage)
        result.update(usage.as_dict())

    return result
