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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
# Text normalisation (shared by evidence matching and upload validation)
# ─────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_text(text: Optional[str]) -> str:
    """Lowercase, punctuation and whitespace collapsed to single spaces."""
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def text_tokens(text: Optional[str]) -> List[str]:
    return normalise_text(text).split()


def token_overlap(gold_text: Optional[str], chunk_text: Optional[str]) -> float:
    """Fraction of the gold passage's tokens (with multiplicity) found in the chunk."""
    gold = Counter(text_tokens(gold_text))
    total = sum(gold.values())
    if total == 0:
        return 0.0
    chunk = Counter(text_tokens(chunk_text))
    common = sum(min(n, chunk.get(tok, 0)) for tok, n in gold.items())
    return round(common / total, 4)


# ─────────────────────────────────────────────
# Usage accumulator
# ─────────────────────────────────────────────

class UsageTracker:
    """Accumulates token usage and cost across generation and judge calls."""

    def __init__(self):
        self._lock = threading.Lock()
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
        with self._lock:
            self._add(call, role)

    def _add(self, call, role: str) -> None:
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
        # The five judge questions are independent, so run them concurrently:
        # wall-clock becomes one judge call instead of five (matters for
        # reasoning judges that take several seconds per call).
        judge_jobs = {
            "context_relevance": lambda: context_relevance(judge, query, context, usage),
            "answer_faithfulness": lambda: answer_faithfulness(judge, context, answer, usage),
            "answer_groundedness": lambda: answer_groundedness(judge, context, answer, usage),
            "answer_relevance": lambda: answer_relevance(judge, query, answer, usage),
            "answer_correctness": lambda: answer_correctness(judge, query, reference_answer, answer, usage),
        }
        with ThreadPoolExecutor(max_workers=len(judge_jobs)) as pool:
            futures = {name: pool.submit(fn) for name, fn in judge_jobs.items()}
            for name, fut in futures.items():
                try:
                    result[name] = fut.result()
                except Exception:
                    result[name] = None
        result.update(usage.as_dict())

    return result


# ─────────────────────────────────────────────
# Evidence matching (user test sets)
#
# Strategies use different chunk sizes, so a retrieved chunk is never compared
# to gold by chunk id. It is a HIT for a gold evidence item when it comes from
# the same document, its page range overlaps the gold pages (when both sides
# have pages) and, if the gold item has a passage, at least
# `overlap_threshold` of the passage's tokens are in the chunk (or the char
# ranges overlap by that fraction). Without a passage the page/document match
# counts, flagged as coarse.
# ─────────────────────────────────────────────

DEFAULT_OVERLAP_THRESHOLD = 0.5


def _same_document(chunk: Dict, gold: Dict) -> bool:
    cid, gid = chunk.get("document_id"), gold.get("document_id")
    if cid is not None and gid is not None:
        return int(cid) == int(gid)
    csrc, gsrc = chunk.get("source"), gold.get("document")
    if not csrc or not gsrc:
        return False
    csrc, gsrc = str(csrc).strip().lower(), str(gsrc).strip().lower()
    return csrc == gsrc or csrc.rsplit("/", 1)[-1] == gsrc.rsplit("/", 1)[-1]


def _pages_overlap(chunk: Dict, gold: Dict) -> Optional[bool]:
    """True/False when both sides carry pages, None when the check does not apply."""
    gs, ge = gold.get("page_start"), gold.get("page_end")
    cs, ce = chunk.get("page_start"), chunk.get("page_end")
    if gs is None or cs is None:
        return None
    ge = ge if ge is not None else gs
    ce = ce if ce is not None else cs
    return int(cs) <= int(ge) and int(ce) >= int(gs)


def char_overlap_fraction(chunk: Dict, gold: Dict) -> Optional[float]:
    """Fraction of the gold char range covered by the chunk; None when ranges are missing."""
    gs, ge = gold.get("char_start"), gold.get("char_end")
    cs, ce = chunk.get("char_start"), chunk.get("char_end")
    if None in (gs, ge, cs, ce):
        return None
    gs, ge, cs, ce = int(gs), int(ge), int(cs), int(ce)
    if ge <= gs:
        return None
    covered = max(0, min(ge, ce) - max(gs, cs))
    return round(covered / (ge - gs), 4)


def match_chunk_to_evidence(chunk: Dict, gold: Dict, overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> Dict:
    """
    Decide whether one retrieved chunk is a hit for one gold evidence item.

    chunk: {document_id, source, page_start, page_end, char_start, char_end, text}
    gold:  {document_id, document, page_start, page_end, passage, [char_start, char_end]}
    Returns {"hit": bool, "coarse": bool, "overlap": float, "reason": str}.
    """
    if not _same_document(chunk, gold):
        return {"hit": False, "coarse": False, "overlap": 0.0, "reason": "different document"}
    pages = _pages_overlap(chunk, gold)
    if pages is False:
        return {"hit": False, "coarse": False, "overlap": 0.0, "reason": "pages do not overlap"}

    passage = gold.get("passage")
    if passage and str(passage).strip():
        tok = token_overlap(passage, chunk.get("text"))
        chars = char_overlap_fraction(chunk, gold)
        best = max(tok, chars or 0.0)
        if best >= overlap_threshold:
            how = "token overlap" if tok >= (chars or 0.0) else "char-range overlap"
            return {"hit": True, "coarse": False, "overlap": best, "reason": f"{how} {best:.0%}"}
        return {"hit": False, "coarse": False, "overlap": best,
                "reason": f"text overlap {best:.0%} below threshold {overlap_threshold:.0%}"}

    if pages is True:
        return {"hit": True, "coarse": True, "overlap": 1.0, "reason": "page overlap only (no passage): coarse match"}
    return {"hit": True, "coarse": True, "overlap": 1.0, "reason": "same document only (no pages/passage): coarse match"}


def match_retrieved_chunks(
    chunks: List[Dict],
    gold_items: List[Dict],
    k: int,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> Dict:
    """
    Match every retrieved chunk (rank order) against every gold evidence item
    and compute the retrieval metrics of section 5 from the matches.

    Returns:
      per_chunk: [{rank, hit, matched_gold: [gold index...], coarse, overlap, reason, in_top_k}]
      gold_first_rank: {gold index: first rank that matched it (any rank)}
      hit_rate_at_k, recall_at_k, precision_at_k, mrr, ndcg_at_k, coarse_match, n_gold
    """
    per_chunk: List[Dict] = []
    gold_first_rank: Dict[int, int] = {}
    for i, chunk in enumerate(chunks):
        rank = int(chunk.get("rank") or (i + 1))
        matched: List[int] = []
        coarse = False
        best_overlap = 0.0
        reasons: List[str] = []
        for gi, gold in enumerate(gold_items):
            m = match_chunk_to_evidence(chunk, gold, overlap_threshold)
            if m["hit"]:
                matched.append(gi)
                coarse = coarse or m["coarse"]
                best_overlap = max(best_overlap, m["overlap"])
                reasons.append(f"gold {gi + 1}: {m['reason']}")
                gold_first_rank.setdefault(gi, rank)
            elif not reasons:
                reasons.append(f"gold {gi + 1}: {m['reason']}")
        per_chunk.append({
            "rank": rank,
            "hit": bool(matched),
            "matched_gold": matched,
            "coarse": coarse if matched else False,
            "overlap": best_overlap,
            "reason": "; ".join(reasons[:3]) if matched else (reasons[0] if reasons else "no gold evidence"),
            "in_top_k": rank <= k,
        })

    n_gold = len(gold_items)
    top = [c for c in per_chunk if c["in_top_k"]]
    matched_in_top = {gi for c in top for gi in c["matched_gold"]}
    hit_ranks = [c["rank"] for c in top if c["hit"]]

    # nDCG with binary gains: a chunk gains 1 only when it is the first in rank
    # order to cover some gold item, so several chunks covering the same passage
    # (small-chunk strategies) do not inflate the score. Ideal: one chunk per
    # gold item at the top.
    credited: set = set()
    dcg = 0.0
    for c in sorted(top, key=lambda c: c["rank"]):
        new = [gi for gi in c["matched_gold"] if gi not in credited]
        if new:
            dcg += 1.0 / math.log2(c["rank"] + 1)
            credited.update(new)
    ideal_n = min(n_gold, max(k, 0))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_n + 1))

    return {
        "per_chunk": per_chunk,
        "gold_first_rank": gold_first_rank,
        "n_gold": n_gold,
        "hit_rate_at_k": 1.0 if hit_ranks else 0.0,
        "recall_at_k": round(len(matched_in_top) / n_gold, 4) if n_gold else 0.0,
        "precision_at_k": round(sum(1 for c in top if c["hit"]) / k, 4) if k else 0.0,
        "mrr": round(1.0 / min(hit_ranks), 4) if hit_ranks else 0.0,
        "ndcg_at_k": round(dcg / idcg, 4) if idcg else 0.0,
        "coarse_match": any(c["coarse"] for c in top if c["hit"]),
    }


# ─────────────────────────────────────────────
# Abstention (unanswerable cases)
# ─────────────────────────────────────────────

_ABSTAIN_PATTERNS = [
    r"\bi\s+don'?t\s+know\b", r"\bdo(es)?\s+not\s+know\b", r"\bnot\s+(?:in|within)\s+the\s+(?:provided\s+)?(?:context|documents?|text|passages?)\b",
    r"\b(?:context|documents?|text|passages?|information\s+provided)\s+(?:do(?:es)?\s+not|doesn'?t|don'?t)\s+(?:contain|include|mention|provide|specify|state|say)\b",
    r"\bno\s+(?:information|mention|details?|answer)\b", r"\bcannot\s+(?:be\s+)?(?:answer|determine|find|locate)\w*\b",
    r"\bcan'?t\s+(?:answer|determine|find)\b", r"\bunable\s+to\s+(?:answer|determine|find)\b",
    r"\bnot\s+(?:mentioned|provided|specified|stated|available|found|present|addressed|covered)\b",
    r"\binsufficient\s+(?:information|context)\b", r"\bis\s+not\s+(?:contained|included)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstention(answer: Optional[str]) -> bool:
    """True when the generated answer says the documents do not contain the answer."""
    text = (answer or "").strip()
    if not text:
        return True
    head = text[:400]
    return bool(_ABSTAIN_RE.search(head))


def abstention_metrics(abstained_flags: List[Optional[bool]]) -> Dict:
    """Correct-abstention and false-answer rates over unanswerable cases (None = not judged)."""
    judged = [a for a in abstained_flags if a is not None]
    n = len(judged)
    if n == 0:
        return {"n": 0, "correct_abstention_rate": None, "false_answer_rate": None}
    correct = sum(1 for a in judged if a)
    return {
        "n": n,
        "correct_abstention_rate": round(correct / n, 4),
        "false_answer_rate": round((n - correct) / n, 4),
    }


# ─────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────

def bootstrap_ci(
    values: List[Optional[float]],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    min_size: Optional[int] = None,
) -> Dict:
    """
    Percentile bootstrap interval for the mean over test cases. Deterministic
    for a given seed. Below `min_size` cases the interval is not reported and
    enough_data is False ("not enough data" in the UI).
    """
    import random

    if min_size is None:
        try:
            from config import MIN_CASES_FOR_CI as min_size
        except Exception:
            min_size = 10
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    mean = round(sum(clean) / n, 4) if n else None
    out = {"mean": mean, "low": None, "high": None, "n": n, "confidence": confidence, "enough_data": n >= min_size and n > 1}
    if not out["enough_data"]:
        return out
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_i = max(0, int(math.floor(alpha * n_boot)))
    hi_i = min(n_boot - 1, int(math.ceil((1.0 - alpha) * n_boot)) - 1)
    out["low"] = round(means[lo_i], 4)
    out["high"] = round(means[hi_i], 4)
    return out


def format_ci(ci: Optional[Dict]) -> str:
    """'0.750 [0.600, 0.900]' or '0.750 (not enough data)' or '—'."""
    if not ci or ci.get("mean") is None:
        return "—"
    if not ci.get("enough_data"):
        return f"{ci['mean']:.3f} (not enough data)"
    return f"{ci['mean']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
