"""
benchmark.py — RAG Pipeline Benchmark Runner

Orchestrates evaluation of the selected retrieval pipelines across the
user's evaluation queries with identical settings (same dataset, same K,
same generator and judge models, same prompts).

Typical call:
    report = run_benchmark(user_id=1, k=5, pipelines=PIPELINE_NAMES)

Each run records its full configuration (dataset fingerprint, model ids,
embedding model, code revision) and is persisted per user so it can be
reproduced and compared later. Results are also cached in st.session_state
to avoid re-computation within a session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from retriver.retriver import retrieve_pipeline, PIPELINE_NAMES
from utils.eval_dataset import load_eval_queries, dataset_fingerprint
from utils.evaluator import (
    DEFAULT_OVERLAP_THRESHOLD,
    abstention_metrics,
    bootstrap_ci,
    evaluate_query,
    format_ci,
    is_abstention,
    match_retrieved_chunks,
)

logger = logging.getLogger("ragvexa")

FAILURE_CATEGORIES = ("ok", "missing_evidence", "retrieval", "ranking", "generation")
GENERATION_FAIL_THRESHOLD = 0.5

_METRIC_KEYS = [
    "hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
    "retrieval_latency_ms", "reranking_latency_ms",
    "context_relevance", "answer_faithfulness", "answer_groundedness",
    "answer_relevance", "answer_correctness",
    "generation_latency_ms",
    "gen_input_tokens", "gen_output_tokens", "judge_input_tokens", "judge_output_tokens",
    "total_tokens", "cost_usd",
]


@dataclass
class BenchmarkReport:
    pipelines: List[str]
    k: int
    per_pipeline: Dict[str, Dict[str, Any]]   # aggregate metrics per pipeline
    per_query: List[Dict[str, Any]]            # full detail for query inspection
    total_queries: int
    total_duration_ms: int
    timestamp: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[int] = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _cache_key(user_id: int, k: int, pipelines: List[str], query_hash: str, extra: str = "") -> str:
    combo = f"{user_id}-{k}-{'-'.join(sorted(pipelines))}-{query_hash}-{extra}"
    return "benchmark_" + hashlib.md5(combo.encode()).hexdigest()[:12]


def _avg(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _sum(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean), 6) if clean else None


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
    except Exception:
        return None


def _session_cache():
    """st.session_state when running under Streamlit, else a module dict."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return st.session_state
    except Exception:
        pass
    global _LOCAL_CACHE
    try:
        return _LOCAL_CACHE
    except NameError:
        _LOCAL_CACHE = {}
        return _LOCAL_CACHE


# ─────────────────────────────────────────────
# Failure classification
# ─────────────────────────────────────────────

def classify_failure(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int,
    indexed_ids: Optional[set] = None,
    candidate_ids: Optional[List[str]] = None,
    metrics: Optional[Dict] = None,
) -> str:
    """
    Decide where a query failed for one pipeline.

    missing_evidence : none of the labelled chunks exist in the index any more
    retrieval        : labelled chunks exist but were not in the wider candidate pool
    ranking          : a labelled chunk was in the candidate pool but not in top-k
    generation       : retrieval succeeded but the judged answer is poor
    ok               : retrieval succeeded and (if judged) the answer is acceptable
    """
    relevant = set(relevant_ids)
    if not relevant:
        return "missing_evidence"
    if indexed_ids is not None and not (relevant & set(indexed_ids)):
        return "missing_evidence"

    top_k = set(retrieved_ids[:k])
    if relevant & top_k:
        if metrics:
            scores = [
                metrics.get("answer_faithfulness"),
                metrics.get("answer_groundedness"),
                metrics.get("answer_correctness"),
            ]
            judged = [s for s in scores if s is not None]
            if judged and min(judged) < GENERATION_FAIL_THRESHOLD:
                return "generation"
        return "ok"

    if candidate_ids is not None and (relevant & set(candidate_ids)):
        return "ranking"
    return "retrieval"


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_benchmark(
    user_id: int,
    k: int = 5,
    pipelines: Optional[List[str]] = None,
    include_llm_judges: bool = False,
    progress_callback=None,
    force_rerun: bool = False,
    reviewed_only: bool = False,
    persist: bool = True,
    enforce_quota: bool = True,
    candidate_pool_fn: Optional[Callable[[str], List[str]]] = None,
    api_key: Optional[str] = None,
    answer_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    test_set_id: Optional[int] = None,
    overlap_threshold: Optional[float] = None,
) -> BenchmarkReport:
    """
    Run the benchmark for the given user.

    With `test_set_id` the run uses one of the user's own test sets instead of
    the eval queries: the set is snapshotted first (immutable JSON stored in
    the run config), every strategy evaluates exactly that snapshot, and
    retrieval metrics come from evidence matching (same document, page
    overlap, passage token overlap >= `overlap_threshold`), never chunk ids.

    Args:
        user_id: authenticated user's DB ID (all data access is scoped to it)
        k: top-K used for all retrieval and metrics
        pipelines: pipeline names to evaluate (default: all)
        include_llm_judges: generate answers and run judge metrics (slow, costs tokens)
        progress_callback: optional callable(current_step, total_steps, message)
        force_rerun: skip session cache and re-run
        reviewed_only: use only human-reviewed ground truth
        persist: save the report to the benchmark_runs table
        enforce_quota: apply the per-user daily benchmark quota
        candidate_pool_fn: query -> wider candidate chunk ids (for failure analysis);
                           defaults to Naive retrieval at 4*k
        api_key: the user's own NVIDIA key; None means the server (demo) key
        answer_model / judge_model: override the configured model ids
        test_set_id: run on this user test set (snapshotted) instead of eval queries
        overlap_threshold: passage token-overlap needed for an evidence hit (default 0.5)
    """
    from utils.documents import check_daily_quota, record_usage

    if pipelines is None:
        pipelines = list(PIPELINE_NAMES)
    unknown = [p for p in pipelines if p not in PIPELINE_NAMES]
    if unknown:
        raise ValueError(f"Unknown pipelines: {unknown}")

    snapshot = None
    queries: List[Dict] = []
    if test_set_id is not None:
        from utils.test_sets import test_set_snapshot
        snapshot = test_set_snapshot(user_id, int(test_set_id))
        if snapshot is None:
            raise ValueError("Test set not found")
        if not snapshot["cases"]:
            return BenchmarkReport(
                pipelines=pipelines, k=k, per_pipeline={}, per_query=[],
                total_queries=0, total_duration_ms=0,
                config={"dataset_kind": "test_set", "test_set_id": test_set_id, "reason": "test set is empty"},
            )
        if overlap_threshold is None:
            try:
                from config import EVIDENCE_OVERLAP_THRESHOLD as overlap_threshold
            except Exception:
                overlap_threshold = DEFAULT_OVERLAP_THRESHOLD
        overlap_threshold = float(overlap_threshold)
        if not (0.0 < overlap_threshold <= 1.0):
            raise ValueError("overlap_threshold must be in (0, 1]")
        fingerprint = snapshot["fingerprint"]
        cache_extra = f"{include_llm_judges}-ts{test_set_id}-{overlap_threshold}"
    else:
        queries = load_eval_queries(user_id, reviewed_only=reviewed_only)
        if not queries:
            return BenchmarkReport(
                pipelines=pipelines, k=k, per_pipeline={}, per_query=[],
                total_queries=0, total_duration_ms=0,
                config={"reviewed_only": reviewed_only, "reason": "no eval queries"},
            )
        fingerprint = dataset_fingerprint(queries)
        cache_extra = f"{include_llm_judges}-{reviewed_only}"

    cache = _session_cache()
    cache_key = _cache_key(user_id, k, pipelines, fingerprint, cache_extra)
    if not force_rerun and cache_key in cache:
        return cache[cache_key]

    if enforce_quota:
        check_daily_quota(user_id, "benchmark")

    from config import LLM_MODEL, JUDGE_MODEL, MODEL_PRICES
    from retriver.vector_store import get_embedding_model_name, get_user_chunk_ids

    from llm.model_registry import key_display
    answer_model = answer_model or LLM_MODEL
    judge_model = judge_model or JUDGE_MODEL
    llm = judge_llm = None
    if include_llm_judges:
        from llm.llm_handler import get_llm, get_judge_llm
        llm = get_llm(answer_model, api_key=api_key)
        judge_llm = get_judge_llm(api_key=api_key, model=judge_model)

    indexed_ids = None
    if snapshot is None:  # chunk-id failure analysis applies to eval queries only
        try:
            indexed_ids = get_user_chunk_ids(user_id)
        except Exception as e:
            logger.warning("Could not load indexed chunk ids for failure analysis: %s", e)

    if candidate_pool_fn is None:
        def candidate_pool_fn(q: str) -> List[str]:
            try:
                wide = retrieve_pipeline(q, user_id, strategy="Naive", k=max(4 * k, 20))
                return [c.chunk_id for c in wide.chunks]
            except Exception:
                return []

    config = {
        "k": k,
        "pipelines": list(pipelines),
        "include_llm_judges": include_llm_judges,
        "reviewed_only": reviewed_only,
        "generator_model": answer_model if include_llm_judges else None,
        "judge_model": judge_model if include_llm_judges else None,
        "judge_independent": (judge_model != answer_model) if include_llm_judges else None,
        "api_key_source": "user" if api_key else "server",
        "api_key_display": key_display(api_key) if api_key else "server key",
        "embedding_model": get_embedding_model_name(),
        "dataset_fingerprint": fingerprint,
        "dataset_size": len(queries),
        "dataset_human_queries": sum(1 for q in queries if q.get("origin") == "human"),
        "dataset_reviewed_queries": sum(1 for q in queries if q.get("reviewed")),
        "dataset_generator_models": sorted({m for q in queries for m in q.get("generator_models", [])}),
        "prices_configured": bool(MODEL_PRICES),
        "code_revision": _git_revision(),
        "started_at": time.time(),
    }

    total_steps = len(queries) * len(pipelines)
    step = 0
    global_t0 = time.time()

    accum: Dict[str, Dict[str, list]] = {p: {key: [] for key in _METRIC_KEYS} for p in pipelines}
    failures: Dict[str, Dict[str, int]] = {p: {c: 0 for c in FAILURE_CATEGORIES} for p in pipelines}
    per_query_rows: List[Dict] = []
    extra_per_pipeline: Dict[str, Dict] = {}

    if snapshot is not None:
        from config import MIN_CASES_FOR_CI
        cases = snapshot["cases"]
        config.update({
            "dataset_kind": "test_set",
            "test_set_id": snapshot["test_set_id"],
            "test_set_name": snapshot["name"],
            "test_set_fingerprint": snapshot["fingerprint"],
            # Immutable copy of the cases every strategy was evaluated on.
            "test_set_snapshot": json.loads(json.dumps(snapshot, default=str)),
            "overlap_threshold": overlap_threshold,
            "min_cases_for_ci": MIN_CASES_FOR_CI,
            "dataset_size": len(cases),
            "dataset_human_queries": len(cases),
            "dataset_reviewed_queries": len(cases),
            "dataset_answerable": sum(1 for c in cases if c.get("answerable")),
            "dataset_unanswerable": sum(1 for c in cases if not c.get("answerable")),
        })
        total_steps = len(cases) * len(pipelines)
        per_query_rows, failures, extra_per_pipeline = _run_on_test_set(
            user_id, cases, pipelines, k, overlap_threshold, llm, judge_llm,
            include_llm_judges, progress_callback, accum, total_steps,
        )
        queries = cases  # for total_queries below

    for q_info in ([] if snapshot is not None else queries):
        query = q_info["query"]
        relevant_ids = q_info["relevant_chunk_ids"]
        candidate_ids: Optional[List[str]] = None  # computed lazily, once per query

        for pipeline_name in pipelines:
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"{pipeline_name}: {query[:50]}…")

            pipeline_result = retrieve_pipeline(query, user_id, strategy=pipeline_name, k=k)

            metrics = evaluate_query(
                query=query,
                pipeline_result=pipeline_result,
                relevant_chunk_ids=relevant_ids,
                k=k,
                llm=llm,
                include_llm_judges=include_llm_judges,
                judge_llm=judge_llm,
                reference_answer=q_info.get("reference_answer"),
            )
            retrieved_ids = [c.chunk_id for c in pipeline_result.chunks]
            if not (set(relevant_ids) & set(retrieved_ids[:k])) and candidate_ids is None:
                candidate_ids = candidate_pool_fn(query)
            category = classify_failure(relevant_ids, retrieved_ids, k, indexed_ids, candidate_ids, metrics)
            failures[pipeline_name][category] += 1

            metrics["query"] = query
            metrics["relevant_chunk_ids"] = relevant_ids
            metrics["reference_answer"] = q_info.get("reference_answer")
            metrics["origin"] = q_info.get("origin", "synthetic")
            metrics["reviewed"] = bool(q_info.get("reviewed"))
            metrics["failure_category"] = category
            metrics["retrieved_chunks"] = [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "text": c.text[:300],
                    "score": c.score,
                    "rank": c.rank,
                    "relevant": c.chunk_id in relevant_ids,
                }
                for c in pipeline_result.chunks
            ]
            per_query_rows.append(metrics)

            amap = accum[pipeline_name]
            for key in amap:
                val = metrics.get(key)
                if val is not None:
                    amap[key].append(val)

    per_pipeline: Dict[str, Dict] = {}
    for p in pipelines:
        amap = accum[p]
        agg = {key: _avg(amap[key]) for key in amap}
        # Tokens and cost are totals, not means
        for key in ("gen_input_tokens", "gen_output_tokens", "judge_input_tokens", "judge_output_tokens", "total_tokens"):
            agg[key] = int(_sum(amap[key]) or 0)
        # Cost is only reported when EVERY judged query had a priced cost
        n_judged = len(amap["generation_latency_ms"])
        agg["cost_usd"] = _sum(amap["cost_usd"]) if (n_judged and len(amap["cost_usd"]) == n_judged) else None
        agg["failures"] = failures[p]
        agg["pipeline"] = p
        agg.update(extra_per_pipeline.get(p, {}))
        per_pipeline[p] = agg

    total_duration_ms = int((time.time() - global_t0) * 1000)
    config["finished_at"] = time.time()
    switches = {}
    if llm is not None and getattr(llm, "switched_from", None):
        switches["generator"] = {"from": llm.switched_from, "to": llm.model}
        config["generator_model"] = llm.model
    if judge_llm is not None and getattr(judge_llm, "switched_from", None):
        switches["judge"] = {"from": judge_llm.switched_from, "to": judge_llm.model}
        config["judge_model"] = judge_llm.model
    if switches:
        config["model_switches"] = switches
        config["judge_independent"] = config["judge_model"] != config["generator_model"]

    report = BenchmarkReport(
        pipelines=list(pipelines),
        k=k,
        per_pipeline=per_pipeline,
        per_query=per_query_rows,
        total_queries=len(queries),
        total_duration_ms=total_duration_ms,
        config=config,
    )

    total_in = sum(per_pipeline[p]["gen_input_tokens"] + per_pipeline[p]["judge_input_tokens"] for p in pipelines)
    total_out = sum(per_pipeline[p]["gen_output_tokens"] + per_pipeline[p]["judge_output_tokens"] for p in pipelines)
    costs = [per_pipeline[p]["cost_usd"] for p in pipelines]
    total_cost = round(sum(costs), 6) if costs and all(c is not None for c in costs) else None
    try:
        record_usage(user_id, "benchmark", total_in, total_out, total_cost)
    except Exception:
        logger.exception("Failed to record benchmark usage")

    if persist:
        try:
            report.run_id = save_run(user_id, report)
        except Exception:
            logger.exception("Failed to persist benchmark run for user_id=%s", user_id)

    cache[cache_key] = report
    return report


# ─────────────────────────────────────────────
# Evidence-based evaluation on a user test set
# ─────────────────────────────────────────────

_CI_METRICS = ("hit_rate_at_k", "recall_at_k", "mrr", "ndcg_at_k",
               "answer_faithfulness", "answer_groundedness", "answer_correctness", "answer_relevance")


def _chunk_dict(c, snippet_chars: int = 300) -> Dict:
    """RetrievedChunk -> plain dict with origin fields (what the matcher and UI consume)."""
    return {
        "chunk_id": c.chunk_id,
        "source": c.source,
        "document_id": getattr(c, "document_id", None),
        "page_start": getattr(c, "page_start", None),
        "page_end": getattr(c, "page_end", None),
        "char_start": getattr(c, "char_start", None),
        "char_end": getattr(c, "char_end", None),
        "text": c.text,
        "snippet": c.text[:snippet_chars],
        "score": c.score,
        "rank": c.rank,
    }


def classify_evidence_failure(
    answerable: bool,
    hit_in_top_k: bool,
    evidence_indexed: bool,
    hit_in_candidates: Optional[bool],
    metrics: Optional[Dict] = None,
    abstained: Optional[bool] = None,
) -> str:
    """
    Failure stage for one test case x strategy, from evidence matching.

    Unanswerable case: ok when the system abstained (or was not judged),
                       generation when it answered anyway.
    missing_evidence : no gold document has chunks in the index any more
    retrieval        : gold evidence indexed but not in the wider candidate pool
    ranking          : in the candidate pool but outside top-k
    generation       : evidence retrieved but the judged answer is poor
    ok               : evidence retrieved and (if judged) the answer is acceptable
    """
    if not answerable:
        return "generation" if abstained is False else "ok"
    if not evidence_indexed:
        return "missing_evidence"
    if hit_in_top_k:
        if metrics:
            scores = [metrics.get("answer_faithfulness"), metrics.get("answer_groundedness"), metrics.get("answer_correctness")]
            judged = [s for s in scores if s is not None]
            if judged and min(judged) < GENERATION_FAIL_THRESHOLD:
                return "generation"
        return "ok"
    if hit_in_candidates:
        return "ranking"
    return "retrieval"


def _run_on_test_set(
    user_id: int,
    cases: List[Dict],
    pipelines: List[str],
    k: int,
    overlap_threshold: float,
    llm,
    judge_llm,
    include_llm_judges: bool,
    progress_callback,
    accum: Dict[str, Dict[str, list]],
    total_steps: int,
):
    """
    Evaluate every strategy on the same immutable list of cases. Retrieval
    metrics are computed by matching retrieved chunks to gold evidence; the
    per-case rows carry everything the evidence view needs.
    """
    from retriver.vector_store import get_user_indexed_document_ids

    try:
        indexed_docs: Optional[set] = get_user_indexed_document_ids(user_id)
    except Exception as e:
        logger.warning("Could not load indexed document ids for failure analysis: %s", e)
        indexed_docs = None

    def candidate_chunks(q: str) -> List[Dict]:
        try:
            wide = retrieve_pipeline(q, user_id, strategy="Naive", k=max(4 * k, 20))
            return [_chunk_dict(c) for c in wide.chunks]
        except Exception:
            return []

    failures: Dict[str, Dict[str, int]] = {p: {c: 0 for c in FAILURE_CATEGORIES} for p in pipelines}
    rows: List[Dict] = []
    abstain: Dict[str, List[Optional[bool]]] = {p: [] for p in pipelines}
    coarse_counts: Dict[str, int] = {p: 0 for p in pipelines}
    by_tag: Dict[str, Dict[str, Dict[str, list]]] = {p: defaultdict(lambda: defaultdict(list)) for p in pipelines}
    step = 0

    for case in cases:
        question = case["question"]
        gold = list(case.get("gold_evidence") or [])
        answerable = bool(case.get("answerable"))
        tags = list(case.get("tags") or [])
        candidates: Optional[List[Dict]] = None  # computed lazily, once per case
        evidence_indexed = True
        if answerable and indexed_docs is not None:
            evidence_indexed = any(
                (g.get("document_id") is not None and int(g["document_id"]) in indexed_docs) for g in gold
            )

        for pipeline_name in pipelines:
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"{pipeline_name}: {question[:50]}…")

            pipeline_result = retrieve_pipeline(question, user_id, strategy=pipeline_name, k=k)
            chunks = [_chunk_dict(c) for c in pipeline_result.chunks]

            metrics = evaluate_query(
                query=question,
                pipeline_result=pipeline_result,
                relevant_chunk_ids=[],
                k=k,
                llm=llm,
                include_llm_judges=include_llm_judges,
                judge_llm=judge_llm,
                reference_answer=case.get("expected_answer"),
            )

            abstained: Optional[bool] = None
            if metrics.get("answer") is not None:
                abstained = is_abstention(metrics["answer"])

            if answerable:
                match = match_retrieved_chunks(chunks, gold, k, overlap_threshold)
                hit_top = match["hit_rate_at_k"] > 0
                hit_cand: Optional[bool] = None
                if not hit_top and evidence_indexed:
                    if candidates is None:
                        candidates = candidate_chunks(question)
                    hit_cand = match_retrieved_chunks(candidates, gold, len(candidates) or 1, overlap_threshold)["hit_rate_at_k"] > 0
                for key in ("hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
                    metrics[key] = match[key]
                metrics["coarse_match"] = match["coarse_match"]
                metrics["n_gold"] = match["n_gold"]
                metrics["gold_first_rank"] = {str(gi): r for gi, r in match["gold_first_rank"].items()}
                per_chunk = match["per_chunk"]
                if match["coarse_match"]:
                    coarse_counts[pipeline_name] += 1
            else:
                # Unanswerable: excluded from retrieval metrics (None is skipped by _avg).
                for key in ("hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
                    metrics[key] = None
                metrics["coarse_match"] = False
                metrics["n_gold"] = 0
                metrics["gold_first_rank"] = {}
                hit_top, hit_cand = False, None
                per_chunk = [{"rank": c["rank"], "hit": False, "matched_gold": [], "coarse": False,
                              "overlap": 0.0, "reason": "unanswerable case", "in_top_k": True} for c in chunks]
                abstain[pipeline_name].append(abstained)

            category = classify_evidence_failure(answerable, hit_top, evidence_indexed, hit_cand, metrics, abstained)
            failures[pipeline_name][category] += 1

            metrics.update({
                "query": question,
                "case_id": case.get("case_id"),
                "answerable": answerable,
                "tags": tags,
                "expected_answer": case.get("expected_answer"),
                "reference_answer": case.get("expected_answer"),
                "gold_evidence": gold,
                "origin": "human",
                "reviewed": True,
                "relevant_chunk_ids": [],
                "abstained": abstained,
                "abstention_correct": (abstained if not answerable else None) if abstained is not None else None,
                "evidence_indexed": evidence_indexed,
                "failure_category": category,
                "retrieved_chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "document_id": c["document_id"],
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                        "char_start": c["char_start"],
                        "char_end": c["char_end"],
                        "text": c["snippet"],
                        "score": c["score"],
                        "rank": c["rank"],
                        "relevant": m["hit"],
                        "hit": m["hit"],
                        "matched_gold": m["matched_gold"],
                        "matched_gold_label": ", ".join(f"gold {gi + 1}" for gi in m["matched_gold"]) or None,
                        "coarse": m["coarse"],
                        "overlap": m["overlap"],
                        "reason": m["reason"],
                    }
                    for c, m in zip(chunks, per_chunk)
                ],
            })
            rows.append(metrics)

            amap = accum[pipeline_name]
            for key in amap:
                val = metrics.get(key)
                if val is not None:
                    amap[key].append(val)
            if answerable:
                for tag in tags:
                    for key in ("hit_rate_at_k", "recall_at_k", "mrr", "ndcg_at_k"):
                        by_tag[pipeline_name][tag][key].append(metrics[key])

    extra: Dict[str, Dict] = {}
    n_answerable = sum(1 for c in cases if c.get("answerable"))
    n_unanswerable = len(cases) - n_answerable
    for p in pipelines:
        amap = accum[p]
        extra[p] = {
            "n_cases": len(cases),
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "ci": {m: bootstrap_ci(amap[m]) for m in _CI_METRICS if amap.get(m)},
            "abstention": abstention_metrics(abstain[p]),
            "coarse_matches": coarse_counts[p],
            "by_tag": {
                tag: {"n": len(vals["recall_at_k"]), **{m: _avg(vals[m]) for m in ("hit_rate_at_k", "recall_at_k", "mrr", "ndcg_at_k")}}
                for tag, vals in sorted(by_tag[p].items())
            },
        }
    return rows, failures, extra


# ─────────────────────────────────────────────
# Persistence / export
# ─────────────────────────────────────────────

def report_to_dict(report: BenchmarkReport) -> Dict:
    return asdict(report)


def report_to_json(report: BenchmarkReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, default=str)


def save_run(user_id: int, report: BenchmarkReport) -> int:
    from utils.database import SessionLocal, BenchmarkRun
    db = SessionLocal()
    try:
        row = BenchmarkRun(
            user_id=user_id,
            config_json=json.dumps(report.config, default=str),
            summary_json=json.dumps(report.per_pipeline, default=str),
            report_json=report_to_json(report),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()


def list_runs(user_id: int, limit: int = 20) -> List[Dict]:
    from utils.database import SessionLocal, BenchmarkRun
    db = SessionLocal()
    try:
        rows = (
            db.query(BenchmarkRun)
            .filter(BenchmarkRun.user_id == user_id)
            .order_by(BenchmarkRun.created_at.desc())
            .limit(limit)
            .all()
        )
        out = []
        for r in rows:
            cfg = json.loads(r.config_json or "{}")
            out.append({
                "id": r.id,
                "created_at": r.created_at,
                "k": cfg.get("k"),
                "pipelines": cfg.get("pipelines", []),
                "judged": cfg.get("include_llm_judges"),
                "reviewed_only": cfg.get("reviewed_only"),
                "dataset_fingerprint": cfg.get("dataset_fingerprint"),
                "dataset_size": cfg.get("dataset_size"),
                "dataset": f"test set: {cfg.get('test_set_name')}" if cfg.get("dataset_kind") == "test_set" else "eval queries",
            })
        return out
    finally:
        db.close()


def load_run(user_id: int, run_id: int) -> Optional[BenchmarkReport]:
    """Load a persisted run. Returns None if it does not exist OR is not owned by user_id."""
    from utils.database import SessionLocal, BenchmarkRun
    db = SessionLocal()
    try:
        row = db.query(BenchmarkRun).filter(BenchmarkRun.id == run_id, BenchmarkRun.user_id == user_id).first()
        if row is None:
            return None
        data = json.loads(row.report_json)
        data["run_id"] = row.id
        return BenchmarkReport(**data)
    finally:
        db.close()


# ─────────────────────────────────────────────
# Presentation helpers
# ─────────────────────────────────────────────

def get_pipeline_summary_table(report: BenchmarkReport) -> List[Dict]:
    """Return a list of dicts suitable for st.dataframe display."""
    rows = []
    for p in report.pipelines:
        m = report.per_pipeline.get(p, {})

        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"

        cost = m.get("cost_usd")
        rows.append({
            "Pipeline": p,
            "Recall@K": fmt(m.get("recall_at_k")),
            "Precision@K": fmt(m.get("precision_at_k")),
            "MRR": fmt(m.get("mrr")),
            "nDCG@K": fmt(m.get("ndcg_at_k")),
            "Context Rel.": fmt(m.get("context_relevance")),
            "Faithfulness": fmt(m.get("answer_faithfulness")),
            "Groundedness": fmt(m.get("answer_groundedness")),
            "Ans. Rel.": fmt(m.get("answer_relevance")),
            "Correctness": fmt(m.get("answer_correctness")),
            "Retr. ms": int(m.get("retrieval_latency_ms") or 0),
            "Rerank ms": int(m.get("reranking_latency_ms") or 0),
            "Gen. ms": int(m.get("generation_latency_ms") or 0) if m.get("generation_latency_ms") else "—",
            "Tokens": int(m.get("total_tokens") or 0) or "—",
            "Cost (USD)": f"{cost:.4f}" if cost is not None else "n/a",
        })
    return rows


def is_test_set_report(report: BenchmarkReport) -> bool:
    return (report.config or {}).get("dataset_kind") == "test_set"


def get_test_set_summary_table(report: BenchmarkReport) -> List[Dict]:
    """Strategies x metrics with bootstrap intervals, abstention rates and failure counts."""
    rows = []
    for p in report.pipelines:
        m = report.per_pipeline.get(p, {})
        ci = m.get("ci") or {}
        ab = m.get("abstention") or {}

        def pct(v):
            return f"{v:.1%}" if v is not None else "—"

        rows.append({
            "Pipeline": p,
            f"Hit rate@{report.k}": format_ci(ci.get("hit_rate_at_k")),
            f"Recall@{report.k}": format_ci(ci.get("recall_at_k")),
            "MRR": format_ci(ci.get("mrr")),
            f"nDCG@{report.k}": format_ci(ci.get("ndcg_at_k")),
            "Correct abstention": pct(ab.get("correct_abstention_rate")),
            "False answer": pct(ab.get("false_answer_rate")),
            "Faithfulness": format_ci(ci.get("answer_faithfulness")),
            "Correctness": format_ci(ci.get("answer_correctness")),
            "Coarse matches": int(m.get("coarse_matches") or 0),
            "Retr. ms": int(m.get("retrieval_latency_ms") or 0),
            "Answerable / Unanswerable": f"{m.get('n_answerable', 0)} / {m.get('n_unanswerable', 0)}",
        })
    return rows


def get_tag_breakdown_table(report: BenchmarkReport) -> List[Dict]:
    """One row per (tag, pipeline) with the retrieval metrics over answerable cases with that tag."""
    rows = []
    for p in report.pipelines:
        for tag, vals in (report.per_pipeline.get(p, {}).get("by_tag") or {}).items():
            rows.append({
                "Tag": tag,
                "Pipeline": p,
                "Cases": vals.get("n", 0),
                f"Hit rate@{report.k}": vals.get("hit_rate_at_k"),
                f"Recall@{report.k}": vals.get("recall_at_k"),
                "MRR": vals.get("mrr"),
                f"nDCG@{report.k}": vals.get("ndcg_at_k"),
            })
    rows.sort(key=lambda r: (r["Tag"], r["Pipeline"]))
    return rows


def get_evidence_rows(
    report: BenchmarkReport,
    failed_only: bool = False,
    tag: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> List[Dict]:
    """Per-case x strategy rows for the evidence view, optionally filtered."""
    out = []
    for row in report.per_query:
        if pipeline and row.get("pipeline") != pipeline:
            continue
        if tag and tag not in (row.get("tags") or []):
            continue
        if failed_only and row.get("failure_category", "ok") == "ok":
            continue
        out.append(row)
    return out


def get_evidence_view(report: BenchmarkReport, case_id) -> Dict:
    """
    Everything the UI shows for one test case: the case itself and, per
    strategy, retrieved chunks with hit/miss marks, the matched gold item,
    answer, judge scores and failure label.
    """
    rows = [r for r in report.per_query if r.get("case_id") == case_id]
    if not rows:
        return {}
    first = rows[0]
    return {
        "case_id": case_id,
        "question": first.get("query"),
        "expected_answer": first.get("expected_answer"),
        "answerable": first.get("answerable"),
        "tags": first.get("tags") or [],
        "gold_evidence": first.get("gold_evidence") or [],
        "per_pipeline": [
            {
                "pipeline": r.get("pipeline"),
                "hit_rate_at_k": r.get("hit_rate_at_k"),
                "recall_at_k": r.get("recall_at_k"),
                "mrr": r.get("mrr"),
                "ndcg_at_k": r.get("ndcg_at_k"),
                "coarse_match": r.get("coarse_match"),
                "failure_category": r.get("failure_category"),
                "abstained": r.get("abstained"),
                "answer": r.get("answer"),
                "answer_faithfulness": r.get("answer_faithfulness"),
                "answer_groundedness": r.get("answer_groundedness"),
                "answer_correctness": r.get("answer_correctness"),
                "answer_relevance": r.get("answer_relevance"),
                "rewritten_query": r.get("rewritten_query"),
                "error": r.get("error"),
                "retrieved_chunks": r.get("retrieved_chunks") or [],
            }
            for r in rows
        ],
    }


def get_failure_analysis(report: BenchmarkReport) -> List[Dict]:
    """Return queries where ALL pipelines achieved Recall@K = 0."""
    recall_by_query: Dict[str, List[float]] = defaultdict(list)
    for row in report.per_query:
        recall_by_query[row["query"]].append(row.get("recall_at_k", 0.0))

    failures = []
    for query, recalls in recall_by_query.items():
        if all(r == 0.0 for r in recalls):
            failures.append({"query": query, "pipelines_tried": len(recalls)})
    return failures


def get_failure_breakdown(report: BenchmarkReport) -> List[Dict]:
    """Per-pipeline counts of each failure category (for a table)."""
    rows = []
    for p in report.pipelines:
        f = report.per_pipeline.get(p, {}).get("failures") or {}
        rows.append({"Pipeline": p, **{c: int(f.get(c, 0)) for c in FAILURE_CATEGORIES}})
    return rows


def get_limitations(report: BenchmarkReport) -> List[str]:
    """Plain-language caveats that must accompany any numbers from this report."""
    cfg = report.config or {}
    notes = []
    if cfg.get("dataset_kind") == "test_set":
        size = cfg.get("dataset_size", report.total_queries) or 0
        min_ci = cfg.get("min_cases_for_ci") or 0
        if size < min_ci:
            notes.append(f"Only {size} test cases: below the {min_ci} needed for a bootstrap confidence interval, "
                         "so no interval (and no verdict) is reported.")
        coarse = sum(int((report.per_pipeline.get(p, {}) or {}).get("coarse_matches") or 0) for p in report.pipelines)
        if coarse:
            notes.append(f"{coarse} case×strategy hits are coarse matches (gold item without a passage: matched on "
                         "document/page only). Add passages to those cases for precise matching.")
        if not cfg.get("include_llm_judges"):
            notes.append("No answers were generated, so abstention and answer-quality metrics are not available.")
        if cfg.get("include_llm_judges"):
            if cfg.get("judge_independent") is False:
                notes.append("The judge model is the same as the generator; judge scores may be biased.")
            notes.append("Judge scores are LLM opinions on a 0–1 scale and are not a substitute for human review.")
        for role, sw in (cfg.get("model_switches") or {}).items():
            notes.append(f"The {role} model '{sw['from']}' was not usable with the active API key; "
                         f"'{sw['to']}' was used instead.")
        return notes
    human = cfg.get("dataset_human_queries", 0) or 0
    reviewed = cfg.get("dataset_reviewed_queries", 0) or 0
    size = cfg.get("dataset_size", report.total_queries) or 0
    if size and reviewed < size:
        notes.append(
            f"{size - reviewed} of {size} queries are unreviewed synthetic labels written by an LLM. "
            "Retrieval metrics against them show relative pipeline differences, not absolute accuracy."
        )
    if size and human == 0:
        notes.append("No human-authored ground truth is included in this run.")
    if cfg.get("include_llm_judges"):
        if cfg.get("judge_independent") is False:
            notes.append("The judge model is the same as the generator; judge scores may be biased.")
        notes.append("Judge scores are LLM opinions on a 0–1 scale and are not a substitute for human review.")
    for role, sw in (cfg.get("model_switches") or {}).items():
        notes.append(f"The {role} model '{sw['from']}' was not usable with the active API key; "
                     f"'{sw['to']}' was used instead.")
    if cfg.get("include_llm_judges") and not cfg.get("prices_configured"):
        notes.append("Cost is reported as n/a because MODEL_PRICES_JSON is not configured; token counts are exact.")
    if size and size < 20:
        notes.append(f"Only {size} queries were evaluated; differences between pipelines may not be significant.")
    return notes
