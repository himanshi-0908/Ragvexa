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
from utils.evaluator import evaluate_query

logger = logging.getLogger("ragvexa")

FAILURE_CATEGORIES = ("ok", "missing_evidence", "retrieval", "ranking", "generation")
GENERATION_FAIL_THRESHOLD = 0.5

_METRIC_KEYS = [
    "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
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
) -> BenchmarkReport:
    """
    Run the benchmark for the given user.

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
    """
    from utils.documents import check_daily_quota, record_usage

    if pipelines is None:
        pipelines = list(PIPELINE_NAMES)
    unknown = [p for p in pipelines if p not in PIPELINE_NAMES]
    if unknown:
        raise ValueError(f"Unknown pipelines: {unknown}")

    queries = load_eval_queries(user_id, reviewed_only=reviewed_only)
    if not queries:
        return BenchmarkReport(
            pipelines=pipelines, k=k, per_pipeline={}, per_query=[],
            total_queries=0, total_duration_ms=0,
            config={"reviewed_only": reviewed_only, "reason": "no eval queries"},
        )

    fingerprint = dataset_fingerprint(queries)
    cache = _session_cache()
    cache_key = _cache_key(user_id, k, pipelines, fingerprint, f"{include_llm_judges}-{reviewed_only}")
    if not force_rerun and cache_key in cache:
        return cache[cache_key]

    if enforce_quota:
        check_daily_quota(user_id, "benchmark")

    from config import LLM_MODEL, JUDGE_MODEL, MODEL_PRICES
    from retriver.vector_store import get_embedding_model_name, get_user_chunk_ids

    llm = judge_llm = None
    if include_llm_judges:
        from llm.llm_handler import get_llm, get_judge_llm
        llm = get_llm()
        judge_llm = get_judge_llm()

    try:
        indexed_ids = get_user_chunk_ids(user_id)
    except Exception as e:
        logger.warning("Could not load indexed chunk ids for failure analysis: %s", e)
        indexed_ids = None

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
        "generator_model": LLM_MODEL if include_llm_judges else None,
        "judge_model": JUDGE_MODEL if include_llm_judges else None,
        "judge_independent": (JUDGE_MODEL != LLM_MODEL) if include_llm_judges else None,
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

    for q_info in queries:
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
        per_pipeline[p] = agg

    total_duration_ms = int((time.time() - global_t0) * 1000)
    config["finished_at"] = time.time()

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
    if cfg.get("include_llm_judges") and not cfg.get("prices_configured"):
        notes.append("Cost is reported as n/a because MODEL_PRICES_JSON is not configured; token counts are exact.")
    if size and size < 20:
        notes.append(f"Only {size} queries were evaluated; differences between pipelines may not be significant.")
    return notes
