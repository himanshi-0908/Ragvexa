"""
benchmark.py — RAG Pipeline Benchmark Runner

Orchestrates evaluation of all selected retrieval pipelines across all
synthetic evaluation queries for a user.

Typical call:
    report = run_benchmark(user_id=1, k=5, pipelines=PIPELINE_NAMES)

Returns a BenchmarkReport with per-pipeline aggregates and per-query details.
Results are cached in st.session_state to avoid re-computation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from retriver.retriver import retrieve_pipeline, PIPELINE_NAMES
from utils.eval_dataset import load_eval_queries
from utils.evaluator import evaluate_query


@dataclass
class BenchmarkReport:
    pipelines: List[str]
    k: int
    per_pipeline: Dict[str, Dict[str, Any]]   # aggregate metrics per pipeline
    per_query: List[Dict[str, Any]]            # full detail for query inspection
    total_queries: int
    total_duration_ms: int
    timestamp: float = field(default_factory=time.time)


def _cache_key(user_id: int, k: int, pipelines: List[str], query_hash: str) -> str:
    combo = f"{user_id}-{k}-{'-'.join(sorted(pipelines))}-{query_hash}"
    return "benchmark_" + hashlib.md5(combo.encode()).hexdigest()[:12]


def _avg(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def run_benchmark(
    user_id: int,
    k: int = 5,
    pipelines: Optional[List[str]] = None,
    include_llm_judges: bool = False,
    progress_callback=None,
    force_rerun: bool = False,
) -> BenchmarkReport:
    """
    Run the benchmark for the given user.

    Args:
        user_id: authenticated user's DB ID
        k: top-K to use for all retrieval and metrics
        pipelines: list of pipeline names to evaluate (default: all 4)
        include_llm_judges: if True, generate answers and run LLM-judge metrics
                           (much slower — 3× LLM calls per query per pipeline)
        progress_callback: optional callable(current_step, total_steps, message)
        force_rerun: skip session cache and re-run

    Returns:
        BenchmarkReport with per-pipeline averages and full per-query detail
    """
    import streamlit as st

    if pipelines is None:
        pipelines = PIPELINE_NAMES

    queries = load_eval_queries(user_id)
    if not queries:
        return BenchmarkReport(
            pipelines=pipelines,
            k=k,
            per_pipeline={},
            per_query=[],
            total_queries=0,
            total_duration_ms=0,
        )

    query_hash = hashlib.md5(str([q["query"] for q in queries]).encode()).hexdigest()[:8]
    cache_key = _cache_key(user_id, k, pipelines, query_hash)

    if not force_rerun and cache_key in st.session_state:
        return st.session_state[cache_key]

    llm = None
    if include_llm_judges:
        from llm.llm_handler import get_llm
        llm = get_llm()

    total_steps = len(queries) * len(pipelines)
    step = 0
    global_t0 = time.time()

    # Accumulate per-pipeline metric lists
    accum: Dict[str, Dict[str, list]] = {
        p: {
            "recall_at_k": [], "precision_at_k": [], "mrr": [], "ndcg_at_k": [],
            "retrieval_latency_ms": [], "reranking_latency_ms": [],
            "context_relevance": [], "answer_faithfulness": [], "answer_relevance": [],
            "generation_latency_ms": [],
        }
        for p in pipelines
    }

    per_query_rows: List[Dict] = []

    for q_info in queries:
        query = q_info["query"]
        relevant_ids = q_info["relevant_chunk_ids"]

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
            )
            metrics["query"] = query
            metrics["relevant_chunk_ids"] = relevant_ids
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

            # Accumulate
            amap = accum[pipeline_name]
            for key in amap:
                val = metrics.get(key)
                if val is not None:
                    amap[key].append(val)

    # Aggregate per pipeline
    per_pipeline: Dict[str, Dict] = {}
    for p in pipelines:
        amap = accum[p]
        per_pipeline[p] = {k_: _avg(amap[k_]) for k_ in amap}
        per_pipeline[p]["pipeline"] = p

    total_duration_ms = int((time.time() - global_t0) * 1000)

    report = BenchmarkReport(
        pipelines=pipelines,
        k=k,
        per_pipeline=per_pipeline,
        per_query=per_query_rows,
        total_queries=len(queries),
        total_duration_ms=total_duration_ms,
    )

    st.session_state[cache_key] = report
    return report


def get_pipeline_summary_table(report: BenchmarkReport) -> List[Dict]:
    """Return a list of dicts suitable for st.dataframe display."""
    rows = []
    for p in report.pipelines:
        m = report.per_pipeline.get(p, {})

        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"

        rows.append({
            "Pipeline": p,
            "Recall@K": fmt(m.get("recall_at_k")),
            "Precision@K": fmt(m.get("precision_at_k")),
            "MRR": fmt(m.get("mrr")),
            "nDCG@K": fmt(m.get("ndcg_at_k")),
            "Context Rel.": fmt(m.get("context_relevance")),
            "Faithfulness": fmt(m.get("answer_faithfulness")),
            "Ans. Rel.": fmt(m.get("answer_relevance")),
            "Retr. ms": int(m.get("retrieval_latency_ms") or 0),
            "Rerank ms": int(m.get("reranking_latency_ms") or 0),
            "Gen. ms": int(m.get("generation_latency_ms") or 0) if m.get("generation_latency_ms") else "—",
        })
    return rows


def get_failure_analysis(report: BenchmarkReport) -> List[Dict]:
    """Return queries where ALL pipelines achieved Recall@K = 0."""
    from collections import defaultdict
    recall_by_query: Dict[str, List[float]] = defaultdict(list)
    for row in report.per_query:
        recall_by_query[row["query"]].append(row.get("recall_at_k", 0.0))

    failures = []
    for query, recalls in recall_by_query.items():
        if all(r == 0.0 for r in recalls):
            failures.append({"query": query, "pipelines_tried": len(recalls)})
    return failures
