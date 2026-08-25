"""
retriver.py — Multi-pipeline retrieval with standardised output schema.

Every pipeline returns a PipelineResult containing a list of RetrievedChunk
objects so that the evaluator and UI always receive the same structure.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from .vector_store import get_vector_store

# ─────────────────────────────────────────────
# Standardised schema
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str        # md5 hash stored in metadata at ingest time
    source: str          # source document filename
    text: str            # chunk text
    score: float         # higher = more relevant (0-1 range where possible)
    rank: int            # 1-based rank within this pipeline's result list
    pipeline: str        # which pipeline produced this result

@dataclass
class PipelineResult:
    pipeline: str
    chunks: List[RetrievedChunk]
    retrieval_latency_ms: int
    reranking_latency_ms: int = 0
    rewritten_query: Optional[str] = None
    error: Optional[str] = None     # non-None when pipeline fell back

# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _doc_to_chunk(doc: Document, rank: int, score: float, pipeline: str) -> RetrievedChunk:
    meta = doc.metadata or {}
    return RetrievedChunk(
        chunk_id=meta.get("chunk_id", ""),
        source=meta.get("source", "unknown"),
        text=doc.page_content,
        score=round(float(score), 4),
        rank=rank,
        pipeline=pipeline,
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)

def _rrf_fuse(ranked_lists: List[List[Document]], k_rrf: int = 60) -> List[tuple[Document, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content[:100]          # dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            doc_map[key] = doc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in fused]

def _build_bm25(user_id: int, k: int) -> Optional[BM25Retriever]:
    vector_store = get_vector_store()
    res = vector_store.get(where={"user_id": user_id})
    docs = []
    if res and "documents" in res:
        for text, meta in zip(res["documents"], res["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta))
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

def _ndcg_normalize(scored_docs: List[tuple[Document, float]]) -> List[tuple[Document, float]]:
    """Normalize scores to [0, 1] range using min-max."""
    if not scored_docs:
        return []
    scores = [s for _, s in scored_docs]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [(d, 1.0) for d, _ in scored_docs]
    return [(d, (s - mn) / (mx - mn)) for d, s in scored_docs]

# ─────────────────────────────────────────────
# Pipeline 1: Naive Dense Retrieval
# ─────────────────────────────────────────────

def _pipeline_naive(query: str, user_id: int, k: int) -> PipelineResult:
    t0 = time.time()
    vector_store = get_vector_store()
    try:
        raw = vector_store.similarity_search_with_score(
            query, k=k, filter={"user_id": user_id}
        )
        # Chroma returns (doc, distance) — convert L2 distance to similarity
        # score = 1 / (1 + distance)
        chunks = [
            _doc_to_chunk(doc, rank + 1, 1.0 / (1.0 + dist), "Naive")
            for rank, (doc, dist) in enumerate(raw)
        ]
    except Exception as e:
        chunks = []
    latency = int((time.time() - t0) * 1000)
    return PipelineResult("Naive", chunks, latency)

# ─────────────────────────────────────────────
# Pipeline 2: Hybrid Dense + BM25 with RRF
# ─────────────────────────────────────────────

def _pipeline_hybrid(query: str, user_id: int, k: int) -> PipelineResult:
    t0 = time.time()
    vector_store = get_vector_store()
    try:
        dense_docs = vector_store.similarity_search(
            query, k=k * 2, filter={"user_id": user_id}
        )
        bm25 = _build_bm25(user_id, k=k * 2)
        kw_docs = bm25.invoke(query) if bm25 else []
        fused = _rrf_fuse([dense_docs, kw_docs])[:k]
        fused = _ndcg_normalize(fused)
        chunks = [
            _doc_to_chunk(doc, rank + 1, score, "Hybrid")
            for rank, (doc, score) in enumerate(fused)
        ]
    except Exception as e:
        chunks = []
    latency = int((time.time() - t0) * 1000)
    return PipelineResult("Hybrid", chunks, latency)

# ─────────────────────────────────────────────
# Pipeline 3: Hybrid + Cosine Reranker
# ─────────────────────────────────────────────

def _pipeline_reranker(query: str, user_id: int, k: int) -> PipelineResult:
    retrieval_t0 = time.time()
    vector_store = get_vector_store()
    error_msg = None
    try:
        dense_docs = vector_store.similarity_search(
            query, k=k * 3, filter={"user_id": user_id}
        )
        bm25 = _build_bm25(user_id, k=k * 3)
        kw_docs = bm25.invoke(query) if bm25 else []
        candidates = _rrf_fuse([dense_docs, kw_docs])
        candidates = [doc for doc, _ in candidates]
    except Exception as e:
        candidates = []
        error_msg = str(e)
    retrieval_latency = int((time.time() - retrieval_t0) * 1000)

    rerank_t0 = time.time()
    try:
        emb = vector_store.embeddings
        query_emb = np.array(emb.embed_query(query))
        doc_embs = np.array(emb.embed_documents([d.page_content for d in candidates]))
        scored = sorted(
            zip(candidates, [cosine_similarity(query_emb, de) for de in doc_embs]),
            key=lambda x: x[1], reverse=True
        )[:k]
        chunks = [
            _doc_to_chunk(doc, rank + 1, score, "Hybrid + Reranker")
            for rank, (doc, score) in enumerate(scored)
        ]
    except Exception as e:
        # Fallback: return hybrid results unchanged
        error_msg = f"Reranker failed ({e}), using hybrid fallback"
        hybrid = _pipeline_hybrid(query, user_id, k)
        for c in hybrid.chunks:
            c.pipeline = "Hybrid + Reranker"
        chunks = hybrid.chunks
    rerank_latency = int((time.time() - rerank_t0) * 1000)
    return PipelineResult("Hybrid + Reranker", chunks, retrieval_latency, rerank_latency, error=error_msg)

# ─────────────────────────────────────────────
# Pipeline 4: Query Rewriting → Naive
# ─────────────────────────────────────────────

def _pipeline_query_rewriting(query: str, user_id: int, k: int) -> PipelineResult:
    from llm.llm_handler import get_llm
    rewritten = None
    error_msg = None
    try:
        llm = get_llm()
        prompt = (
            "You are a search query optimizer. "
            "Rewrite the user question to be more descriptive and optimal for "
            "semantic document retrieval. Return ONLY the rewritten query text, "
            "nothing else.\n\n"
            f"Original query: {query}\n"
            "Rewritten query:"
        )
        rewritten = llm.invoke(prompt).content.strip().strip('"').strip("'")
    except Exception as e:
        rewritten = query
        error_msg = f"Query rewriting failed ({e}), using original query"

    result = _pipeline_naive(rewritten or query, user_id, k)
    result.pipeline = "Query Rewriting"
    result.rewritten_query = rewritten
    result.error = error_msg
    for c in result.chunks:
        c.pipeline = "Query Rewriting"
    return result

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

PIPELINE_NAMES = ["Naive", "Hybrid", "Hybrid + Reranker", "Query Rewriting"]

def retrieve_pipeline(
    query: str,
    user_id: int,
    strategy: str = "Naive",
    k: int = 5,
) -> PipelineResult:
    if strategy == "Naive":
        return _pipeline_naive(query, user_id, k)
    elif strategy == "Hybrid":
        return _pipeline_hybrid(query, user_id, k)
    elif strategy == "Hybrid + Reranker":
        return _pipeline_reranker(query, user_id, k)
    elif strategy == "Query Rewriting":
        return _pipeline_query_rewriting(query, user_id, k)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def retrieve_context(query: str, user_id: int, k: int = 4) -> str:
    """Backward-compatible helper used by existing chat code."""
    result = retrieve_pipeline(query, user_id, "Naive", k)
    return "\n\n".join(c.text for c in result.chunks)