"""
test_pipelines.py — Unit tests for the retrieval pipeline abstraction.
Tests standardised PipelineResult schema and pipeline fallback logic
using mocked vector store and BM25.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from retriver.retriver import (
    RetrievedChunk,
    PipelineResult,
    PIPELINE_NAMES,
    retrieve_pipeline,
    retrieve_context,
    _rrf_fuse,
)


# ─────────────────────────────────────────────
# Fixture: mock vector store
# ─────────────────────────────────────────────
def _make_doc(text, chunk_id, source="test.pdf", user_id=1):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "user_id": user_id},
    )


MOCK_DOCS = [
    _make_doc("RAG retrieves relevant passages.", "chunk001"),
    _make_doc("BM25 is a keyword-based scorer.", "chunk002"),
    _make_doc("Embeddings encode semantic meaning.", "chunk003"),
]

# (doc, L2_distance) pairs returned by similarity_search_with_score
MOCK_SCORED = [(doc, 0.2) for doc in MOCK_DOCS]


def _patch_vs(mock_vs_cls):
    """Return a mock that simulates a populated vector store."""
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = MOCK_SCORED
    vs.similarity_search.return_value = MOCK_DOCS
    vs.get.return_value = {
        "documents": [d.page_content for d in MOCK_DOCS],
        "metadatas": [d.metadata for d in MOCK_DOCS],
    }
    vs.embeddings = MagicMock()
    vs.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    vs.embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]] * len(MOCK_DOCS)
    mock_vs_cls.return_value = vs
    return vs


# ─────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────
class TestPipelineResultSchema:
    @patch("retriver.retriver.get_vector_store")
    def test_naive_returns_pipeline_result(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Naive", k=3)
        assert isinstance(result, PipelineResult)
        assert result.pipeline == "Naive"
        assert isinstance(result.chunks, list)
        assert isinstance(result.retrieval_latency_ms, int)

    @patch("retriver.retriver.get_vector_store")
    def test_naive_chunk_schema(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Naive", k=3)
        for chunk in result.chunks:
            assert isinstance(chunk, RetrievedChunk)
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "source")
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "score")
            assert hasattr(chunk, "rank")
            assert hasattr(chunk, "pipeline")

    @patch("retriver.retriver.get_vector_store")
    def test_naive_ranks_are_sequential(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Naive", k=3)
        ranks = [c.rank for c in result.chunks]
        assert ranks == list(range(1, len(ranks) + 1))

    @patch("retriver.retriver.get_vector_store")
    def test_naive_scores_in_range(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Naive", k=3)
        for c in result.chunks:
            assert 0.0 <= c.score <= 1.0, f"score {c.score} out of range"

    @patch("retriver.retriver.get_vector_store")
    def test_pipeline_name_tagged_on_chunks(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        for pipeline in ["Naive", "Hybrid"]:
            result = retrieve_pipeline("test", user_id=1, strategy=pipeline, k=3)
            for chunk in result.chunks:
                assert chunk.pipeline == pipeline


# ─────────────────────────────────────────────
# Pipeline-specific tests
# ─────────────────────────────────────────────
class TestHybridPipeline:
    @patch("retriver.retriver.get_vector_store")
    def test_hybrid_returns_pipeline_result(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Hybrid", k=3)
        assert isinstance(result, PipelineResult)
        assert result.pipeline == "Hybrid"

    @patch("retriver.retriver.get_vector_store")
    def test_hybrid_chunks_have_source(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Hybrid", k=3)
        for c in result.chunks:
            assert c.source  # not empty


class TestQueryRewritingFallback:
    @patch("llm.llm_handler.get_llm")
    @patch("retriver.retriver.get_vector_store")
    def test_fallback_on_llm_failure(self, mock_vs_cls, mock_get_llm):
        """When LLM fails, Query Rewriting should fall back to original query."""
        _patch_vs(mock_vs_cls)
        # Make LLM raise an exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        with patch("llm.llm_handler.get_llm", mock_get_llm):
            result = retrieve_pipeline("what is RAG?", user_id=1, strategy="Query Rewriting", k=3)

        assert result.pipeline == "Query Rewriting"
        assert result.error is not None          # error is reported
        assert result.rewritten_query == "what is RAG?"   # fell back to original
        assert isinstance(result.chunks, list)


class TestRerankerFallback:
    @patch("retriver.retriver.get_vector_store")
    def test_reranker_returns_pipeline_result(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        result = retrieve_pipeline("test query", user_id=1, strategy="Hybrid + Reranker", k=3)
        assert isinstance(result, PipelineResult)
        assert result.pipeline == "Hybrid + Reranker"
        assert result.reranking_latency_ms >= 0


# ─────────────────────────────────────────────
# PIPELINE_NAMES constant
# ─────────────────────────────────────────────
class TestPipelineNames:
    def test_all_four_pipelines_defined(self):
        assert "Naive" in PIPELINE_NAMES
        assert "Hybrid" in PIPELINE_NAMES
        assert "Hybrid + Reranker" in PIPELINE_NAMES
        assert "Query Rewriting" in PIPELINE_NAMES
        assert len(PIPELINE_NAMES) == 4

    @patch("retriver.retriver.get_vector_store")
    def test_all_pipelines_return_correct_schema(self, mock_vs_cls):
        vs = _patch_vs(mock_vs_cls)
        # Mock LLM for query rewriting
        with patch("llm.llm_handler.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="optimized query")
            mock_get_llm.return_value = mock_llm

            for p in PIPELINE_NAMES:
                result = retrieve_pipeline("any query", user_id=1, strategy=p, k=3)
                assert isinstance(result, PipelineResult), f"Failed for {p}"
                assert result.pipeline == p, f"Wrong pipeline name for {p}"


# ─────────────────────────────────────────────
# Backward compat
# ─────────────────────────────────────────────
class TestBackwardCompat:
    @patch("retriver.retriver.get_vector_store")
    def test_retrieve_context_returns_string(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        ctx = retrieve_context("what is RAG?", user_id=1, k=3)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    @patch("retriver.retriver.get_vector_store")
    def test_unknown_strategy_raises(self, mock_vs_cls):
        _patch_vs(mock_vs_cls)
        with pytest.raises(ValueError):
            retrieve_pipeline("test", user_id=1, strategy="NonExistent", k=3)
