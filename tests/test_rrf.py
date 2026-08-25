"""
test_rrf.py — Unit tests for the Reciprocal Rank Fusion implementation.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from retriver.retriver import _rrf_fuse, cosine_similarity
import numpy as np


class TestRRF:
    def _make_docs(self, texts):
        return [Document(page_content=t) for t in texts]

    def test_single_list_preserves_order(self):
        docs = self._make_docs(["alpha", "beta", "gamma"])
        result = _rrf_fuse([docs], k_rrf=60)
        texts = [d.page_content for d, _ in result]
        assert texts == ["alpha", "beta", "gamma"]

    def test_fused_scores_are_positive(self):
        a = self._make_docs(["x", "y", "z"])
        b = self._make_docs(["z", "x", "w"])
        result = _rrf_fuse([a, b], k_rrf=60)
        assert all(score > 0 for _, score in result)

    def test_higher_ranked_in_both_lists_gets_higher_score(self):
        # "shared" appears at rank 1 in both lists
        shared = "shared_top"
        a = self._make_docs([shared, "only_a"])
        b = self._make_docs([shared, "only_b"])
        result = _rrf_fuse([a, b], k_rrf=60)
        top_text = result[0][0].page_content
        assert top_text == shared

    def test_deduplication(self):
        # Same document in both lists should appear once
        docs = self._make_docs(["dup", "other"])
        result = _rrf_fuse([docs, docs])
        texts = [d.page_content for d, _ in result]
        assert len(texts) == len(set(texts))

    def test_empty_lists(self):
        result = _rrf_fuse([[], []])
        assert result == []

    def test_known_rrf_values(self):
        # With k_rrf=60: score for rank-1 doc = 1/(60+1) = 0.016393
        docs = self._make_docs(["only"])
        result = _rrf_fuse([docs], k_rrf=60)
        expected = 1.0 / 61.0
        assert result[0][1] == pytest.approx(expected, abs=1e-6)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.5, -0.3])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_known_value(self):
        a = np.array([3.0, 4.0])
        b = np.array([4.0, 3.0])
        # dot = 12+12=24; |a|=5, |b|=5; sim=24/25=0.96
        assert cosine_similarity(a, b) == pytest.approx(0.96, abs=1e-4)
