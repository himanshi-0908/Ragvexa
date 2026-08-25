"""
test_metrics.py — Unit tests for retrieval and generation metrics.
"""

import json
import os
import sys
import pytest

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.evaluator import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
)

# ─────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "eval_fixture.json")


def load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# recall_at_k
# ─────────────────────────────────────────────
class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=5) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=5) == 0.0

    def test_partial_recall(self):
        r = recall_at_k(["a", "x", "y"], ["a", "b", "c"], k=3)
        assert r == pytest.approx(1 / 3, abs=1e-4)

    def test_k_cuts_off(self):
        # Only top-k matters; "b" is beyond k=1
        assert recall_at_k(["x", "b"], ["b"], k=1) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0

    def test_fixture(self):
        data = load_fixture()
        for item in data:
            relevant = item["relevant_chunk_ids"]
            # Simulate perfect retrieval
            r = recall_at_k(relevant, relevant, k=len(relevant))
            assert r == 1.0


# ─────────────────────────────────────────────
# precision_at_k
# ─────────────────────────────────────────────
class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a"], k=2) == pytest.approx(0.5, abs=1e-4)

    def test_k_zero(self):
        assert precision_at_k(["a", "b"], ["a"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=5) == 0.0


# ─────────────────────────────────────────────
# mrr
# ─────────────────────────────────────────────
class TestMRR:
    def test_first_result_relevant(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_result_relevant(self):
        assert mrr(["x", "a", "b"], ["a"]) == pytest.approx(0.5, abs=1e-4)

    def test_third_result_relevant(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3, abs=1e-4)

    def test_no_relevant(self):
        assert mrr(["x", "y", "z"], ["a", "b"]) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], ["a"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        # "b" is at rank 2, "c" at rank 3 — MRR should reflect rank of first hit
        assert mrr(["x", "b", "c"], ["b", "c"]) == pytest.approx(0.5, abs=1e-4)


# ─────────────────────────────────────────────
# ndcg_at_k
# ─────────────────────────────────────────────
class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Relevant docs at top → nDCG = 1.0
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0, abs=1e-4)

    def test_zero_relevant(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k([], ["a"], k=5) == 0.0

    def test_empty_relevant(self):
        # No ideal ranking → ideal DCG = 0 → nDCG = 0
        assert ndcg_at_k(["a", "b"], [], k=2) == 0.0

    def test_partial_ranking(self):
        # One relevant at rank 2, one not found
        val = ndcg_at_k(["x", "a", "y"], ["a", "b"], k=3)
        # Actual DCG: 1/log2(3); Ideal DCG: 1/log2(2) + 1/log2(3)
        import math
        actual_dcg = 1.0 / math.log2(3)
        ideal_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = actual_dcg / ideal_dcg
        assert val == pytest.approx(expected, abs=1e-4)


# ─────────────────────────────────────────────
# Edge cases across all metrics
# ─────────────────────────────────────────────
class TestEdgeCases:
    def test_single_result_relevant(self):
        assert recall_at_k(["a"], ["a"], k=1) == 1.0
        assert precision_at_k(["a"], ["a"], k=1) == 1.0
        assert mrr(["a"], ["a"]) == 1.0
        assert ndcg_at_k(["a"], ["a"], k=1) == 1.0

    def test_single_result_irrelevant(self):
        assert recall_at_k(["z"], ["a"], k=1) == 0.0
        assert precision_at_k(["z"], ["a"], k=1) == 0.0
        assert mrr(["z"], ["a"]) == 0.0
        assert ndcg_at_k(["z"], ["a"], k=1) == 0.0
