"""
test_evaluation.py — Cross-model question generation, validation and
de-duplication, judge metrics with usage tracking, cost estimation,
failure classification and the benchmark runner (all mocked, no network).
"""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document

from utils.eval_dataset import (
    validate_question, dedup_questions, is_duplicate, parse_qa_block,
    generate_eval_dataset, load_eval_queries, import_ground_truth, dataset_fingerprint,
)
from utils.evaluator import UsageTracker, evaluate_query, _parse_score, answer_correctness
from utils.benchmark import classify_failure, run_benchmark, get_limitations, get_failure_breakdown
from llm.llm_handler import estimate_cost, invoke_tracked, LLMCall
from retriver.retriver import PipelineResult, RetrievedChunk


# ─────────────────────────────────────────────
# Question validation / dedup
# ─────────────────────────────────────────────
class TestQuestionValidation:
    def test_accepts_real_questions(self):
        assert validate_question("1. What is reciprocal rank fusion?") == "What is reciprocal rank fusion?"
        assert validate_question("Explain how BM25 scores documents") is not None

    @pytest.mark.parametrize("bad", ["", "Short?", "x" * 400, "What does this passage say?",
                                     "Here are the questions:", "12345 67890?"])
    def test_rejects_junk(self, bad):
        assert validate_question(bad) is None

    def test_dedup_exact_and_fuzzy(self):
        qs = ["What is RAG?", "what is rag", "What is RAG ?", "How does BM25 work?", "How does BM-25 work?"]
        assert dedup_questions(qs) == ["What is RAG?", "How does BM25 work?"]

    def test_not_duplicate_when_different(self):
        assert not is_duplicate("What is recall?", ["What is precision?"])

    def test_parse_qa_block(self):
        text = "Q: What is X?\nA: X is a thing.\nQ2: Why Y?\nA: Because.\n"
        assert parse_qa_block(text) == [("What is X?", "X is a thing."), ("Why Y?", "Because.")]

    def test_parse_plain_lines_fallback(self):
        assert parse_qa_block("What is X?\nWhy Y?") == [("What is X?", None), ("Why Y?", None)]


class TestCrossModelGeneration:
    def _chunks(self):
        return [Document(page_content="Reciprocal Rank Fusion combines ranked lists by summing 1/(k+rank). " * 3,
                         metadata={"chunk_id": "chunk-rrf"})]

    def test_uses_every_model_and_dedups_across_them(self, fresh_db):
        outputs = {
            "model-a": "Q: What does Reciprocal Rank Fusion combine?\nA: Ranked lists.\nQ: How is the RRF score computed?\nA: Sum of 1/(k+rank).",
            "model-b": "Q: What does reciprocal rank fusion combine?\nA: ranked lists\nQ: Which constant appears in RRF?\nA: k.",
        }
        factory_calls = []

        def factory(model):
            factory_calls.append(model)
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content=outputs[model])
            return llm

        saved = generate_eval_dataset(self._chunks(), user_id=1, source_name="s", doc_hash="h1",
                                      generator_models=["model-a", "model-b"], llm_factory=factory)
        assert factory_calls == ["model-a", "model-b"]
        assert saved == 3  # the near-duplicate from model-b was dropped
        qs = load_eval_queries(1)
        assert {q["query"] for q in qs} == {
            "What does Reciprocal Rank Fusion combine?", "How is the RRF score computed?", "Which constant appears in RRF?",
        }
        by_q = {q["query"]: q for q in qs}
        assert by_q["Which constant appears in RRF?"]["generator_models"] == ["model-b"]
        assert by_q["Which constant appears in RRF?"]["reference_answer"] == "k."
        assert all(q["origin"] == "synthetic" and q["reviewed"] is False for q in qs)

    def test_one_failing_model_does_not_block_others(self, fresh_db):
        def factory(model):
            llm = MagicMock()
            if model == "broken":
                llm.invoke.side_effect = RuntimeError("down")
            else:
                llm.invoke.return_value = MagicMock(content="Q: What is fused in RRF?\nA: lists")
            return llm
        saved = generate_eval_dataset(self._chunks(), 1, "s", "h2", generator_models=["broken", "ok"], llm_factory=factory)
        assert saved == 1

    def test_skips_when_dataset_exists(self, fresh_db):
        factory = lambda m: MagicMock(invoke=MagicMock(return_value=MagicMock(content="Q: What is fused in RRF?\nA: x")))
        assert generate_eval_dataset(self._chunks(), 1, "s", "h3", generator_models=["m"], llm_factory=factory) == 1
        assert generate_eval_dataset(self._chunks(), 1, "s", "h3", generator_models=["m"], llm_factory=factory) == 0


class TestGroundTruthImport:
    @patch("retriver.vector_store.get_vector_store")
    def test_import_matches_passages_and_marks_reviewed(self, mock_vs, fresh_db):
        mock_vs.return_value.get.return_value = {
            "documents": ["Precision at K is the fraction of retrieved items that are relevant.", "Other text."],
            "metadatas": [{"chunk_id": "prec1", "source": "a.pdf"}, {"chunk_id": "oth", "source": "a.pdf"}],
        }
        res = import_ground_truth(1, [
            {"query": "What is precision at K?", "relevant_passages": ["fraction of retrieved items that are relevant"],
             "reference_answer": "The fraction of retrieved items that are relevant."},
            {"query": "What is foo?", "relevant_chunk_ids": ["does-not-exist"]},
            {"query": "bad"},
        ])
        assert res["saved"] == 1 and len(res["skipped"]) == 2
        qs = load_eval_queries(1, reviewed_only=True)
        assert qs[0]["relevant_chunk_ids"] == ["prec1"] and qs[0]["origin"] == "human" and qs[0]["reviewed"]
        assert load_eval_queries(2, reviewed_only=True) == []

    def test_fingerprint_is_order_independent(self):
        a = [{"query": "q1", "relevant_chunk_ids": ["a", "b"]}, {"query": "q2", "relevant_chunk_ids": ["c"]}]
        b = [{"query": "q2", "relevant_chunk_ids": ["c"]}, {"query": "q1", "relevant_chunk_ids": ["b", "a"]}]
        assert dataset_fingerprint(a) == dataset_fingerprint(b)
        assert dataset_fingerprint(a) != dataset_fingerprint(a[:1])


# ─────────────────────────────────────────────
# Usage, cost, judge
# ─────────────────────────────────────────────
class TestUsageAndCost:
    def test_cost_only_for_priced_models(self):
        assert estimate_cost("priced-model", 1_000_000, 500_000) == 2.0
        assert estimate_cost("unknown-model", 100, 100) is None

    def test_invoke_tracked_reads_usage(self):
        llm = MagicMock(model="priced-model")
        llm.invoke.return_value = MagicMock(content=" Score: 0.8 ", usage_metadata={"input_tokens": 100, "output_tokens": 10})
        call = invoke_tracked(llm, "p")
        assert (call.text, call.input_tokens, call.output_tokens) == ("Score: 0.8", 100, 10)
        assert call.cost_usd == pytest.approx(0.00012)

    def test_tracker_never_guesses_cost(self):
        t = UsageTracker()
        t.add(LLMCall("a", "m", 10, 5, 1, 0.5), "gen")
        assert t.cost_usd == 0.5
        t.add(LLMCall("b", "m", 10, 5, 1, None), "judge")
        assert t.cost_usd is None and t.total_tokens == 30
        assert t.as_dict()["judge_input_tokens"] == 10

    def test_parse_score_clamps(self):
        assert _parse_score("Score: 3") == 1.0
        assert _parse_score("I'd say 0.75") == 0.75
        assert _parse_score("no number") is None

    def test_correctness_requires_reference(self):
        assert answer_correctness(MagicMock(), "q", None, "ans") is None
        assert answer_correctness(MagicMock(), "q", "ref", "") == 0.0


def _pr(ids, pipeline="Naive"):
    return PipelineResult(pipeline, [RetrievedChunk(c, "s", f"text {c}", 0.5, i + 1, pipeline) for i, c in enumerate(ids)], 3)


class TestEvaluateQuery:
    def test_retrieval_only(self):
        m = evaluate_query("q", _pr(["a", "x", "b"]), ["a", "b"], k=3, include_llm_judges=False)
        assert m["recall_at_k"] == 1.0 and m["precision_at_k"] == pytest.approx(0.6667) and m["answer"] is None

    def test_independent_judge_and_usage(self):
        gen = MagicMock(model="gen-model")
        gen.invoke.return_value = MagicMock(content="generated", usage_metadata={"input_tokens": 50, "output_tokens": 5})
        judge = MagicMock(model="judge-model")
        judge.invoke.return_value = MagicMock(content="Score: 0.9", usage_metadata={"input_tokens": 20, "output_tokens": 2})
        m = evaluate_query("q", _pr(["a"]), ["a"], k=1, llm=gen, judge_llm=judge, reference_answer="ref")
        assert m["answer"] == "generated"
        assert m["judge_independent"] is True and m["judge_model"] == "judge-model"
        assert judge.invoke.call_count == 5  # relevance, faithfulness, groundedness, ans. relevance, correctness
        assert m["answer_correctness"] == 0.9 and m["answer_groundedness"] == 0.9
        assert m["gen_input_tokens"] == 50 and m["judge_input_tokens"] == 100 and m["total_tokens"] == 165
        assert m["cost_usd"] is None  # models not priced → no made-up cost

    def test_judge_failure_gives_none_not_fallback(self):
        gen = MagicMock(model="g"); gen.invoke.return_value = MagicMock(content="ans", usage_metadata={})
        judge = MagicMock(model="j"); judge.invoke.side_effect = RuntimeError("down")
        m = evaluate_query("q", _pr(["a"]), ["a"], k=1, llm=gen, judge_llm=judge)
        assert m["answer_faithfulness"] is None and m["context_relevance"] is None


# ─────────────────────────────────────────────
# Failure classification
# ─────────────────────────────────────────────
class TestClassifyFailure:
    def test_categories(self):
        idx = {"a", "b", "c"}
        assert classify_failure(["z"], ["a"], 3, idx, []) == "missing_evidence"
        assert classify_failure(["a"], ["b", "c"], 2, idx, ["b", "c"]) == "retrieval"
        assert classify_failure(["a"], ["b", "c"], 2, idx, ["b", "c", "a"]) == "ranking"
        assert classify_failure(["a"], ["a"], 1, idx, None) == "ok"
        assert classify_failure(["a"], ["a"], 1, idx, None, {"answer_faithfulness": 0.2}) == "generation"
        assert classify_failure(["a"], ["a"], 1, idx, None, {"answer_faithfulness": 0.9, "answer_correctness": None}) == "ok"
        assert classify_failure([], ["a"], 1, idx) == "missing_evidence"


# ─────────────────────────────────────────────
# Benchmark runner (end-to-end, mocked retrieval)
# ─────────────────────────────────────────────
class TestBenchmarkRunner:
    def _seed(self, user_id):
        from utils.database import SessionLocal, SyntheticEvalItem
        db = SessionLocal()
        db.add_all([
            SyntheticEvalItem(user_id=user_id, query_text="What is alpha?", chunk_id="alpha", source_doc="d", doc_hash="h",
                              reviewed=True, origin="human", reference_answer="alpha is first"),
            SyntheticEvalItem(user_id=user_id, query_text="What is gone?", chunk_id="deleted", source_doc="d", doc_hash="h"),
        ])
        db.commit(); db.close()

    @patch("utils.benchmark.retrieve_pipeline")
    @patch("retriver.vector_store.get_user_chunk_ids", return_value={"alpha", "beta"})
    def test_runs_all_pipelines_same_settings_and_persists(self, _ids, mock_retrieve, two_users):
        a, b = two_users
        self._seed(a)
        mock_retrieve.side_effect = lambda q, uid, strategy, k: _pr(["beta", "alpha"] if "alpha" in q else ["beta"], strategy)
        from retriver.retriver import PIPELINE_NAMES
        report = run_benchmark(a, k=2, include_llm_judges=False, force_rerun=True,
                               candidate_pool_fn=lambda q: ["beta", "alpha"])
        assert report.total_queries == 2 and set(report.per_pipeline) == set(PIPELINE_NAMES)
        for c in mock_retrieve.call_args_list:
            assert c.args[1] == a and c.kwargs["k"] == 2
        assert report.config["dataset_size"] == 2 and report.config["dataset_reviewed_queries"] == 1
        assert report.config["embedding_model"] and report.config["dataset_fingerprint"]
        naive = report.per_pipeline["Naive"]
        assert naive["recall_at_k"] == 0.5 and naive["cost_usd"] is None
        assert naive["failures"]["ok"] == 1 and naive["failures"]["missing_evidence"] == 1
        assert report.run_id is not None
        from utils.benchmark import list_runs
        assert len(list_runs(a)) == 1 and list_runs(b) == []
        assert any("synthetic" in n for n in get_limitations(report))
        assert get_failure_breakdown(report)[0]["missing_evidence"] == 1

    @patch("utils.benchmark.retrieve_pipeline")
    @patch("retriver.vector_store.get_user_chunk_ids", return_value={"alpha"})
    def test_reviewed_only_filters_dataset(self, _ids, mock_retrieve, two_users):
        a, _ = two_users
        self._seed(a)
        mock_retrieve.side_effect = lambda q, uid, strategy, k: _pr(["alpha"], strategy)
        report = run_benchmark(a, k=1, pipelines=["Naive"], reviewed_only=True, force_rerun=True, persist=False)
        assert report.total_queries == 1 and report.per_query[0]["origin"] == "human"
        assert report.per_pipeline["Naive"]["recall_at_k"] == 1.0

    @patch("utils.benchmark.retrieve_pipeline")
    @patch("retriver.vector_store.get_user_chunk_ids", return_value={"alpha"})
    def test_daily_benchmark_quota(self, _ids, mock_retrieve, two_users):
        from utils.documents import QuotaExceeded, record_usage
        a, _ = two_users
        self._seed(a)
        mock_retrieve.side_effect = lambda q, uid, strategy, k: _pr(["alpha"], strategy)
        for _ in range(int(os.environ["MAX_BENCHMARKS_PER_DAY"])):
            record_usage(a, "benchmark")
        with pytest.raises(QuotaExceeded):
            run_benchmark(a, k=1, pipelines=["Naive"], force_rerun=True)

    def test_unknown_pipeline_rejected(self, two_users):
        a, _ = two_users
        with pytest.raises(ValueError):
            run_benchmark(a, pipelines=["Nope"], force_rerun=True)

    def test_empty_dataset_returns_empty_report(self, two_users):
        a, _ = two_users
        report = run_benchmark(a, force_rerun=True)
        assert report.total_queries == 0 and report.per_pipeline == {}
