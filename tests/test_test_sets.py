"""
test_test_sets.py — User test sets and evidence-based evaluation (spec §10).

Covers: chunk origin at ingest, upload validation (schema, evidence rules,
unknown / foreign documents, oversize file, duplicates, row cap), evidence
matching across chunk sizes, metrics against hand-computed values, abstention
handling, identical snapshot for every strategy, cascade deletion, evidence
view data and cross-user isolation. Everything runs offline: the vector
store, retrieval and LLM calls are mocked.
"""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document

from retriver.retriver import PipelineResult, RetrievedChunk
from retriver.vector_store import annotate_chunk_origins, normalise_document_text
from utils.chunking import chunk_documents
from utils.evaluator import (
    abstention_metrics,
    bootstrap_ci,
    format_ci,
    is_abstention,
    match_chunk_to_evidence,
    match_retrieved_chunks,
    token_overlap,
)
from utils.test_sets import (
    TestSetError,
    delete_test_set,
    list_test_sets,
    load_test_set,
    parse_pages,
    parse_test_set_file,
    rows_to_csv,
    safe_csv_cell,
    save_test_set,
    validate_test_set,
)
from utils.test_sets import test_set_snapshot as snapshot_of  # alias: not a test function

PASSAGE = "Refunds are accepted within 30 days of purchase."
PAGE4 = "Our policy. " + PASSAGE + " Exchanges are accepted within 60 days of purchase."
DOC_CHUNKS = {
    "policy.pdf": [
        {"text": "Welcome to the store handbook.", "page_start": 1, "page_end": 1, "char_start": 0, "char_end": 30},
        {"text": PAGE4, "page_start": 4, "page_end": 4, "char_start": 300, "char_end": 300 + len(PAGE4)},
    ],
    "notes.txt": [
        {"text": "hello world these are plain notes", "page_start": None, "page_end": None, "char_start": 0, "char_end": 33},
    ],
}


def _loader(user_id, source):
    return DOC_CHUNKS.get(source, [])


def _register(user_id, source="policy.pdf", doc_hash="h-policy"):
    from utils.documents import register_document
    return register_document(user_id, source, doc_hash, "file", 2)


def _rows(*extra):
    base = [
        {"question": "What is the refund window?", "expected_answer": "30 days", "answerable": True, "tags": ["policy"],
         "gold_evidence": [{"document": "policy.pdf", "pages": [4], "passage": PASSAGE}]},
        {"question": "Who is the CEO?", "answerable": False, "tags": ["abstain"]},
    ]
    return base + list(extra)


# ─────────────────────────────────────────────
# Ingestion: chunk origin
# ─────────────────────────────────────────────
class TestChunkOrigin:
    def test_offsets_and_pages_recorded(self):
        docs = [Document(page_content=f"Page {i} text. " * 40, metadata={"page": i, "source": "x.pdf"}) for i in range(3)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=40)
        located = annotate_chunk_origins(chunks, docs, document_id=7)
        assert located == len(chunks) > 3
        full = " ".join(normalise_document_text(d.page_content) for d in docs)
        for c in chunks:
            m = c.metadata
            assert m["document_id"] == 7 and m["page_start"] == m["page_end"] in (1, 2, 3)
            assert full[m["char_start"]:m["char_end"]] == normalise_document_text(c.page_content)
        assert chunks[0].metadata["page_start"] == 1
        assert chunks[-1].metadata["page_start"] == 3

    def test_txt_has_offsets_but_no_pages(self):
        docs = [Document(page_content="hello  world\nmore text", metadata={"source": "t.txt"})]
        chunks = chunk_documents(docs)
        annotate_chunk_origins(chunks, docs, document_id=3)
        m = chunks[0].metadata
        assert (m["char_start"], m["char_end"]) == (0, len("hello world more text"))
        assert "page_start" not in m and "page_end" not in m  # null pages are omitted for Chroma

    @patch("retriver.vector_store.get_vector_store")
    def test_add_to_vector_store_annotates(self, mock_vs):
        from retriver.vector_store import add_to_vector_store
        docs = [Document(page_content="alpha beta gamma", metadata={"page": 0})]
        chunks = chunk_documents(docs)
        add_to_vector_store(chunks, user_id=1, source="a.pdf", document_id=11, documents=docs)
        m = chunks[0].metadata
        assert m["document_id"] == 11 and m["page_start"] == 1 and m["char_start"] == 0 and m["user_id"] == 1
        assert None not in m.values()

    def test_retrieved_chunk_reads_origin(self):
        from retriver.retriver import _doc_to_chunk
        d = Document(page_content="t", metadata={"chunk_id": "c", "source": "s", "document_id": 5, "page_start": 2, "page_end": 2, "char_start": 10, "char_end": 20})
        c = _doc_to_chunk(d, 1, 0.5, "Naive")
        assert (c.document_id, c.page_start, c.char_start, c.char_end) == (5, 2, 10, 20)
        assert _doc_to_chunk(Document(page_content="t", metadata={}), 1, 0.5, "Naive").page_start is None


# ─────────────────────────────────────────────
# Upload parsing and validation
# ─────────────────────────────────────────────
class TestParsing:
    def test_json_and_csv_equivalent(self):
        js = json.dumps(_rows()).encode()
        csv_text = (
            "question,expected_answer,answerable,tags,evidence_document,evidence_pages,evidence_passage\n"
            f'What is the refund window?,30 days,true,policy,policy.pdf,4,"{PASSAGE}"\n'
            "Who is the CEO?,,false,abstain,,,\n"
        ).encode()
        a = parse_test_set_file("t.json", js)
        b = parse_test_set_file("t.csv", csv_text)
        assert [r["question"] for r in a] == [r["question"] for r in b]
        assert b[0]["evidence_passage"] == PASSAGE

    def test_csv_gold_evidence_json_column(self):
        csv_text = ('question,answerable,gold_evidence\n'
                    'What is the refund window?,true,"[{""document"": ""policy.pdf"", ""pages"": ""4"", ""passage"": ""' + PASSAGE + '""}]"\n').encode()
        rows = parse_test_set_file("t.csv", csv_text)
        assert json.loads(rows[0]["gold_evidence"])[0]["document"] == "policy.pdf"

    @pytest.mark.parametrize("name,data,msg", [
        ("t.json", b"not json", "Invalid JSON"),
        ("t.json", b'{"foo": 1}', "list"),
        ("t.csv", b"answerable,tags\ntrue,x\n", "question"),
        ("t.csv", b"question,answerable,bogus\nq,true,1\n", "Unknown CSV column"),
        ("t.xlsx", b"x", "Unsupported"),
        ("t.json", b"", "empty"),
        ("t.json", b"[]", "No test cases"),
    ])
    def test_bad_schema_rejected_with_clear_error(self, name, data, msg):
        with pytest.raises(TestSetError, match=msg):
            parse_test_set_file(name, data)

    def test_oversize_file_rejected(self):
        from config import MAX_TEST_SET_BYTES
        with pytest.raises(TestSetError, match="exceeds"):
            parse_test_set_file("t.json", b"[" + b" " * MAX_TEST_SET_BYTES + b"]")

    def test_row_cap_enforced(self):
        from config import MAX_TEST_SET_ROWS
        rows = [{"question": f"q{i}", "answerable": False} for i in range(MAX_TEST_SET_ROWS + 1)]
        with pytest.raises(TestSetError, match="Too many rows"):
            parse_test_set_file("t.json", json.dumps(rows).encode())

    def test_parse_pages_forms(self):
        assert parse_pages(4) == (4, 4)
        assert parse_pages("4-6") == (4, 6)
        assert parse_pages([2, 9]) == (2, 9)
        assert parse_pages("[3]") == (3, 3)
        assert parse_pages("") == (None, None)
        with pytest.raises(ValueError):
            parse_pages("four")
        with pytest.raises(ValueError):
            parse_pages(0)


class TestValidation:
    def test_valid_rows_pass_and_resolve_document(self, two_users):
        a, _ = two_users
        doc_id = _register(a)
        prev = validate_test_set(a, _rows(), chunk_loader=_loader)
        assert prev.error_rows == 0 and prev.valid_rows == 2
        ev = prev.rows[0].evidence[0]
        assert (ev.document_id, ev.page_start, ev.page_end, ev.passage) == (doc_id, 4, 4, PASSAGE)
        assert prev.rows[1].evidence == [] and prev.rows[1].answerable is False

    def test_missing_fields_and_evidence_rules(self, two_users):
        a, _ = two_users
        _register(a)
        rows = [
            {"question": "", "answerable": True, "gold_evidence": [{"document": "policy.pdf", "pages": 4}]},
            {"question": "Is answerable set?", "gold_evidence": [{"document": "policy.pdf", "pages": 4}]},
            {"question": "Answerable without evidence?", "answerable": True},
            {"question": "Unanswerable with evidence?", "answerable": False,
             "gold_evidence": [{"document": "policy.pdf", "pages": 4}]},
            {"question": "Evidence with nothing to locate?", "answerable": True, "gold_evidence": [{"document": "policy.pdf"}]},
            {"question": "Bad pages?", "answerable": True, "gold_evidence": [{"document": "policy.pdf", "pages": "four"}]},
            "not an object",
        ]
        prev = validate_test_set(a, rows, chunk_loader=_loader)
        errs = [" | ".join(r.errors) for r in prev.rows]
        assert "question is required" in errs[0]
        assert "answerable is required" in errs[1]
        assert "need gold_evidence" in errs[2]
        assert "must not have gold_evidence" in errs[3]
        assert "pages and/or a passage" in errs[4]
        assert "page number" in errs[5]
        assert "not an object" in errs[6]
        assert prev.valid_rows == 0 and prev.error_rows == 7

    def test_unknown_and_foreign_documents_rejected(self, two_users):
        a, b = two_users
        _register(a, "policy.pdf", "ha")
        _register(b, "secret.pdf", "hb")
        rows = [
            {"question": "Unknown document?", "answerable": True, "gold_evidence": [{"document": "nope.pdf", "pages": 1}]},
            {"question": "Bob's document?", "answerable": True, "gold_evidence": [{"document": "secret.pdf", "pages": 1}]},
        ]
        prev = validate_test_set(a, rows, chunk_loader=_loader)
        assert all("not one of your documents" in " | ".join(r.errors) for r in prev.rows)
        # Bob sees his own document but not Alice's
        prev_b = validate_test_set(b, rows[1:] + [{"question": "Alice's?", "answerable": True,
                                                  "gold_evidence": [{"document": "policy.pdf", "pages": 1}]}],
                                   chunk_loader=lambda u, s: [{"text": "x y z", "page_start": 1, "page_end": 1}])
        assert prev_b.rows[0].ok and not prev_b.rows[1].ok

    def test_passage_must_appear_in_document(self, two_users):
        a, _ = two_users
        _register(a)
        rows = [
            {"question": "Wrong passage?", "answerable": True,
             "gold_evidence": [{"document": "policy.pdf", "pages": 4, "passage": "The sky is green and the grass is purple today"}]},
            {"question": "Right passage, wrong page?", "answerable": True,
             "gold_evidence": [{"document": "policy.pdf", "pages": 1, "passage": PASSAGE}]},
            {"question": "Near-exact passage?", "answerable": True,
             "gold_evidence": [{"document": "policy.pdf", "pages": 4, "passage": "refunds are accepted within 30 days of purchase"}]},
        ]
        prev = validate_test_set(a, rows, chunk_loader=_loader)
        assert "passage not found" in prev.rows[0].errors[0]
        assert "passage not found" in prev.rows[1].errors[0]
        assert prev.rows[2].ok

    def test_duplicate_questions_flagged(self, two_users):
        a, _ = two_users
        _register(a)
        rows = _rows({"question": "what is the refund window ?", "answerable": False})
        prev = validate_test_set(a, rows, chunk_loader=_loader)
        assert prev.rows[2].errors == ["duplicate of row 1"]

    def test_save_refuses_errors_unless_skipping(self, two_users):
        a, _ = two_users
        _register(a)
        prev = validate_test_set(a, _rows({"question": "No evidence?", "answerable": True}), chunk_loader=_loader)
        with pytest.raises(TestSetError, match="errors"):
            save_test_set(a, "s", prev)
        sid = save_test_set(a, "s", prev, skip_invalid=True)
        assert list_test_sets(a)[0]["cases"] == 2
        loaded = load_test_set(a, sid)
        assert [c["question"] for c in loaded["cases"]] == ["What is the refund window?", "Who is the CEO?"]
        assert loaded["cases"][0]["gold_evidence"][0]["document"] == "policy.pdf"
        assert loaded["cases"][0]["tags"] == ["policy"]

    def test_test_sets_are_private(self, two_users):
        a, b = two_users
        _register(a)
        sid = save_test_set(a, "mine", validate_test_set(a, _rows(), chunk_loader=_loader))
        assert load_test_set(b, sid) is None
        assert snapshot_of(b, sid) is None
        assert list_test_sets(b) == []
        assert delete_test_set(b, sid) == 0 and load_test_set(a, sid) is not None
        assert delete_test_set(a, sid) == 2 and load_test_set(a, sid) is None

    def test_csv_export_is_formula_safe(self):
        assert safe_csv_cell("=SUM(A1)") == "'=SUM(A1)"
        assert safe_csv_cell("+1") == "'+1" and safe_csv_cell("-x") == "'-x" and safe_csv_cell("@cmd") == "'@cmd"
        assert safe_csv_cell("plain") == "plain" and safe_csv_cell(None) == ""
        out = rows_to_csv([{"q": "=HYPERLINK()", "n": 1}])
        assert out.splitlines()[1].startswith("'=HYPERLINK()")


# ─────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────
GOLD = {"document_id": 1, "document": "policy.pdf", "page_start": 4, "page_end": 4, "passage": PASSAGE}


def _chunk(text, page=4, rank=1, doc_id=1, source="policy.pdf", **kw):
    return {"document_id": doc_id, "source": source, "page_start": page, "page_end": page, "text": text, "rank": rank, **kw}


class TestMatching:
    def test_same_text_split_differently_is_matched(self):
        big = _chunk("Intro paragraph. " + PAGE4 + " Closing remarks about the store.")
        assert match_chunk_to_evidence(big, GOLD)["hit"] is True
        # The passage split in two halves by a smaller chunker: both halves count at the default 0.5 threshold
        left, right = _chunk("policy. Refunds are accepted within 30"), _chunk("30 days of purchase. Exchanges are")
        assert match_chunk_to_evidence(left, GOLD)["hit"] and match_chunk_to_evidence(right, GOLD)["hit"]
        assert match_chunk_to_evidence(left, GOLD, overlap_threshold=0.9)["hit"] is False
        assert token_overlap(PASSAGE, left["text"]) == pytest.approx(5 / 8)

    def test_unrelated_chunk_on_same_page_is_not_a_hit(self):
        m = match_chunk_to_evidence(_chunk("The quarterly sales meeting is on Tuesday in room 4."), GOLD)
        assert m["hit"] is False and "below threshold" in m["reason"]

    def test_other_document_or_page_is_not_a_hit(self):
        assert match_chunk_to_evidence(_chunk(PAGE4, doc_id=2, source="other.pdf"), GOLD)["reason"] == "different document"
        assert match_chunk_to_evidence(_chunk(PAGE4, page=5), GOLD)["reason"] == "pages do not overlap"
        # a chunk spanning pages 3-5 overlaps page 4
        assert match_chunk_to_evidence({**_chunk(PAGE4), "page_start": 3, "page_end": 5}, GOLD)["hit"]

    def test_coarse_page_only_match_is_flagged(self):
        gold_no_passage = {**GOLD, "passage": None}
        m = match_chunk_to_evidence(_chunk("Anything on page four."), gold_no_passage)
        assert m["hit"] and m["coarse"] and "coarse" in m["reason"]
        precise = match_chunk_to_evidence(_chunk(PAGE4), GOLD)
        assert precise["hit"] and not precise["coarse"]

    def test_falls_back_to_source_name_and_char_ranges(self):
        # Chunks indexed before origin tracking have no document_id: match by source name.
        legacy = {"source": "policy.pdf", "text": PAGE4, "rank": 1}
        assert match_chunk_to_evidence(legacy, GOLD)["hit"]
        # Char-range overlap can carry a hit when the text overlap is low.
        gold_chars = {**GOLD, "passage": "x y z w", "char_start": 100, "char_end": 200}
        assert match_chunk_to_evidence(_chunk("unrelated", char_start=120, char_end=300), gold_chars)["hit"]
        assert not match_chunk_to_evidence(_chunk("unrelated", char_start=180, char_end=300), gold_chars)["hit"]

    def test_txt_source_matches_without_pages(self):
        gold = {"document_id": 2, "document": "notes.txt", "page_start": None, "page_end": None, "passage": "plain notes"}
        chunk = {"document_id": 2, "source": "notes.txt", "page_start": None, "page_end": None, "text": "these are plain notes", "rank": 1}
        assert match_chunk_to_evidence(chunk, gold)["hit"]


# ─────────────────────────────────────────────
# Metrics (hand-computed)
# ─────────────────────────────────────────────
class TestMetrics:
    def test_retrieval_metrics_from_evidence(self):
        gold2 = {"document_id": 1, "document": "policy.pdf", "page_start": 1, "page_end": 1, "passage": "Welcome to the store handbook."}
        chunks = [
            _chunk("The quarterly sales meeting is on Tuesday.", rank=1),       # miss
            _chunk(PAGE4, rank=2),                                               # hit gold 1
            _chunk("Welcome to the store handbook.", page=1, rank=3),           # hit gold 2
            _chunk("Refunds are accepted within 30 days of purchase.", rank=4),  # duplicate of gold 1 (outside k)
        ]
        r = match_retrieved_chunks(chunks, [GOLD, gold2], k=3)
        assert r["hit_rate_at_k"] == 1.0 and r["recall_at_k"] == 1.0 and r["mrr"] == 0.5
        assert r["precision_at_k"] == pytest.approx(2 / 3, abs=1e-4)
        import math
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert r["ndcg_at_k"] == pytest.approx(dcg / idcg, abs=1e-4)
        assert [c["hit"] for c in r["per_chunk"]] == [False, True, True, True]
        assert r["per_chunk"][3]["in_top_k"] is False and r["gold_first_rank"] == {0: 2, 1: 3}

        r2 = match_retrieved_chunks(chunks, [GOLD, gold2], k=2)
        assert r2["recall_at_k"] == 0.5 and r2["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)), abs=1e-4)
        assert match_retrieved_chunks(chunks[:1], [GOLD], k=1)["recall_at_k"] == 0.0

    def test_duplicate_hits_do_not_inflate_ndcg(self):
        chunks = [_chunk(PAGE4, rank=1), _chunk(PASSAGE, rank=2)]
        assert match_retrieved_chunks(chunks, [GOLD], k=2)["ndcg_at_k"] == 1.0

    def test_abstention_rates(self):
        assert abstention_metrics([True, True, False, True]) == {"n": 4, "correct_abstention_rate": 0.75, "false_answer_rate": 0.25}
        assert abstention_metrics([None, None])["correct_abstention_rate"] is None
        assert is_abstention("The provided context does not contain information about the CEO.")
        assert is_abstention("I don't know based on the documents.")
        assert not is_abstention("The CEO is Jane Doe.")

    def test_bootstrap_ci_and_not_enough_data(self):
        vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        ci = bootstrap_ci(vals, min_size=10)
        assert ci["enough_data"] and ci["mean"] == 0.75 and ci["low"] <= 0.75 <= ci["high"] and 0 <= ci["low"] < ci["high"] <= 1
        assert bootstrap_ci(vals, min_size=10) == ci  # deterministic
        small = bootstrap_ci([1, 0, 1], min_size=10)
        assert small["enough_data"] is False and small["low"] is None and small["mean"] == pytest.approx(0.6667)
        assert "not enough data" in format_ci(small) and format_ci(ci).startswith("0.750 [")
        assert format_ci(None) == "—"


# ─────────────────────────────────────────────
# Benchmark on a test set (mocked retrieval + LLM)
# ─────────────────────────────────────────────
def _retriever(doc_id):
    def mk(cid, text, page, rank, pipeline):
        return RetrievedChunk(cid, "policy.pdf", text, 0.5, rank, pipeline, document_id=doc_id, page_start=page, page_end=page)

    def retr(q, uid, strategy, k):
        if "refund" in q:
            chunks = [mk("x", "Welcome to the store handbook.", 1, 1, strategy), mk("y", PAGE4, 4, 2, strategy)]
        elif "exchange" in q:
            # only the wide Naive pool (k >= 20) contains the evidence → "ranking" for every strategy
            chunks = [mk("z", "Welcome to the store handbook.", 1, 1, strategy)]
            if k >= 20:
                chunks.append(mk("y", PAGE4, 4, 2, strategy))
        else:
            chunks = [mk("x", "Welcome to the store handbook.", 1, 1, strategy)]
        return PipelineResult(strategy, chunks[:k], 3)
    return retr


def _llms(answer_for):
    gen = MagicMock(model="gen-model", switched_from=None)
    gen.invoke.side_effect = lambda p: MagicMock(content=answer_for(p), usage_metadata={"input_tokens": 5, "output_tokens": 2})
    judge = MagicMock(model="judge-model", switched_from=None)
    judge.invoke.return_value = MagicMock(content="Score: 0.9", usage_metadata={})
    return gen, judge


class TestBenchmarkOnTestSet:
    def _seed(self, user_id):
        doc_id = _register(user_id)
        rows = _rows({"question": "What is the exchange window?", "answerable": True, "tags": ["policy", "hard"],
                      "gold_evidence": [{"document": "policy.pdf", "pages": 4, "passage": "Exchanges are accepted within 60 days of purchase."}]})
        sid = save_test_set(user_id, "set-1", validate_test_set(user_id, rows, chunk_loader=_loader))
        return doc_id, sid

    def test_all_strategies_see_identical_cases_and_snapshot_is_immutable(self, two_users):
        from utils.benchmark import run_benchmark, load_run
        from retriver.retriver import PIPELINE_NAMES
        a, _ = two_users
        doc_id, sid = self._seed(a)
        seen = {}
        base = _retriever(doc_id)

        def spy(q, uid, strategy, k):
            seen.setdefault(strategy, []).append((q, uid, k))
            return base(q, uid, strategy, k)

        with patch("utils.benchmark.retrieve_pipeline", side_effect=spy), \
             patch("retriver.vector_store.get_user_indexed_document_ids", return_value={doc_id}):
            report = run_benchmark(a, k=2, include_llm_judges=False, test_set_id=sid, force_rerun=True)
        per_strategy = {p: [(q, uid, k) for (q, uid, k) in calls if k == 2] for p, calls in seen.items()}
        assert set(per_strategy) == set(PIPELINE_NAMES)
        assert len({tuple(v) for v in per_strategy.values()}) == 1  # identical questions, user, k
        assert all(uid == a for calls in seen.values() for (_, uid, _) in calls)

        snap = report.config["test_set_snapshot"]
        assert report.config["dataset_kind"] == "test_set" and snap["fingerprint"] == report.config["test_set_fingerprint"]
        assert [c["question"] for c in snap["cases"]] == [r["query"] for r in report.per_query if r["pipeline"] == "Naive"]

        # Changing the test set afterwards does not change the stored run
        delete_test_set(a, sid)
        stored = load_run(a, report.run_id)
        assert stored.config["test_set_snapshot"] == snap and len(stored.per_query) == 3 * len(PIPELINE_NAMES)
        assert stored.config["dataset_human_queries"] == 3 and stored.config["dataset_reviewed_queries"] == 3

    def test_metrics_failure_labels_and_evidence_view(self, two_users):
        from utils.benchmark import (run_benchmark, get_evidence_view, get_evidence_rows, get_test_set_summary_table,
                                     get_tag_breakdown_table, get_limitations, list_runs)
        a, _ = two_users
        doc_id, sid = self._seed(a)
        gen, judge = _llms(lambda p: "The context does not contain this information." if "CEO" in p else "30 days.")
        with patch("utils.benchmark.retrieve_pipeline", side_effect=_retriever(doc_id)), \
             patch("retriver.vector_store.get_user_indexed_document_ids", return_value={doc_id}), \
             patch("llm.llm_handler.get_llm", return_value=gen), patch("llm.llm_handler.get_judge_llm", return_value=judge):
            report = run_benchmark(a, k=2, pipelines=["Naive"], include_llm_judges=True, test_set_id=sid, force_rerun=True)

        naive = report.per_pipeline["Naive"]
        # Answerable cases: refund hit at rank 2 (recall 1, mrr .5), exchange missed (0): means over 2 answerable only
        assert naive["n_answerable"] == 2 and naive["n_unanswerable"] == 1
        assert naive["hit_rate_at_k"] == 0.5 and naive["recall_at_k"] == 0.5 and naive["mrr"] == 0.25
        assert naive["abstention"] == {"n": 1, "correct_abstention_rate": 1.0, "false_answer_rate": 0.0}
        assert naive["failures"] == {"ok": 2, "missing_evidence": 0, "retrieval": 0, "ranking": 1, "generation": 0}
        assert naive["ci"]["recall_at_k"]["enough_data"] is False
        assert naive["by_tag"]["hard"] == {"n": 1, "hit_rate_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
        assert naive["by_tag"]["policy"]["recall_at_k"] == 0.5

        rows = {r["query"]: r for r in report.per_query}
        unans = rows["Who is the CEO?"]
        assert unans["recall_at_k"] is None and unans["abstained"] is True and unans["failure_category"] == "ok"
        assert rows["What is the exchange window?"]["failure_category"] == "ranking"
        refund = rows["What is the refund window?"]
        assert [(c["rank"], c["hit"], c["matched_gold_label"]) for c in refund["retrieved_chunks"]] == [(1, False, None), (2, True, "gold 1")]
        assert refund["retrieved_chunks"][1]["page_start"] == 4 and refund["answer_correctness"] == 0.9

        view = get_evidence_view(report, refund["case_id"])
        assert view["answerable"] and view["gold_evidence"][0]["passage"] == PASSAGE and view["expected_answer"] == "30 days"
        pp = view["per_pipeline"][0]
        assert pp["failure_category"] == "ok" and pp["answer"] == "30 days." and pp["retrieved_chunks"][1]["hit"]

        assert [r["query"] for r in get_evidence_rows(report, failed_only=True)] == ["What is the exchange window?"]
        assert [r["query"] for r in get_evidence_rows(report, tag="abstain")] == ["Who is the CEO?"]
        assert get_evidence_rows(report, pipeline="Hybrid") == []

        table = get_test_set_summary_table(report)[0]
        assert "not enough data" in table["Recall@2"] and table["Correct abstention"] == "100.0%"
        tags = {(r["Tag"], r["Pipeline"]): r for r in get_tag_breakdown_table(report)}
        assert tags[("hard", "Naive")]["Cases"] == 1
        assert any("not enough" in n.lower() or "below the" in n for n in get_limitations(report))
        assert list_runs(a)[0]["dataset"] == "test set: set-1"

    def test_false_answer_and_missing_evidence(self, two_users):
        from utils.benchmark import run_benchmark, classify_evidence_failure
        a, _ = two_users
        doc_id, sid = self._seed(a)
        gen, judge = _llms(lambda p: "The CEO is Jane Doe." if "CEO" in p else "30 days.")
        with patch("utils.benchmark.retrieve_pipeline", side_effect=_retriever(doc_id)), \
             patch("retriver.vector_store.get_user_indexed_document_ids", return_value=set()), \
             patch("llm.llm_handler.get_llm", return_value=gen), patch("llm.llm_handler.get_judge_llm", return_value=judge):
            report = run_benchmark(a, k=2, pipelines=["Naive"], include_llm_judges=True, test_set_id=sid, force_rerun=True, persist=False)
        naive = report.per_pipeline["Naive"]
        assert naive["abstention"]["false_answer_rate"] == 1.0
        assert naive["failures"]["generation"] == 1 and naive["failures"]["missing_evidence"] == 2
        assert classify_evidence_failure(True, True, True, None, {"answer_faithfulness": 0.2}) == "generation"
        assert classify_evidence_failure(True, False, True, True) == "ranking"
        assert classify_evidence_failure(True, False, True, False) == "retrieval"
        assert classify_evidence_failure(False, False, True, None, abstained=None) == "ok"

    def test_retrieval_only_run_has_no_abstention_data(self, two_users):
        from utils.benchmark import run_benchmark
        a, _ = two_users
        doc_id, sid = self._seed(a)
        with patch("utils.benchmark.retrieve_pipeline", side_effect=_retriever(doc_id)), \
             patch("retriver.vector_store.get_user_indexed_document_ids", return_value={doc_id}):
            report = run_benchmark(a, k=2, pipelines=["Naive"], include_llm_judges=False, test_set_id=sid, force_rerun=True, persist=False)
        assert report.per_pipeline["Naive"]["abstention"]["n"] == 0
        assert all(r["abstained"] is None for r in report.per_query)

    def test_cross_user_test_set_and_threshold_validation(self, two_users):
        from utils.benchmark import run_benchmark
        a, b = two_users
        doc_id, sid = self._seed(a)
        with pytest.raises(ValueError, match="not found"):
            run_benchmark(b, k=2, test_set_id=sid, force_rerun=True)
        with pytest.raises(ValueError, match="overlap_threshold"):
            run_benchmark(a, k=2, test_set_id=sid, overlap_threshold=0.0, force_rerun=True)

    def test_threshold_changes_hits(self, two_users):
        from utils.benchmark import run_benchmark
        a, _ = two_users
        doc_id, sid = self._seed(a)

        def half(q, uid, strategy, k):
            return PipelineResult(strategy, [RetrievedChunk("h", "policy.pdf", "Refunds are accepted within", 0.5, 1, strategy,
                                                            document_id=doc_id, page_start=4, page_end=4)], 1)
        with patch("utils.benchmark.retrieve_pipeline", side_effect=half), \
             patch("retriver.vector_store.get_user_indexed_document_ids", return_value={doc_id}):
            loose = run_benchmark(a, k=1, pipelines=["Naive"], test_set_id=sid, overlap_threshold=0.5, force_rerun=True, persist=False, enforce_quota=False)
            strict = run_benchmark(a, k=1, pipelines=["Naive"], test_set_id=sid, overlap_threshold=0.9, force_rerun=True, persist=False, enforce_quota=False)
        refund_loose = next(r for r in loose.per_query if "refund" in r["query"])
        refund_strict = next(r for r in strict.per_query if "refund" in r["query"])
        assert refund_loose["hit_rate_at_k"] == 1.0 and refund_strict["hit_rate_at_k"] == 0.0
        assert loose.config["overlap_threshold"] == 0.5 and strict.config["overlap_threshold"] == 0.9


# ─────────────────────────────────────────────
# Deletion
# ─────────────────────────────────────────────
class TestDeletion:
    def _count(self):
        from utils.database import SessionLocal, TestSet, TestCase, TestCaseEvidence
        db = SessionLocal()
        try:
            return (db.query(TestSet).count(), db.query(TestCase).count(), db.query(TestCaseEvidence).count())
        finally:
            db.close()

    def test_deleting_document_removes_dependent_cases_and_evidence(self, two_users):
        from utils.documents import delete_document
        a, b = two_users
        pol = _register(a, "policy.pdf", "hp")
        _register(a, "notes.txt", "hn")
        _register(b, "policy.pdf", "hb")
        rows = _rows({"question": "What do the notes say?", "answerable": True,
                      "gold_evidence": [{"document": "notes.txt", "passage": "these are plain notes"}]})
        sid = save_test_set(a, "s", validate_test_set(a, rows, chunk_loader=_loader))
        sid_b = save_test_set(b, "b", validate_test_set(b, _rows(), chunk_loader=_loader))
        assert self._count() == (2, 5, 3)
        with patch("retriver.vector_store.delete_user_source", return_value=0):
            removed = delete_document(a, pol)
        assert removed["test_cases"] == 1 and removed["evidence"] == 1
        left = load_test_set(a, sid)
        assert [c["question"] for c in left["cases"]] == ["Who is the CEO?", "What do the notes say?"]
        assert len(load_test_set(b, sid_b)["cases"]) == 2  # Bob's identically named document is untouched
        assert self._count() == (2, 4, 2)

    def test_delete_all_leaves_no_residue_for_that_user_only(self, two_users):
        from utils.documents import delete_all_user_data
        a, b = two_users
        _register(a, "policy.pdf", "hp")
        _register(b, "policy.pdf", "hb")
        save_test_set(a, "s", validate_test_set(a, _rows(), chunk_loader=_loader))
        save_test_set(b, "t", validate_test_set(b, _rows(), chunk_loader=_loader))
        with patch("retriver.vector_store.clear_user_index", return_value=0):
            counts = delete_all_user_data(a)
        assert (counts["test_sets"], counts["test_cases"], counts["evidence"]) == (1, 2, 1)
        assert list_test_sets(a) == [] and self._count() == (1, 2, 1)
        assert [t["name"] for t in list_test_sets(b)] == ["t"]
