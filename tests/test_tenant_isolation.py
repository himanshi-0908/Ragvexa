"""
test_tenant_isolation.py — Every data path must be scoped to one user.

Covers: vector-store filters on all pipelines, ownership-checked deletion,
chat history, eval datasets, benchmark runs and session cache keys.
"""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document

from retriver.vector_store import tenant_filter
from retriver.retriver import retrieve_pipeline, PIPELINE_NAMES


def _doc(text, cid, user_id):
    return Document(page_content=text, metadata={"chunk_id": cid, "source": "t.pdf", "user_id": user_id})


def _mock_vs(mock_cls):
    vs = MagicMock()
    docs = [_doc("alpha text", "c1", 1), _doc("beta text", "c2", 1)]
    vs.similarity_search_with_score.return_value = [(d, 0.1) for d in docs]
    vs.similarity_search.return_value = docs
    vs.get.return_value = {"ids": ["a", "b"], "documents": [d.page_content for d in docs],
                           "metadatas": [d.metadata for d in docs]}
    vs.embeddings = MagicMock()
    vs.embeddings.embed_query.return_value = [0.1, 0.2]
    vs.embeddings.embed_documents.return_value = [[0.1, 0.2]] * 2
    mock_cls.return_value = vs
    return vs


class TestTenantFilter:
    def test_filter_contains_user_id(self):
        assert tenant_filter(7) == {"user_id": 7}

    def test_filter_with_source(self):
        f = tenant_filter(7, "doc.pdf")
        assert {"user_id": 7} in f["$and"] and {"source": "doc.pdf"} in f["$and"]

    @pytest.mark.parametrize("bad", [None, 0, -1, "1", 1.5, True])
    def test_invalid_user_id_rejected(self, bad):
        with pytest.raises(ValueError):
            tenant_filter(bad)


class TestPipelinesAlwaysScoped:
    @patch("retriver.retriver.get_vector_store")
    def test_every_pipeline_passes_user_filter(self, mock_cls):
        vs = _mock_vs(mock_cls)
        with patch("llm.llm_handler.get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="rewritten")
            for p in PIPELINE_NAMES:
                vs.reset_mock()
                retrieve_pipeline("q", user_id=42, strategy=p, k=3)
                calls = vs.similarity_search_with_score.call_args_list + vs.similarity_search.call_args_list
                assert calls, f"{p} made no vector search"
                for c in calls:
                    assert c.kwargs.get("filter") == {"user_id": 42}, f"{p} queried without tenant filter"
                for c in vs.get.call_args_list:
                    assert c.kwargs.get("where") == {"user_id": 42}

    @patch("retriver.retriver.get_vector_store")
    def test_pipeline_refuses_missing_user(self, mock_cls):
        _mock_vs(mock_cls)
        for p in PIPELINE_NAMES:
            with pytest.raises(ValueError):
                retrieve_pipeline("q", user_id=None, strategy=p, k=3)


class TestVectorStoreHelpers:
    @patch("retriver.vector_store.get_vector_store")
    def test_delete_user_source_scoped(self, mock_cls):
        from retriver.vector_store import delete_user_source
        vs = mock_cls.return_value
        vs.get.return_value = {"ids": ["x", "y"]}
        n = delete_user_source(3, "a.pdf")
        assert n == 2
        assert vs.get.call_args.kwargs["where"] == tenant_filter(3, "a.pdf")
        vs.delete.assert_called_once_with(ids=["x", "y"])

    @patch("retriver.vector_store.get_vector_store")
    def test_add_tags_owner(self, mock_cls):
        from retriver.vector_store import add_to_vector_store
        chunks = [Document(page_content="hello world", metadata={})]
        add_to_vector_store(chunks, user_id=9, source="s.txt")
        assert chunks[0].metadata["user_id"] == 9
        assert chunks[0].metadata["source"] == "s.txt"
        assert len(chunks[0].metadata["chunk_id"]) == 12

    def test_add_rejects_invalid_user(self):
        from retriver.vector_store import add_to_vector_store
        with pytest.raises(ValueError):
            add_to_vector_store([Document(page_content="x")], user_id=None)


class TestDocumentOwnership:
    def test_list_and_delete_are_per_user(self, two_users):
        from utils.documents import register_document, list_documents, delete_document, NotOwner
        a, b = two_users
        with patch("retriver.vector_store.delete_user_source", return_value=4) as del_src:
            doc_a = register_document(a, "a.pdf", "hash-a", "file", 4)
            doc_b = register_document(b, "b.pdf", "hash-b", "file", 2)

            assert [d.id for d in list_documents(a)] == [doc_a]
            assert [d.id for d in list_documents(b)] == [doc_b]

            # Bob cannot delete Alice's document
            with pytest.raises(NotOwner):
                delete_document(b, doc_a)
            assert [d.id for d in list_documents(a)] == [doc_a]
            del_src.assert_not_called()

            # Alice can, and the vector deletion is scoped to her
            removed = delete_document(a, doc_a)
            assert removed["chunks"] == 4
            del_src.assert_called_once_with(a, "a.pdf")
            assert list_documents(a) == []

    def test_delete_document_removes_eval_items_only_for_owner(self, two_users):
        from utils.documents import register_document, delete_document
        from utils.database import SessionLocal, SyntheticEvalItem
        a, b = two_users
        db = SessionLocal()
        db.add_all([
            SyntheticEvalItem(user_id=a, query_text="qa", chunk_id="c", source_doc="a.pdf", doc_hash="h"),
            SyntheticEvalItem(user_id=b, query_text="qb", chunk_id="c", source_doc="b.pdf", doc_hash="h"),
        ])
        db.commit(); db.close()
        with patch("retriver.vector_store.delete_user_source", return_value=0):
            doc_a = register_document(a, "a.pdf", "h", "file", 1)
            removed = delete_document(a, doc_a)
        assert removed["eval_items"] == 1
        db = SessionLocal()
        remaining = db.query(SyntheticEvalItem).all()
        db.close()
        assert [r.user_id for r in remaining] == [b]

    def test_delete_all_user_data_leaves_other_user_intact(self, two_users):
        from utils.documents import register_document, delete_all_user_data, list_documents, record_usage
        from utils.memory import UserMemory
        a, b = two_users
        with patch("retriver.vector_store.clear_user_index", return_value=1):
            register_document(a, "a.pdf", "ha", "file", 1)
            register_document(b, "b.pdf", "hb", "file", 1)
            UserMemory(a).add_interaction("hi", "there")
            UserMemory(b).add_interaction("yo", "bob")
            record_usage(a, "query")
            counts = delete_all_user_data(a)
        assert counts["documents"] == 1 and counts["chats"] == 1
        assert list_documents(a) == []
        assert [d.source for d in list_documents(b)] == ["b.pdf"]
        assert UserMemory(b).get_history() == [{"question": "yo", "answer": "bob"}]
        assert UserMemory(a).get_history() == []


class TestEvalAndRunsScoped:
    def test_eval_queries_and_review_scoped(self, two_users):
        from utils.database import SessionLocal, SyntheticEvalItem
        from utils.eval_dataset import load_eval_queries, set_reviewed, list_eval_items, delete_eval_items
        a, b = two_users
        db = SessionLocal()
        ia = SyntheticEvalItem(user_id=a, query_text="What is A?", chunk_id="ca", source_doc="a", doc_hash="ha")
        ib = SyntheticEvalItem(user_id=b, query_text="What is B?", chunk_id="cb", source_doc="b", doc_hash="hb")
        db.add_all([ia, ib]); db.commit(); ida, idb = ia.id, ib.id; db.close()

        assert [q["query"] for q in load_eval_queries(a)] == ["What is A?"]
        # Alice tries to review / delete Bob's item: no effect
        assert set_reviewed(a, [idb], True) == 0
        assert delete_eval_items(a, [idb]) == 0
        assert [i["id"] for i in list_eval_items(b)] == [idb]
        assert load_eval_queries(b)[0]["reviewed"] is False

    def test_benchmark_runs_scoped(self, two_users):
        from utils.benchmark import BenchmarkReport, save_run, list_runs, load_run
        a, b = two_users
        rep = BenchmarkReport(pipelines=["Naive"], k=5, per_pipeline={}, per_query=[], total_queries=1,
                              total_duration_ms=1, config={"k": 5})
        rid = save_run(a, rep)
        assert [r["id"] for r in list_runs(a)] == [rid]
        assert list_runs(b) == []
        assert load_run(b, rid) is None
        assert load_run(a, rid).run_id == rid

    def test_cache_key_includes_user(self):
        from utils.benchmark import _cache_key
        assert _cache_key(1, 5, ["Naive"], "h") != _cache_key(2, 5, ["Naive"], "h")
