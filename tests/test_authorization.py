"""
test_authorization.py — FastAPI endpoints enforce authentication, ownership
and quotas. Uses TestClient with the vector store and LLM mocked (no network).
"""

import io
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient


@pytest.fixture
def client(two_users):
    from app import app
    from utils.rate_limit import limiter
    limiter.enabled = False
    with TestClient(app) as c:
        yield c
    limiter.enabled = True


def _token(user_id):
    from utils.routing import create_access_token
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


class TestAuthentication:
    def test_protected_routes_require_token(self, client):
        for method, path in [("get", "/documents"), ("get", "/history"), ("get", "/me/usage"),
                             ("post", "/query"), ("delete", "/documents/1"), ("delete", "/me/data")]:
            kwargs = {"json": {"question": "x"}} if method == "post" else {}
            r = getattr(client, method)(path, **kwargs)
            assert r.status_code == 401, path

    def test_garbage_token_rejected(self, client):
        r = client.get("/documents", headers={"Authorization": "Bearer not.a.jwt"})
        assert r.status_code == 401

    def test_token_for_deleted_user_rejected(self, client):
        r = client.get("/documents", headers=_token(99999))
        assert r.status_code == 401

    def test_login_wrong_password(self, client, two_users):
        r = client.post("/login", data={"username": "alice", "password": "nope"})
        assert r.status_code == 401

    def test_login_success_returns_scoped_token(self, client, two_users):
        from utils.database import SessionLocal, User
        from utils.routing import hash_password
        a, _ = two_users
        db = SessionLocal()
        u = db.query(User).get(a); u.hashed_password = hash_password("correct-horse"); db.commit(); db.close()
        r = client.post("/login", data={"username": "alice", "password": "correct-horse"})
        assert r.status_code == 200
        tok = r.json()["access_token"]
        r2 = client.get("/documents", headers={"Authorization": f"Bearer {tok}"})
        assert r2.status_code == 200 and r2.json() == []


class TestOwnership:
    def test_cannot_delete_other_users_document(self, client, two_users):
        from utils.documents import register_document, list_documents
        a, b = two_users
        doc_a = register_document(a, "a.pdf", "h", "file", 1)
        with patch("retriver.vector_store.delete_user_source", return_value=1):
            r = client.delete(f"/documents/{doc_a}", headers=_token(b))
            assert r.status_code == 404  # not 403: existence is not revealed
            assert [d.id for d in list_documents(a)] == [doc_a]
            r = client.delete(f"/documents/{doc_a}", headers=_token(a))
            assert r.status_code == 200
        assert list_documents(a) == []

    def test_history_is_per_user(self, client, two_users):
        from utils.memory import UserMemory
        a, b = two_users
        UserMemory(a).add_interaction("alice q", "alice a")
        assert client.get("/history", headers=_token(b)).json() == []
        assert client.get("/history", headers=_token(a)).json()[0]["question"] == "alice q"

    def test_documents_listing_is_per_user(self, client, two_users):
        from utils.documents import register_document
        a, b = two_users
        register_document(a, "a.pdf", "h1", "file", 1)
        register_document(b, "b.pdf", "h2", "file", 1)
        assert [d["source"] for d in client.get("/documents", headers=_token(a)).json()] == ["a.pdf"]
        assert [d["source"] for d in client.get("/documents", headers=_token(b)).json()] == ["b.pdf"]


class TestQueryEndpoint:
    @patch("utils.routing.get_llm")
    @patch("utils.routing.retrieve_pipeline")
    def test_query_scoped_and_usage_recorded(self, mock_retrieve, mock_llm, client, two_users):
        from retriver.retriver import PipelineResult, RetrievedChunk
        a, _ = two_users
        mock_retrieve.return_value = PipelineResult("Naive", [RetrievedChunk("c1", "a.pdf", "ctx", 0.9, 1, "Naive")], 5)
        msg = MagicMock(content="the answer", usage_metadata={"input_tokens": 10, "output_tokens": 3})
        mock_llm.return_value.invoke.return_value = msg
        mock_llm.return_value.model = "m"
        r = client.post("/query", json={"question": "hi?", "strategy": "Naive", "k": 3}, headers=_token(a))
        assert r.status_code == 200, r.text
        assert mock_retrieve.call_args.kwargs["user_id"] == a
        assert r.json()["usage"]["input_tokens"] == 10
        usage = client.get("/me/usage", headers=_token(a)).json()
        assert usage["queries_today"] == 1 and usage["input_tokens_total"] == 10

    def test_unknown_strategy_rejected(self, client, two_users):
        a, _ = two_users
        r = client.post("/query", json={"question": "hi?", "strategy": "Evil"}, headers=_token(a))
        assert r.status_code == 400

    def test_question_length_capped(self, client, two_users):
        a, _ = two_users
        r = client.post("/query", json={"question": "x" * 5000}, headers=_token(a))
        assert r.status_code == 422

    @patch("utils.routing.get_llm")
    @patch("utils.routing.retrieve_pipeline")
    def test_daily_query_quota(self, mock_retrieve, mock_llm, client, two_users):
        from retriver.retriver import PipelineResult
        from utils.documents import record_usage
        a, _ = two_users
        mock_retrieve.return_value = PipelineResult("Naive", [], 1)
        mock_llm.return_value.invoke.return_value = MagicMock(content="a", usage_metadata={})
        for _ in range(int(os.environ["MAX_QUERIES_PER_DAY"])):
            record_usage(a, "query")
        r = client.post("/query", json={"question": "hi?"}, headers=_token(a))
        assert r.status_code == 429


class TestUploadEndpoint:
    def test_rejects_bad_extension(self, client, two_users):
        a, _ = two_users
        r = client.post("/upload", files={"file": ("evil.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
                        headers=_token(a))
        assert r.status_code == 400

    def test_rejects_fake_pdf(self, client, two_users):
        a, _ = two_users
        r = client.post("/upload", files={"file": ("x.pdf", io.BytesIO(b"not a pdf at all"), "application/pdf")},
                        headers=_token(a))
        assert r.status_code == 400

    def test_rejects_oversize(self, client, two_users):
        from config import MAX_UPLOAD_SIZE_BYTES
        a, _ = two_users
        big = b"a" * (MAX_UPLOAD_SIZE_BYTES + 1)
        r = client.post("/upload", files={"file": ("big.txt", io.BytesIO(big), "text/plain")}, headers=_token(a))
        assert r.status_code == 413

    @patch("utils.routing.add_to_vector_store", return_value=1)
    def test_txt_upload_registers_ownership_and_cleans_temp(self, mock_add, client, two_users):
        from utils.documents import list_documents
        from config import UPLOAD_DIR
        a, b = two_users
        r = client.post("/upload", files={"file": ("../../notes.txt", io.BytesIO(b"hello world " * 20), "text/plain")},
                        headers=_token(a))
        assert r.status_code == 200, r.text
        assert mock_add.call_args.kwargs["user_id"] == a
        docs = list_documents(a)
        assert len(docs) == 1 and docs[0].source == "notes.txt"
        assert list_documents(b) == []
        # temp file removed, nothing escaped the private dir
        user_dir = os.path.join(UPLOAD_DIR, str(a))
        assert not os.listdir(user_dir)
        # duplicate upload is detected
        r2 = client.post("/upload", files={"file": ("notes.txt", io.BytesIO(b"hello world " * 20), "text/plain")},
                         headers=_token(a))
        assert r2.json().get("duplicate") is True

    @patch("utils.routing.add_to_vector_store", return_value=1)
    def test_document_quota(self, mock_add, client, two_users):
        from utils.documents import register_document
        a, _ = two_users
        for i in range(int(os.environ["MAX_DOCS_PER_USER"])):
            register_document(a, f"d{i}.txt", f"h{i}", "file", 1)
        r = client.post("/upload", files={"file": ("new.txt", io.BytesIO(b"more text here " * 5), "text/plain")},
                        headers=_token(a))
        assert r.status_code == 429


class TestUrlEndpoint:
    def test_private_url_blocked(self, client, two_users):
        a, _ = two_users
        r = client.post("/ingest-url", json={"url": "http://127.0.0.1:8000/admin"}, headers=_token(a))
        assert r.status_code == 400
        assert "blocked" in r.json()["detail"].lower()

    def test_metadata_endpoint_blocked(self, client, two_users):
        a, _ = two_users
        r = client.post("/ingest-url", json={"url": "http://169.254.169.254/latest/meta-data/"}, headers=_token(a))
        assert r.status_code == 400
