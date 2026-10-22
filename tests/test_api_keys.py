"""
test_api_keys.py — Per-user API keys and automatic model fallback.

Covers: key validation messages, probe cache keyed by fingerprint (never the
key), suggest_model ordering, NvidiaChat switching once to a working model,
no switch on 401, the benchmark recording key source and switches without
the key, and the API using a per-request header without storing it.
"""

import json
import os
import sys
from contextlib import contextmanager

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm import model_registry as reg
from llm.llm_handler import NvidiaChat, ModelUnavailable, get_llm, invoke_tracked, resolve_api_key

KEY = "nvapi-test-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    reg.clear_cache()
    yield
    reg.clear_cache()


class TestKeyHandling:
    def test_fingerprint_and_display_never_expose_key(self):
        fp = reg.key_fingerprint(KEY)
        assert KEY not in fp and len(fp) == 12
        assert reg.key_display(KEY) == "…cdef"
        assert reg.key_display("") == "none"

    def test_validate_key_rejects_bad_prefix_and_empty(self):
        assert reg.validate_key("")[0] is False
        ok, msg, _ = reg.validate_key("sk-not-nvidia")
        assert ok is False and "nvapi-" in msg

    @patch("llm.model_registry._request", return_value=(200, {"data": [{"id": "m1"}, {"id": "m2"}]}))
    def test_validate_key_ok(self, _req):
        ok, msg, ids = reg.validate_key(KEY)
        assert ok and ids == ["m1", "m2"] and "2 models" in msg and KEY not in msg

    @patch("llm.model_registry._request", return_value=(401, "Unauthorized"))
    def test_validate_key_401(self, _req):
        ok, msg, _ = reg.validate_key(KEY)
        assert not ok and "401" in msg

    def test_resolve_api_key_prefers_user_key(self):
        assert resolve_api_key("  nvapi-user  ") == "nvapi-user"
        with patch("llm.llm_handler.NVIDIA_API_KEY", "nvapi-server"):
            assert resolve_api_key(None) == "nvapi-server"
        with patch("llm.llm_handler.NVIDIA_API_KEY", None):
            with pytest.raises(ValueError):
                resolve_api_key("")


class TestProbeAndSuggest:
    def _fake_probe(self, ok_models):
        def _p(api_key, model):
            return reg.ProbeResult(model, model in ok_models, 200 if model in ok_models else 404, 1, 5, "")
        return _p

    def test_suggest_prefers_non_reasoning_for_answers(self):
        with patch("llm.model_registry.probe_model", side_effect=self._fake_probe(
                {"nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-ultra-550b-a55b"})):
            assert reg.suggest_model(KEY, role="answer") == "nvidia/nemotron-3-ultra-550b-a55b"
            assert reg.suggest_model(KEY, exclude=["nvidia/nemotron-3-ultra-550b-a55b"]) == "nvidia/nemotron-3.5-lightning-30b-a3b"

    def test_suggest_none_when_nothing_works(self):
        with patch("llm.model_registry.probe_model", side_effect=self._fake_probe(set())):
            assert reg.suggest_model(KEY) is None

    def test_cache_is_per_key_fingerprint(self):
        calls = []

        def _p(api_key, model):
            calls.append(model)
            return reg.ProbeResult(model, True, 200, 1, 5, "")
        with patch("llm.model_registry.probe_model", side_effect=_p):
            reg.working_models(KEY)
            n = len(calls)
            reg.working_models(KEY)              # cached
            assert len(calls) == n
            reg.working_models("nvapi-other-key-9876543210")   # different key → new probes
            assert len(calls) == 2 * n
        assert all(KEY not in k for k in reg._cache)  # cache keyed by fingerprint, not by key


class TestNvidiaChatFallback:
    @contextmanager
    def _wrapper(self, first_exc, fallback="nvidia/nemotron-3-ultra-550b-a55b"):
        clients = []

        def build(self, model):
            c = MagicMock(name=model)
            if not clients:
                c.invoke.side_effect = first_exc
            else:
                c.invoke.return_value = MagicMock(content="ok", usage_metadata={"input_tokens": 3, "output_tokens": 1})
            clients.append((model, c))
            return c
        with patch.object(NvidiaChat, "_build", build), \
             patch("llm.model_registry.suggest_model", return_value=fallback) as suggest, \
             patch("llm.model_registry.mark_failed") as mark:
            llm = NvidiaChat("nvidia/nemotron-3-super-120b-a12b", KEY)
            yield llm, clients, suggest, mark

    def test_switches_once_on_404_and_reports_it(self):
      with self._wrapper(Exception("[404] Not Found\nFunction 'x': Not found for account 'y'")) as (llm, clients, suggest, mark):
        call = invoke_tracked(llm, "hi")
        assert call.text == "ok" and call.model == "nvidia/nemotron-3-ultra-550b-a55b"
        assert call.switched_from == "nvidia/nemotron-3-super-120b-a12b"
        assert [m for m, _ in clients] == ["nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b"]
        suggest.assert_called_once()
        assert suggest.call_args.kwargs["exclude"] == ["nvidia/nemotron-3-super-120b-a12b"]
        mark.assert_called_once()
        # second call reuses the replacement without probing again
        invoke_tracked(llm, "again")
        suggest.assert_called_once()

    def test_switches_on_410_and_403(self):
        for msg in ("[410] Gone end of life", "[403] Forbidden Authorization failed"):
            with self._wrapper(Exception(msg)) as (llm, clients, _, _):
                assert invoke_tracked(llm, "x").switched_from == "nvidia/nemotron-3-super-120b-a12b"

    def test_no_switch_on_bad_key(self):
        with self._wrapper(Exception("[401] Unauthorized")) as (llm, clients, suggest, _):
            with pytest.raises(ModelUnavailable, match="401"):
                llm.invoke("x")
            suggest.assert_not_called()

    def test_raises_clearly_when_no_replacement(self):
        with self._wrapper(Exception("[404] Not Found for account"), fallback=None) as (llm, _, _, _):
            with pytest.raises(ModelUnavailable, match="No working replacement"):
                llm.invoke("x")
            assert llm.switched_from is None

    def test_unrelated_errors_propagate(self):
        with self._wrapper(RuntimeError("connection reset")) as (llm, _, suggest, _):
            with pytest.raises(RuntimeError, match="connection reset"):
                llm.invoke("x")
            suggest.assert_not_called()

    def test_fallback_can_be_disabled(self):
        with patch.object(NvidiaChat, "_build", lambda self, m: MagicMock(invoke=MagicMock(side_effect=Exception("[404] account")))):
            llm = NvidiaChat("m", KEY, allow_fallback=False)
            with pytest.raises(ModelUnavailable):
                llm.invoke("x")

    def test_repr_hides_key(self):
        with patch.object(NvidiaChat, "_build", lambda self, m: MagicMock()):
            llm = NvidiaChat("m", KEY)
        assert KEY not in repr(llm) and KEY not in str(llm) and KEY not in f"{llm}"

    def test_get_llm_uses_user_key_over_server(self):
        with patch.object(NvidiaChat, "_build", lambda self, m: MagicMock()), \
             patch("llm.llm_handler.NVIDIA_API_KEY", "nvapi-server-key-000000"):
            assert get_llm(api_key=KEY).key_fingerprint == reg.key_fingerprint(KEY)
            assert get_llm().key_fingerprint == reg.key_fingerprint("nvapi-server-key-000000")


def _pr(ids, pipeline="Naive"):
    from retriver.retriver import PipelineResult, RetrievedChunk
    return PipelineResult(pipeline, [RetrievedChunk(c, "s", f"text {c}", 0.5, i + 1, pipeline) for i, c in enumerate(ids)], 3)


class TestBenchmarkRecordsKeyAndSwitch:
    @patch("utils.benchmark.retrieve_pipeline")
    @patch("retriver.vector_store.get_user_chunk_ids", return_value={"a"})
    def test_config_has_key_source_and_switch_but_never_key(self, _ids, mock_retrieve, two_users):
        from utils.database import SessionLocal, SyntheticEvalItem
        from utils.benchmark import run_benchmark, report_to_json, get_limitations
        a, _ = two_users
        db = SessionLocal()
        db.add(SyntheticEvalItem(user_id=a, query_text="What is a?", chunk_id="a", source_doc="d", doc_hash="h", reference_answer="A"))
        db.commit(); db.close()
        mock_retrieve.side_effect = lambda q, uid, strategy, k: _pr(["a"], strategy)

        gen = MagicMock(model="nvidia/nemotron-3-ultra-550b-a55b", switched_from="nvidia/nemotron-3-super-120b-a12b")
        gen.invoke.return_value = MagicMock(content="ans", usage_metadata={"input_tokens": 1, "output_tokens": 1})
        judge = MagicMock(model="j", switched_from=None)
        judge.invoke.return_value = MagicMock(content="Score: 0.9", usage_metadata={})
        with patch("llm.llm_handler.get_llm", return_value=gen) as g, patch("llm.llm_handler.get_judge_llm", return_value=judge) as j:
            rep = run_benchmark(a, k=1, pipelines=["Naive"], include_llm_judges=True, force_rerun=True,
                                persist=False, enforce_quota=False, api_key=KEY, answer_model="nvidia/nemotron-3-super-120b-a12b",
                                judge_model="j")
        assert g.call_args.kwargs["api_key"] == KEY and j.call_args.kwargs["api_key"] == KEY
        cfg = rep.config
        assert cfg["api_key_source"] == "user" and cfg["api_key_display"] == "…cdef"
        assert cfg["model_switches"]["generator"] == {"from": "nvidia/nemotron-3-super-120b-a12b", "to": "nvidia/nemotron-3-ultra-550b-a55b"}
        assert cfg["generator_model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert KEY not in report_to_json(rep)
        assert any("was not usable" in n for n in get_limitations(rep))

    @patch("utils.benchmark.retrieve_pipeline")
    @patch("retriver.vector_store.get_user_chunk_ids", return_value={"a"})
    def test_server_key_source_by_default(self, _ids, mock_retrieve, two_users):
        from utils.database import SessionLocal, SyntheticEvalItem
        from utils.benchmark import run_benchmark
        a, _ = two_users
        db = SessionLocal(); db.add(SyntheticEvalItem(user_id=a, query_text="What is a?", chunk_id="a", source_doc="d", doc_hash="h")); db.commit(); db.close()
        mock_retrieve.side_effect = lambda q, uid, strategy, k: _pr(["a"], strategy)
        rep = run_benchmark(a, k=1, pipelines=["Naive"], force_rerun=True, persist=False, enforce_quota=False)
        assert rep.config["api_key_source"] == "server" and "model_switches" not in rep.config


class TestApiHeader:
    @pytest.fixture
    def client(self, two_users):
        from app import app
        from utils.rate_limit import limiter
        from fastapi.testclient import TestClient
        limiter.enabled = False
        with TestClient(app) as c:
            yield c
        limiter.enabled = True

    def _token(self, user_id):
        from utils.routing import create_access_token
        return {"Authorization": f"Bearer {create_access_token(user_id)}"}

    @patch("utils.routing.get_llm")
    @patch("utils.routing.retrieve_pipeline")
    def test_header_key_used_for_request_only(self, mock_retrieve, mock_llm, client, two_users, caplog):
        a, _ = two_users
        mock_retrieve.return_value = _pr(["a"])
        mock_llm.return_value.invoke.return_value = MagicMock(content="ans", usage_metadata={})
        mock_llm.return_value.model = "m"
        mock_llm.return_value.switched_from = "old-m"
        r = client.post("/query", json={"question": "hi?"}, headers={**self._token(a), "X-NVIDIA-API-Key": KEY})
        assert r.status_code == 200, r.text
        assert mock_llm.call_args.kwargs["api_key"] == KEY
        body = r.json()
        assert body["api_key_source"] == "user" and body["model_switched_from"] == "old-m"
        assert KEY not in r.text and KEY not in caplog.text
        from utils.database import SessionLocal, UsageEvent
        db = SessionLocal(); rows = db.query(UsageEvent).all(); db.close()
        assert rows and all(KEY not in json.dumps({"k": r.kind}) for r in rows)

    @patch("utils.routing.get_llm")
    @patch("utils.routing.retrieve_pipeline")
    def test_model_unavailable_is_a_400(self, mock_retrieve, mock_llm, client, two_users):
        a, _ = two_users
        mock_retrieve.return_value = _pr(["a"])
        mock_llm.return_value.invoke.side_effect = ModelUnavailable("Model 'x' is not enabled for your NVIDIA key.")
        r = client.post("/query", json={"question": "hi?"}, headers=self._token(a))
        assert r.status_code == 400 and "not enabled" in r.json()["detail"]
