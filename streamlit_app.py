import streamlit as st
import os
import json
import time
import hashlib
import pandas as pd
import streamlit_authenticator as stauth

from ingestion.text_loader import load_document
from ingestion.web_loader import load_web_url
from utils.chunking import chunk_documents
from retriver.vector_store import (
    add_to_vector_store,
    get_embedding_model_name,
)
from retriver.retriver import retrieve_context, retrieve_pipeline, PIPELINE_NAMES
from llm.llm_handler import get_llm, generate_response_tracked, generate_study_guide
from utils.memory import UserMemory
from utils.auth_helper import get_credentials, register_user
from utils.eval_dataset import (
    generate_eval_dataset,
    get_eval_dataset_stats,
    list_eval_items,
    set_reviewed,
    delete_eval_items,
    import_ground_truth,
)
from utils.benchmark import (
    run_benchmark,
    get_pipeline_summary_table,
    get_failure_analysis,
    get_failure_breakdown,
    get_limitations,
    report_to_json,
    list_runs,
    load_run,
    FAILURE_CATEGORIES,
)
from utils.documents import (
    QuotaExceeded,
    check_daily_quota,
    check_ingest_quota,
    content_hash,
    delete_all_user_data,
    delete_document,
    document_exists,
    list_documents,
    record_usage,
    register_document,
    usage_summary,
)
from utils.security import (
    UploadValidationError,
    private_upload_path,
    remove_quietly,
    validate_upload,
    write_private_file,
)
from config import (
    COOKIE_NAME,
    COOKIE_SECRET,
    COOKIE_EXPIRY_DAYS,
    EXTERNAL_LLM_NOTICE,
    EVAL_GENERATOR_MODELS,
    JUDGE_MODEL,
    LLM_MODEL,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_SIZE_MB,
)

from PIL import Image

favicon = Image.open("favicon.png") if os.path.exists("favicon.png") else "🌿"

st.set_page_config(
    page_title="Ragvexa — RAG Engineering Platform",
    page_icon=favicon,
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# SEO Meta Tags
# ─────────────────────────────────────────────────────────────────
st.markdown("""
    <!-- Basic SEO -->
    <meta name="description" content="Ragvexa is the ultimate RAG (Retrieval-Augmented Generation) Engineering & Evaluation Platform. Build, test, and benchmark advanced AI retrieval pipelines with real metrics.">
    <meta name="keywords" content="RAG, Retrieval Augmented Generation, LLM, Generative AI, RAG Benchmark, AI Evaluation, Vector Search, Streamlit, Machine Learning, Hybrid Search, RRF, Himanshi">
    <meta name="author" content="Himanshi">
    <meta name="robots" content="index, follow">
    <meta name="language" content="English">

    <!-- OpenGraph (Facebook / LinkedIn) -->
    <meta property="og:title" content="Ragvexa — Advanced RAG Engineering Platform">
    <meta property="og:description" content="The definitive platform for RAG Engineering. Upload documents, chat with your data, and benchmark retrieval strategies head-to-head. Built by Himanshi.">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="Ragvexa">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Ragvexa — Advanced RAG Engineering Platform">
    <meta name="twitter:description" content="Build, test, and benchmark advanced AI retrieval pipelines. The ultimate RAG evaluation tool.">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Global Brand CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    /* Root palette */
    :root {
        --forest: #1C2E1F;
        --sage: #6B8F4E;
        --sage-light: #C7D3B4;
        --warn: #E8C48A;
        --info: #BBD3E0;
        --purple: #D6C7DE;
        --red: #E8B4B4;
    }

    h1,h2,h3,h4,h5,h6 { font-family: 'Lora', serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    .stApp { font-family: 'Inter', sans-serif; }

    /* Form cards */
    div[data-testid="stForm"] {
        border-radius: 16px !important;
        padding: 2rem 2.5rem !important;
        box-shadow: 0 8px 32px rgba(28,46,31,0.06) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 2px solid var(--sage-light); }
    .stTabs [data-baseweb="tab"] { height: 46px; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.95rem; background: transparent; padding: 0 4px; }
    
    /* Auth toggle tabs */
    .auth-tab-active > button { background-color: var(--forest) !important; color: #F7F3EA !important; }
    .auth-tab-inactive > button { background-color: #FFFFFF !important; color: var(--forest) !important; border: 1px solid #E6E2D8 !important; }

    /* Badges */
    .badge {
        display: inline-block; padding: 0.2rem 0.65rem;
        font-size: 0.72rem; font-weight: 600; border-radius: 20px;
        margin-right: 0.4rem; font-family: 'Inter', sans-serif; letter-spacing: 0.02em;
    }
    .badge-chat   { background: var(--sage-light); color: var(--forest); }
    .badge-memory { background: var(--info);        color: #1C3A4A; }
    .badge-source { background: var(--purple);      color: #3A1C4A; }
    .badge-ideas  { background: var(--warn);        color: #4A3000; }
    .badge-eval   { background: var(--red);         color: #4A0000; }
    
    /* DataTable */
    .stDataFrame { border-radius: 10px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Auth Setup
# ─────────────────────────────────────────────────────────────────
credentials = get_credentials()
authenticator = stauth.Authenticate(
    credentials, COOKIE_NAME, COOKIE_SECRET, cookie_expiry_days=COOKIE_EXPIRY_DAYS
)
authenticator.login(location='unrendered')

# Keys that belong to the login widget / cookie and must survive a user switch.
_AUTH_KEYS = {
    "authentication_status", "name", "username", "logout", "init",
    "reg_success_msg", "li_user", "li_pass", "reg_fn", "reg_un", "reg_em", "reg_pw",
}


def _purge_session_for_user(current_user_id):
    """
    Drop every cached value (benchmark reports, study guide, inspector state)
    when the signed-in user changes, so one account can never see another's
    cached results in the same browser tab.
    """
    previous = st.session_state.get("_active_user_id")
    if previous == current_user_id:
        return
    for key in list(st.session_state.keys()):
        if key in _AUTH_KEYS or key.startswith("FormSubmitter"):
            continue
        del st.session_state[key]
    st.session_state["_active_user_id"] = current_user_id


# ─────────────────────────────────────────────────────────────────
# Login / Register Panel
# ─────────────────────────────────────────────────────────────────
if not st.session_state.get('authentication_status'):
    _purge_session_for_user(None)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.write("") # Add a little top spacing
        st.write("")
        with st.container(border=True):
            st.markdown("<div style='text-align:center;margin-bottom:1rem;'>", unsafe_allow_html=True)
            if os.path.exists("logo.png"):
                st.image("logo.png", width=140)
            else:
                st.markdown("<h1 style='font-size:2.5rem;'>🌿 Ragvexa</h1>", unsafe_allow_html=True)
            st.markdown("""
                <p style='font-family:"Lora",serif;font-size:1.1rem;font-style:italic;
                           color:#6B8F4E;margin:0.5rem 0 0;'>
                    Universal RAG. Any Source. Real Answers.
                </p>
                <p style='font-size:0.8rem;color:#8A9B7D;margin-top:0.4rem;'>
                    RAG Engineering &amp; Evaluation Platform
                </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

            with login_tab:
                if 'reg_success_msg' in st.session_state:
                    st.success(st.session_state.pop('reg_success_msg'))
                if st.session_state.get('authentication_status') is False:
                    st.error("Incorrect username or password. Please try again.")

                with st.form("login_form", clear_on_submit=False):
                    u = st.text_input("Username", key="li_user").strip()
                    p = st.text_input("Password", type="password", key="li_pass")
                    submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

                    if submitted:
                        if not u or not p:
                            st.error("Please enter both username and password.")
                        else:
                            with st.spinner("Authenticating…"):
                                ok = authenticator.authentication_controller.login(u, p)
                                if ok:
                                    authenticator.cookie_controller.set_cookie()
                                    st.rerun()
                                else:
                                    st.session_state['authentication_status'] = False
                                    st.rerun()

            with register_tab:
                with st.form("register_form", clear_on_submit=False):
                    fn = st.text_input("Full Name", key="reg_fn")
                    un = st.text_input("Username", key="reg_un")
                    em = st.text_input("Email", key="reg_em")
                    pw = st.text_input("Password", type="password", key="reg_pw",
                                       help="Minimum 8 characters")
                    submitted = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

                    if submitted:
                        if not all([fn, un, em, pw]):
                            st.error("Please fill in all fields.")
                        else:
                            with st.spinner("Creating account…"):
                                ok, msg = register_user(un, em, fn, pw)
                                if ok:
                                    st.session_state['reg_success_msg'] = "✅ Account created! Please switch to the Sign In tab."
                                    st.rerun()
                                else:
                                    st.error(msg)
    st.stop()


# ─────────────────────────────────────────────────────────────────
# Authenticated App
# ─────────────────────────────────────────────────────────────────
name = st.session_state.get('name', 'User')
username = st.session_state.get('username', '')
user_record = credentials["usernames"].get(username)
if not user_record:
    # Valid cookie for an account that no longer exists — sign out cleanly.
    authenticator.authentication_controller.logout()
    _purge_session_for_user(None)
    st.rerun()
user_id = user_record["user_id"]
_purge_session_for_user(user_id)
memory = UserMemory(user_id=user_id)


def _ingest_chunks(chunks, source, doc_hash, kind, size_bytes):
    """Quota check → vector store → ownership row. Returns chunk count."""
    check_ingest_quota(user_id, len(chunks))
    n = add_to_vector_store(chunks, user_id=user_id, source=source)
    register_document(user_id, source, doc_hash, kind, n, size_bytes)
    record_usage(user_id, "ingest")
    return n


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align:center;'>Welcome, {name.split()[0]}! 🌿</h3>",
                unsafe_allow_html=True)
    authenticator.logout("Sign Out", "sidebar")
    st.divider()

    st.markdown("### 🛠️ Active Pipeline")
    active_pipeline = st.selectbox(
        "Chat retrieval strategy:",
        PIPELINE_NAMES,
        key="sidebar_pipeline",
    )
    st.divider()

    # Embedding model info
    em_model = get_embedding_model_name()
    st.markdown(f"""
    <div style='background:#EEF4E6;border:1px solid #C7D3B4;border-radius:8px;padding:0.75rem 1rem;'>
        <p style='margin:0;font-size:0.78rem;font-weight:600;color:#8A9B7D;'>EMBEDDING MODEL (local)</p>
        <p style='margin:0;font-size:0.82rem;color:#1C2E1F;'>{em_model}</p>
        <p style='margin:0.4rem 0 0;font-size:0.78rem;font-weight:600;color:#8A9B7D;'>ANSWER MODEL (NVIDIA)</p>
        <p style='margin:0;font-size:0.82rem;color:#1C2E1F;'>{LLM_MODEL}</p>
        <p style='margin:0.4rem 0 0;font-size:0.78rem;font-weight:600;color:#8A9B7D;'>JUDGE MODEL (NVIDIA)</p>
        <p style='margin:0;font-size:0.82rem;color:#1C2E1F;'>{JUDGE_MODEL}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Usage & quotas
    usage = usage_summary(user_id)
    st.markdown("### 📈 Usage (24h)")
    st.caption(
        f"Queries: {usage['queries_today']}/{usage['max_queries_per_day']} · "
        f"Benchmarks: {usage['benchmarks_today']}/{usage['max_benchmarks_per_day']}"
    )
    cost_txt = f"${usage['cost_usd_total']:.4f}" if usage["cost_usd_total"] is not None else "n/a (no prices configured)"
    st.caption(
        f"Tokens all-time: {usage['input_tokens_total'] + usage['output_tokens_total']:,} · Cost: {cost_txt}"
    )

    # My documents
    st.divider()
    st.markdown("### 📚 My Documents")
    my_docs = list_documents(user_id)
    if not my_docs:
        st.caption("No documents yet.")
    for d in my_docs:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"{'🌐' if d.kind == 'url' else '📄'} {d.source[:40]} · {d.chunk_count} chunks")
        if c2.button("🗑", key=f"del_doc_{d.id}", help="Delete this document, its chunks and its eval items"):
            try:
                removed = delete_document(user_id, d.id)
                st.session_state.pop("benchmark_report", None)
                st.toast(f"Deleted {d.source[:30]} ({removed['chunks']} chunks, {removed['eval_items']} eval items)")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete: {e}")

    with st.expander("⚠️ Delete all my data"):
        st.caption("Removes your index, documents, eval datasets, chat history, benchmark runs and usage log.")
        confirm = st.checkbox("I understand this cannot be undone", key="confirm_wipe")
        if st.button("Delete everything", type="primary", disabled=not confirm, key="wipe_btn"):
            counts = delete_all_user_data(user_id)
            _purge_session_for_user(None)
            st.session_state["_active_user_id"] = user_id
            st.success(f"Deleted: {counts}")
            st.rerun()

    st.divider()
    st.markdown(f"""
    <p style='font-size:0.78rem;color:#8A9B7D;line-height:1.5;'>
        🔒 <strong>Data disclosure:</strong> {EXTERNAL_LLM_NOTICE}
    </p>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;'>
    <h1 style='margin:0;'>🌿 Ragvexa</h1>
    <span style='font-family:"Lora",serif;font-style:italic;font-size:1.1rem;color:#6B8F4E;'>
        RAG Engineering &amp; Evaluation Platform
    </span>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Top-level tabs ────────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs(["💬  Chat & Ingest", "📊  RAG Benchmark"])


# ═════════════════════════════════════════════════════════════════
# TAB 1: CHAT & INGEST
# ═════════════════════════════════════════════════════════════════
with main_tab1:
    col_ingest, col_chat = st.columns([1, 2], gap="large")

    # ── Ingestion Panel ──────────────────────────────────────────
    with col_ingest:
        st.markdown("## 📁 Data Ingestion")
        st.caption("🔒 Document text is sent to the NVIDIA API for answer generation and evaluation. Embeddings stay local. Full notice in the sidebar.")
        ingest_tab1, ingest_tab2 = st.tabs(["Upload Document", "Web URL"])

        with ingest_tab1:
            uploaded_file = st.file_uploader(f"Upload PDF or TXT (max {MAX_UPLOAD_SIZE_MB} MB)", type=["pdf", "txt"])
            gen_eval = st.checkbox(
                "Generate eval dataset after ingest",
                value=False,
                help="Uses the configured generator LLMs to create query→chunk pairs with reference "
                     f"answers for benchmarking. Models: {', '.join(EVAL_GENERATOR_MODELS)}. "
                     "Takes ~1–2 min for large documents.",
            )
            if st.button("⬆️ Ingest Document", use_container_width=True, type="primary"):
                if not uploaded_file:
                    st.warning("Please upload a file first.")
                else:
                    temp_path = None
                    with st.spinner("Processing document…"):
                        try:
                            data = uploaded_file.getvalue()
                            validate_upload(uploaded_file.name, len(data), data[:16])
                            dh = content_hash(data)
                            if document_exists(user_id, dh):
                                st.info("ℹ️ This document is already in your index.")
                            else:
                                temp_path = private_upload_path(user_id, uploaded_file.name)
                                write_private_file(temp_path, data)
                                docs = load_document(temp_path)
                                if docs:
                                    chunks = chunk_documents(docs)
                                    source = os.path.basename(uploaded_file.name)[:200]
                                    n = _ingest_chunks(chunks, source, dh, "file", len(data))
                                    st.success(f"✅ {n} chunks ingested from **{source}**")

                                    if gen_eval:
                                        with st.spinner("Generating eval dataset (LLM)…"):
                                            saved = generate_eval_dataset(
                                                chunks,
                                                user_id=user_id,
                                                source_name=source,
                                                doc_hash=dh,
                                            )
                                        if saved:
                                            st.info(f"🧪 {saved} synthetic eval queries saved.")
                                        else:
                                            st.info("ℹ️ No new eval queries were saved (already exists or generation failed).")
                                else:
                                    st.error("Document is empty or unreadable.")
                        except (UploadValidationError, QuotaExceeded) as e:
                            st.error(f"{e}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            remove_quietly(temp_path)

        with ingest_tab2:
            url_input = st.text_input("Enter URL:", placeholder="https://example.com/page", max_chars=2000)
            gen_eval_url = st.checkbox("Generate eval dataset for URL", value=False, key="eval_url")
            if st.button("🌐 Ingest URL", use_container_width=True, type="primary"):
                if not url_input.strip():
                    st.warning("Please enter a URL first.")
                else:
                    with st.spinner("Fetching webpage…"):
                        try:
                            docs = load_web_url(url_input.strip())
                            if docs:
                                chunks = chunk_documents(docs)
                                url_hash = content_hash("\n".join(d.page_content for d in docs).encode())
                                if document_exists(user_id, url_hash):
                                    st.info("ℹ️ This page is already in your index.")
                                else:
                                    source = url_input.strip()[:200]
                                    n = _ingest_chunks(chunks, source, url_hash, "url", 0)
                                    st.success(f"✅ {n} chunks ingested from URL")

                                    if gen_eval_url:
                                        with st.spinner("Generating eval dataset (LLM)…"):
                                            saved = generate_eval_dataset(
                                                chunks,
                                                user_id=user_id,
                                                source_name=source,
                                                doc_hash=url_hash,
                                            )
                                        if saved:
                                            st.info(f"🧪 {saved} synthetic eval queries saved.")
                            else:
                                st.error("No content found at URL.")
                        except (ValueError, QuotaExceeded) as e:
                            st.error(f"{e}")
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()

        # Proactive tools
        st.markdown("### ⚡ Knowledge Tools")
        st.markdown('<span class="badge badge-ideas">Synthesis</span>', unsafe_allow_html=True)
        if st.button("📝 Synthesize & Summarize Ingested Knowledge", use_container_width=True):
            with st.spinner("Generating synthesis report…"):
                try:
                    check_daily_quota(user_id, "query")
                    ctx = retrieve_context(
                        "summarize key concepts definitions and topics",
                        user_id=user_id, k=8,
                    )
                    if not ctx.strip():
                        st.warning("Please ingest some documents first.")
                    else:
                        llm = get_llm()
                        guide = generate_study_guide(llm, ctx)
                        record_usage(user_id, "query")
                        st.session_state["study_guide"] = guide
                        st.success("Synthesis report ready!")
                except QuotaExceeded as e:
                    st.error(f"{e}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Chat Panel ───────────────────────────────────────────────
    with col_chat:
        st.markdown("## 💬 Chat with your Copilot")

        if "study_guide" in st.session_state:
            with st.expander("📖 Knowledge Synthesis Report", expanded=True):
                st.markdown(st.session_state["study_guide"])
                if st.button("✕ Close Report"):
                    del st.session_state["study_guide"]
                    st.rerun()

        # Display history
        history = memory.get_history(limit=5)
        for item in history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])

        # Chat input
        question = st.chat_input("Ask anything about your ingested data…", max_chars=MAX_QUESTION_CHARS)
        if question:
            last_t = st.session_state.get("last_q_time", 0)
            if time.time() - last_t < 2:
                st.warning("Please wait a moment before asking again.")
            else:
                st.session_state["last_q_time"] = time.time()
                with st.chat_message("user"):
                    st.write(question)

                with st.spinner(f"Retrieving ({active_pipeline})…"):
                    try:
                        check_daily_quota(user_id, "query")
                        # Retrieve using active pipeline
                        pipeline_res = retrieve_pipeline(
                            question, user_id, strategy=active_pipeline, k=5
                        )
                        context = "\n\n".join(c.text for c in pipeline_res.chunks)

                        # Generate answer
                        history_str = memory.get_history_string(limit=3)
                        llm = get_llm()
                        call = generate_response_tracked(llm, context, question, history_string=history_str)
                        answer = call.text
                        memory.add_interaction(question, answer)
                        record_usage(user_id, "query", call.input_tokens, call.output_tokens, call.cost_usd)

                        with st.chat_message("assistant"):
                            st.markdown(
                                f'<span class="badge badge-chat">Chat</span>'
                                f'<span class="badge badge-memory">Memory</span>'
                                f'<span class="badge badge-chat">{active_pipeline}</span>',
                                unsafe_allow_html=True,
                            )
                            st.write(answer)

                            # Source attribution expander with chunk details
                            with st.expander("🔍 View Retrieved Context & Sources"):
                                st.markdown('<span class="badge badge-source">Source Attribution</span>',
                                            unsafe_allow_html=True)
                                if pipeline_res.rewritten_query:
                                    st.markdown(
                                        f"**Rewritten Query:** *\"{pipeline_res.rewritten_query}\"*"
                                    )
                                if pipeline_res.error:
                                    st.warning(f"⚠️ {pipeline_res.error}")

                                cost_s = f" | **Cost:** `${call.cost_usd:.5f}`" if call.cost_usd is not None else ""
                                st.markdown(
                                    f"**Retrieval latency:** `{pipeline_res.retrieval_latency_ms} ms`"
                                    + (f" | **Reranking:** `{pipeline_res.reranking_latency_ms} ms`"
                                       if pipeline_res.reranking_latency_ms else "")
                                    + f" | **Generation:** `{call.latency_ms} ms`"
                                    + f" | **Tokens:** `{call.input_tokens}+{call.output_tokens}`"
                                    + cost_s
                                )
                                for i, chunk in enumerate(pipeline_res.chunks, 1):
                                    st.markdown(
                                        f"**Chunk {i}** — `{chunk.source}` "
                                        f"| score `{chunk.score:.4f}`"
                                    )
                                    st.markdown(
                                        f"> {chunk.text[:400]}{'…' if len(chunk.text) > 400 else ''}"
                                    )

                    except QuotaExceeded as e:
                        st.error(f"{e}")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════
# TAB 2: RAG BENCHMARK
# ═════════════════════════════════════════════════════════════════
with main_tab2:
    st.markdown("## 📊 RAG Pipeline Benchmark")
    st.markdown("""
    <p style='color:#8A9B7D;font-size:0.92rem;'>
        Compare retrieval strategies head-to-head on the same dataset with the same K,
        prompts, generator and judge models. Synthetic labels show <strong>relative</strong>
        pipeline differences; use the Ground Truth tab to add or review human labels
        before making absolute accuracy claims.
    </p>
    """, unsafe_allow_html=True)

    stats = get_eval_dataset_stats(user_id)
    total_q = stats.get("total_queries", 0)
    unique_q = stats.get("unique_queries", 0)
    reviewed_q = stats.get("reviewed_unique_queries", 0)
    sources = stats.get("sources", {})

    bench_tab_run, bench_tab_gt, bench_tab_hist = st.tabs(["▶ Run", "✅ Ground Truth", "🗂 Run History"])

    # ── Ground Truth tab ─────────────────────────────────────────
    with bench_tab_gt:
        st.markdown("### ✅ Ground Truth & Review")
        st.caption(
            "Synthetic items were written by an LLM. Mark the ones you have verified as reviewed, "
            "delete bad ones, or import human-labelled items. Benchmarks can be restricted to reviewed items."
        )
        gt_items = list_eval_items(user_id)
        if gt_items:
            gt_df = pd.DataFrame(gt_items)[
                ["id", "reviewed", "origin", "query", "chunk_id", "source_doc", "generator_model", "reference_answer"]
            ]
            edited = st.data_editor(
                gt_df,
                hide_index=True,
                use_container_width=True,
                disabled=["id", "origin", "query", "chunk_id", "source_doc", "generator_model", "reference_answer"],
                column_config={"reviewed": st.column_config.CheckboxColumn("Reviewed")},
                key="gt_editor",
                height=320,
            )
            gc1, gc2 = st.columns(2)
            if gc1.button("💾 Save review flags", use_container_width=True):
                to_review = [int(r.id) for r in edited.itertuples() if bool(r.reviewed)]
                to_unreview = [int(r.id) for r in edited.itertuples() if not bool(r.reviewed)]
                n1 = set_reviewed(user_id, to_review, True)
                n2 = set_reviewed(user_id, to_unreview, False)
                st.success(f"Updated {n1 + n2} items.")
                st.rerun()
            del_ids = gc2.multiselect("Delete items (by id)", [r["id"] for r in gt_items], key="gt_del")
            if gc2.button("🗑 Delete selected", use_container_width=True, disabled=not del_ids):
                n = delete_eval_items(user_id, [int(i) for i in del_ids])
                st.success(f"Deleted {n} items.")
                st.rerun()
        else:
            st.info("No evaluation items yet. Ingest a document with \"Generate eval dataset\" or import ground truth below.")

        st.divider()
        st.markdown("#### 📥 Import human ground truth (JSON)")
        st.caption(
            'A JSON list of objects: {"query": "...", "reference_answer": "...", '
            '"relevant_chunk_ids": ["..."] and/or "relevant_passages": ["exact text from your document"]}. '
            "Passages are matched against your indexed chunks."
        )
        gt_file = st.file_uploader("Ground-truth JSON", type=["json"], key="gt_upload")
        if gt_file and st.button("Import", key="gt_import_btn"):
            try:
                payload = json.loads(gt_file.getvalue().decode("utf-8"))
                if isinstance(payload, dict):
                    payload = payload.get("items", [])
                if not isinstance(payload, list):
                    raise ValueError("JSON must be a list of items")
                if len(payload) > 500:
                    raise ValueError("At most 500 items per import")
                res = import_ground_truth(user_id, payload, source_doc=gt_file.name[:100])
                st.success(f"Imported {res['saved']} query→chunk pairs.")
                if res["skipped"]:
                    st.warning("Skipped:\n- " + "\n- ".join(res["skipped"][:20]))
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

    # ── Run History tab ──────────────────────────────────────────
    with bench_tab_hist:
        st.markdown("### 🗂 Previous Runs")
        runs = list_runs(user_id)
        if not runs:
            st.info("No saved runs yet.")
        else:
            st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
            pick = st.selectbox("Load run", [r["id"] for r in runs], key="hist_pick")
            if st.button("Load selected run", key="hist_load"):
                loaded = load_run(user_id, int(pick))
                if loaded:
                    st.session_state["benchmark_report"] = loaded
                    st.success(f"Loaded run {pick}. See the Run tab.")
                else:
                    st.error("Run not found.")

    # ── Run tab ──────────────────────────────────────────────────
    with bench_tab_run:
        bm_col1, bm_col2 = st.columns([1, 2], gap="large")

        with bm_col1:
            st.markdown("### 🧪 Evaluation Dataset Status")
            if total_q == 0:
                st.info(
                    "**No eval queries found.**\n\n"
                    "Go to **Chat & Ingest → Upload Document** and check "
                    "\"Generate eval dataset after ingest\", or import ground truth in the Ground Truth tab."
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Query-Chunk Pairs", total_q)
                m2.metric("Unique Queries", unique_q)
                m3.metric("Reviewed Queries", reviewed_q)
                if stats.get("generator_models"):
                    st.caption("Generator models: " + ", ".join(f"{m} ({n})" for m, n in stats["generator_models"].items()))
                if sources:
                    st.markdown("**Sources:**")
                    for src, cnt in sources.items():
                        st.markdown(f"- `{src[:60]}` — {cnt} pairs")

            st.divider()

            st.markdown("### ⚙️ Benchmark Settings")
            k_val = st.slider("Top-K retrieved documents", min_value=1, max_value=10, value=5)
            selected_pipelines = st.multiselect(
                "Pipelines to evaluate:",
                PIPELINE_NAMES,
                default=PIPELINE_NAMES,
            )
            reviewed_only = st.checkbox(
                "Human-reviewed items only",
                value=False,
                disabled=(reviewed_q == 0),
                help="Restrict the dataset to items you marked as reviewed in the Ground Truth tab.",
            )
            include_llm = st.checkbox(
                "Include LLM-judge metrics",
                value=False,
                help=(
                    f"Generates an answer with {LLM_MODEL} and scores it with the independent judge "
                    f"{JUDGE_MODEL} on relevance, faithfulness, groundedness and correctness. "
                    "Makes 6 LLM calls per query per pipeline; token usage is recorded."
                ),
            )
            if include_llm and JUDGE_MODEL == LLM_MODEL:
                st.warning("JUDGE_MODEL equals LLM_MODEL — judge scores will not be independent.")

            effective_q = reviewed_q if reviewed_only else unique_q
            run_btn = st.button(
                "▶ Run Benchmark",
                type="primary",
                use_container_width=True,
                disabled=(effective_q == 0 or not selected_pipelines),
            )

        with bm_col2:
            if total_q == 0:
                st.markdown("""
                <div style='background:#FFFFFF;border:1px dashed #C7D3B4;border-radius:12px;
                            padding:3rem;text-align:center;'>
                    <p style='font-size:2rem;'>🌿</p>
                    <p style='color:#8A9B7D;'>
                        Ingest documents and generate an eval dataset to unlock benchmarking.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # ── Run Benchmark ────────────────────────────────────────
        if run_btn and effective_q > 0 and selected_pipelines:
            progress_bar = st.progress(0, text="Initialising benchmark…")
            status_text = st.empty()

            def _progress(current, total, msg):
                pct = int((current / total) * 100) if total > 0 else 0
                progress_bar.progress(pct, text=f"{pct}% — {msg}")
                status_text.markdown(f"`{msg}`")

            with st.spinner("Running benchmark…"):
                try:
                    report = run_benchmark(
                        user_id=user_id,
                        k=k_val,
                        pipelines=selected_pipelines,
                        include_llm_judges=include_llm,
                        progress_callback=_progress,
                        force_rerun=True,
                        reviewed_only=reviewed_only,
                    )
                    st.session_state["benchmark_report"] = report
                    progress_bar.progress(100, text="✅ Benchmark complete")
                    status_text.empty()
                except QuotaExceeded as e:
                    st.error(f"{e}")
                except Exception as e:
                    st.error(f"Benchmark failed: {e}")

        # ── Results Display ──────────────────────────────────────
        report = st.session_state.get("benchmark_report")
        if report and report.total_queries > 0:
            st.divider()
            run_label = f" | run #{report.run_id}" if report.run_id else ""
            st.markdown(
                f"### 📈 Results — Top-{report.k} | {report.total_queries} queries | "
                f"{report.total_duration_ms:,} ms total{run_label}"
            )

            cfg = report.config or {}
            with st.expander("🧾 Run configuration (for reproducibility)", expanded=False):
                st.json({k_: v for k_, v in cfg.items() if k_ not in ("started_at", "finished_at")})
            st.download_button(
                "⬇️ Download full report (JSON)",
                data=report_to_json(report),
                file_name=f"ragvexa_benchmark_{cfg.get('dataset_fingerprint', 'run')}.json",
                mime="application/json",
            )

            limitations = get_limitations(report)
            if limitations:
                st.warning("**Limitations**\n\n" + "\n".join(f"- {n}" for n in limitations))

            # Summary table
            summary_rows = get_pipeline_summary_table(report)
            if summary_rows:
                df = pd.DataFrame(summary_rows)
                st.dataframe(df, hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Chart section ────────────────────────────────────
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Recall@K vs Pipeline")
                chart_data = {
                    row["Pipeline"]: float(row["Recall@K"]) if row["Recall@K"] != "—" else 0.0
                    for row in summary_rows
                }
                if chart_data:
                    st.bar_chart(chart_data, y_label="Recall@K", use_container_width=True)

            with chart_col2:
                st.markdown("#### MRR vs Pipeline")
                mrr_data = {
                    row["Pipeline"]: float(row["MRR"]) if row["MRR"] != "—" else 0.0
                    for row in summary_rows
                }
                if mrr_data:
                    st.bar_chart(mrr_data, y_label="MRR", use_container_width=True)

            # Latency comparison
            st.markdown("#### Retrieval Latency (ms) vs Pipeline")
            latency_data = {row["Pipeline"]: int(row["Retr. ms"]) for row in summary_rows}
            st.bar_chart(latency_data, y_label="ms", use_container_width=True)

            # ── LLM Judge metrics (if run) ────────────────────────
            if cfg.get("include_llm_judges") or any(row.get("Context Rel.") not in (None, "—") for row in summary_rows):
                st.divider()
                judge_name = cfg.get("judge_model") or "generator model"
                st.markdown(f"#### 🤖 LLM-Judge Generation Metrics (judge: `{judge_name}`)")
                gen_cols = st.columns(len(summary_rows))
                for i, row in enumerate(summary_rows):
                    with gen_cols[i]:
                        st.markdown(f"**{row['Pipeline']}**")
                        st.metric("Context Relevance", row.get("Context Rel.", "—"))
                        st.metric("Faithfulness", row.get("Faithfulness", "—"))
                        st.metric("Groundedness", row.get("Groundedness", "—"))
                        st.metric("Answer Relevance", row.get("Ans. Rel.", "—"))
                        st.metric("Correctness", row.get("Correctness", "—"))
                        st.metric("Tokens / Cost", f"{row.get('Tokens', '—')} / {row.get('Cost (USD)', 'n/a')}")

            # ── Failure Analysis ─────────────────────────────────
            st.divider()
            st.markdown("#### 🧭 Failure Breakdown by Stage")
            st.caption(
                "missing_evidence: labelled chunks are no longer in the index · retrieval: not in the wider "
                "candidate pool · ranking: in the pool but outside top-K · generation: retrieved correctly "
                "but the judged answer scored below 0.5 · ok: retrieval (and answer, if judged) succeeded."
            )
            st.dataframe(pd.DataFrame(get_failure_breakdown(report)), hide_index=True, use_container_width=True)

            failures = get_failure_analysis(report)
            if failures:
                st.markdown("#### ⚠️ Queries with Recall@K = 0 across all pipelines")
                st.warning(
                    f"{len(failures)} queries returned zero relevant results across all {len(report.pipelines)} pipelines. "
                    "These represent gaps in your ingested knowledge, deleted documents, or mismatched chunk granularity."
                )
                st.dataframe(pd.DataFrame(failures), hide_index=True, use_container_width=True)

            # ── Per-query Inspector ───────────────────────────────
            st.divider()
            st.markdown("### 🔎 Query Inspector")
            st.markdown(
                "<p style='color:#8A9B7D;font-size:0.88rem;'>Select a query to inspect "
                "retrieved chunks, ranks, relevance status, failure stage and answers per pipeline.</p>",
                unsafe_allow_html=True,
            )

            all_queries = sorted(set(row["query"] for row in report.per_query))
            selected_q = st.selectbox("Select query:", all_queries, key="inspect_query")

            if selected_q:
                q_rows = [r for r in report.per_query if r["query"] == selected_q]

                relevant_ids = q_rows[0]["relevant_chunk_ids"] if q_rows else []
                left_col, right_col = st.columns([1, 2])

                with left_col:
                    st.markdown("**Query details**")
                    st.markdown(f"> {selected_q}")
                    origin = q_rows[0].get("origin", "synthetic") if q_rows else "synthetic"
                    rev = "reviewed" if (q_rows and q_rows[0].get("reviewed")) else "unreviewed"
                    st.caption(f"Label origin: {origin} · {rev}")
                    st.markdown("**Expected relevant chunk IDs:**")
                    for cid in relevant_ids:
                        st.markdown(f"- `{cid}`")
                    if q_rows and q_rows[0].get("reference_answer"):
                        st.markdown("**Reference answer:**")
                        st.markdown(f"> {q_rows[0]['reference_answer'][:600]}")

                    for row in q_rows:
                        if row.get("rewritten_query"):
                            st.markdown(f"**Rewritten query ({row['pipeline']}):**")
                            st.markdown(f"> *{row['rewritten_query']}*")

                with right_col:
                    for row in q_rows:
                        p_name = row["pipeline"]
                        rec = row.get("recall_at_k", 0)
                        mrr_val = row.get("mrr", 0)
                        prec = row.get("precision_at_k", 0)
                        ndcg = row.get("ndcg_at_k", 0)
                        cat = row.get("failure_category", "—")

                        color = "#EEF4E6" if rec > 0 else "#FFF0F0"
                        st.markdown(
                            f"""<div style='background:{color};border:1px solid #E6E2D8;
                                            border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.75rem;'>
                                <strong>{p_name}</strong>
                                &nbsp; Recall@{report.k}: <code>{rec:.4f}</code>
                                &nbsp; MRR: <code>{mrr_val:.4f}</code>
                                &nbsp; P@{report.k}: <code>{prec:.4f}</code>
                                &nbsp; nDCG@{report.k}: <code>{ndcg:.4f}</code>
                                &nbsp; {row.get("retrieval_latency_ms", "?")}&nbsp;ms
                                &nbsp; stage: <code>{cat}</code>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        retrieved = row.get("retrieved_chunks", [])
                        if retrieved:
                            for chunk in retrieved:
                                icon = "✅" if chunk["relevant"] else "❌"
                                st.markdown(
                                    f"&nbsp;&nbsp;{icon} **Rank {chunk['rank']}** — "
                                    f"`{chunk['chunk_id']}` | `{chunk['source']}` "
                                    f"| score `{chunk['score']:.4f}`"
                                )
                        if row.get("answer"):
                            with st.expander(f"Answer ({p_name})"):
                                st.markdown(row["answer"])
                                st.caption(
                                    f"faithfulness {row.get('answer_faithfulness')} · groundedness {row.get('answer_groundedness')} · "
                                    f"relevance {row.get('answer_relevance')} · correctness {row.get('answer_correctness')} · "
                                    f"tokens {row.get('total_tokens')} · cost {row.get('cost_usd') if row.get('cost_usd') is not None else 'n/a'}"
                                )
                        if row.get("error"):
                            st.warning(f"⚠️ {row['error']}")
