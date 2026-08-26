import streamlit as st
import os
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
    get_user_doc_sources,
)
from retriver.retriver import retrieve_context, retrieve_pipeline, PIPELINE_NAMES
from llm.llm_handler import get_llm, generate_response, generate_study_guide
from utils.memory import UserMemory
from utils.auth_helper import get_credentials, register_user
from utils.eval_dataset import (
    generate_eval_dataset,
    load_eval_queries,
    get_eval_dataset_stats,
    _file_hash,
)
from utils.benchmark import (
    run_benchmark,
    get_pipeline_summary_table,
    get_failure_analysis,
)
from config import COOKIE_NAME, COOKIE_SECRET

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
    credentials, COOKIE_NAME, COOKIE_SECRET, cookie_expiry_days=30
)
authenticator.login(location='unrendered')

# ─────────────────────────────────────────────────────────────────
# Login / Register Panel
# ─────────────────────────────────────────────────────────────────
if not st.session_state.get('authentication_status'):
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
user_id = credentials["usernames"][username]["user_id"]
memory = UserMemory(user_id=user_id)

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
        <p style='margin:0;font-size:0.78rem;font-weight:600;color:#8A9B7D;'>EMBEDDING MODEL</p>
        <p style='margin:0;font-size:0.82rem;color:#1C2E1F;'>{em_model}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:0.82rem;color:#8A9B7D;line-height:1.5;'>
        Ragvexa is a RAG Engineering &amp; Evaluation Platform.
        Ingest documents, chat with your data, and benchmark
        4 retrieval strategies head-to-head.
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
        ingest_tab1, ingest_tab2 = st.tabs(["Upload Document", "Web URL"])

        with ingest_tab1:
            uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
            gen_eval = st.checkbox(
                "Generate eval dataset after ingest",
                value=False,
                help="Uses LLM to create synthetic query→chunk pairs for benchmarking. "
                     "Takes ~1–2 min for large documents.",
            )
            if st.button("⬆️ Ingest Document", use_container_width=True, type="primary"):
                if not uploaded_file:
                    st.warning("Please upload a file first.")
                else:
                    with st.spinner("Processing document…"):
                        temp_path = f"temp_{user_id}_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        try:
                            docs = load_document(temp_path)
                            if docs:
                                chunks = chunk_documents(docs)
                                n = add_to_vector_store(chunks, user_id=user_id)
                                st.success(f"✅ {n} chunks ingested from **{uploaded_file.name}**")

                                if gen_eval:
                                    dh = _file_hash(temp_path)
                                    with st.spinner("Generating eval dataset (LLM)…"):
                                        saved = generate_eval_dataset(
                                            chunks,
                                            user_id=user_id,
                                            source_name=uploaded_file.name,
                                            doc_hash=dh,
                                        )
                                    if saved:
                                        st.info(f"🧪 {saved} synthetic eval queries saved.")
                                    else:
                                        st.info("ℹ️ Eval dataset already exists for this document.")
                            else:
                                st.error("Document is empty or unreadable.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

        with ingest_tab2:
            url_input = st.text_input("Enter URL:", placeholder="https://example.com/page")
            gen_eval_url = st.checkbox("Generate eval dataset for URL", value=False, key="eval_url")
            if st.button("🌐 Ingest URL", use_container_width=True, type="primary"):
                if not url_input.strip():
                    st.warning("Please enter a URL first.")
                else:
                    with st.spinner("Scraping webpage…"):
                        try:
                            docs = load_web_url(url_input.strip())
                            if docs:
                                chunks = chunk_documents(docs)
                                n = add_to_vector_store(chunks, user_id=user_id)
                                st.success(f"✅ {n} chunks ingested from URL")

                                if gen_eval_url:
                                    url_hash = hashlib.md5(url_input.encode()).hexdigest()
                                    with st.spinner("Generating eval dataset (LLM)…"):
                                        saved = generate_eval_dataset(
                                            chunks,
                                            user_id=user_id,
                                            source_name=url_input[:80],
                                            doc_hash=url_hash,
                                        )
                                    if saved:
                                        st.info(f"🧪 {saved} synthetic eval queries saved.")
                            else:
                                st.error("No content found at URL.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()

        # Proactive tools
        st.markdown("### ⚡ Knowledge Tools")
        st.markdown('<span class="badge badge-ideas">Synthesis</span>', unsafe_allow_html=True)
        if st.button("📝 Synthesize & Summarize Ingested Knowledge", use_container_width=True):
            with st.spinner("Generating synthesis report…"):
                try:
                    ctx = retrieve_context(
                        "summarize key concepts definitions and topics",
                        user_id=user_id, k=8,
                    )
                    if not ctx.strip():
                        st.warning("Please ingest some documents first.")
                    else:
                        llm = get_llm()
                        guide = generate_study_guide(llm, ctx)
                        st.session_state["study_guide"] = guide
                        st.success("Synthesis report ready!")
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
        question = st.chat_input("Ask anything about your ingested data…")
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
                        # Retrieve using active pipeline
                        pipeline_res = retrieve_pipeline(
                            question, user_id, strategy=active_pipeline, k=5
                        )
                        context = "\n\n".join(c.text for c in pipeline_res.chunks)

                        # Generate answer
                        history_str = memory.get_history_string(limit=3)
                        llm = get_llm()
                        answer = generate_response(llm, context, question, history_string=history_str)
                        memory.add_interaction(question, answer)

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

                                st.markdown(
                                    f"**Retrieval latency:** `{pipeline_res.retrieval_latency_ms} ms`"
                                    + (f" | **Reranking:** `{pipeline_res.reranking_latency_ms} ms`"
                                       if pipeline_res.reranking_latency_ms else "")
                                )
                                for i, chunk in enumerate(pipeline_res.chunks, 1):
                                    st.markdown(
                                        f"**Chunk {i}** — `{chunk.source}` "
                                        f"| score `{chunk.score:.4f}`"
                                    )
                                    st.markdown(
                                        f"> {chunk.text[:400]}{'…' if len(chunk.text) > 400 else ''}"
                                    )

                    except Exception as e:
                        st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════
# TAB 2: RAG BENCHMARK
# ═════════════════════════════════════════════════════════════════
with main_tab2:
    st.markdown("## 📊 RAG Pipeline Benchmark")
    st.markdown("""
    <p style='color:#8A9B7D;font-size:0.92rem;'>
        Compare retrieval strategies head-to-head using your synthetic evaluation dataset.
        Metrics are computed against LLM-generated ground-truth query→chunk pairs.
        <strong>This is a synthetic evaluation</strong> — results indicate relative pipeline
        performance, not absolute accuracy against a human-labeled gold standard.
    </p>
    """, unsafe_allow_html=True)

    # ── Eval Dataset Status ──────────────────────────────────────
    stats = get_eval_dataset_stats(user_id)
    total_q = stats.get("total_queries", 0)
    unique_q = stats.get("unique_queries", 0)
    sources = stats.get("sources", {})

    bm_col1, bm_col2 = st.columns([1, 2], gap="large")

    with bm_col1:
        st.markdown("### 🧪 Evaluation Dataset Status")
        if total_q == 0:
            st.info(
                "**No synthetic eval queries found.**\n\n"
                "Go to **Chat & Ingest → Upload Document** and check "
                "\"Generate eval dataset after ingest\" to create one."
            )
        else:
            m1, m2 = st.columns(2)
            m1.metric("Total Query-Chunk Pairs", total_q)
            m2.metric("Unique Queries", unique_q)
            if sources:
                st.markdown("**Indexed source documents:**")
                for src, cnt in sources.items():
                    st.markdown(f"- `{src}` — {cnt} pairs")

        st.divider()

        st.markdown("### ⚙️ Benchmark Settings")
        k_val = st.slider("Top-K retrieved documents", min_value=1, max_value=10, value=5)
        selected_pipelines = st.multiselect(
            "Pipelines to evaluate:",
            PIPELINE_NAMES,
            default=PIPELINE_NAMES,
        )
        include_llm = st.checkbox(
            "Include LLM-judge metrics",
            value=False,
            help=(
                "Generates an answer and runs 3 LLM-judge evaluations per query per pipeline. "
                "Each benchmark call makes ~12+ LLM API calls. Enable only when needed."
            ),
        )

        run_btn = st.button(
            "▶ Run Benchmark",
            type="primary",
            use_container_width=True,
            disabled=(total_q == 0 or not selected_pipelines),
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

    # ── Run Benchmark ────────────────────────────────────────────
    if run_btn and total_q > 0 and selected_pipelines:
        progress_bar = st.progress(0, text="Initialising benchmark…")
        status_text = st.empty()
        total_steps = unique_q * len(selected_pipelines)

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
                )
                st.session_state["benchmark_report"] = report
                progress_bar.progress(100, text="✅ Benchmark complete")
                status_text.empty()
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    # ── Results Display ──────────────────────────────────────────
    report = st.session_state.get("benchmark_report")
    if report and report.total_queries > 0:
        st.divider()
        st.markdown(f"### 📈 Results — Top-{report.k} | {report.total_queries} queries | {report.total_duration_ms:,} ms total")

        # Summary table
        summary_rows = get_pipeline_summary_table(report)
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart section ────────────────────────────────────────
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

        # ── LLM Judge metrics (if run) ────────────────────────────
        if include_llm or any(row.get("Context Rel.") not in (None, "—") for row in summary_rows):
            st.divider()
            st.markdown("#### 🤖 LLM-Judge Generation Metrics")
            gen_cols = st.columns(len(selected_pipelines))
            for i, row in enumerate(summary_rows):
                with gen_cols[i]:
                    st.markdown(f"**{row['Pipeline']}**")
                    st.metric("Context Relevance", row.get("Context Rel.", "—"))
                    st.metric("Faithfulness", row.get("Faithfulness", "—"))
                    st.metric("Answer Relevance", row.get("Ans. Rel.", "—"))

        # ── Failure Analysis ─────────────────────────────────────
        failures = get_failure_analysis(report)
        if failures:
            st.divider()
            st.markdown("#### ⚠️ Failure Analysis — Queries with Recall@K = 0 across all pipelines")
            st.warning(
                f"{len(failures)} queries returned zero relevant results across all {len(selected_pipelines)} pipelines. "
                "These represent gaps in your ingested knowledge or mismatched chunk granularity."
            )
            fail_df = pd.DataFrame(failures)
            st.dataframe(fail_df, hide_index=True, use_container_width=True)

        # ── Per-query Inspector ───────────────────────────────────
        st.divider()
        st.markdown("### 🔎 Query Inspector")
        st.markdown(
            "<p style='color:#8A9B7D;font-size:0.88rem;'>Select a query to inspect "
            "retrieved chunks, ranks, and relevance status per pipeline.</p>",
            unsafe_allow_html=True,
        )

        all_queries = sorted(set(row["query"] for row in report.per_query))
        selected_q = st.selectbox("Select query:", all_queries, key="inspect_query")

        if selected_q:
            q_rows = [r for r in report.per_query if r["query"] == selected_q]

            # Expected chunks
            relevant_ids = q_rows[0]["relevant_chunk_ids"] if q_rows else []
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown("**Query details**")
                st.markdown(f"> {selected_q}")
                st.markdown(f"**Expected relevant chunk IDs:**")
                for cid in relevant_ids:
                    st.markdown(f"- `{cid}`")

                # Show rewritten query if Query Rewriting was selected
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

                    color = "#EEF4E6" if rec > 0 else "#FFF0F0"
                    st.markdown(
                        f"""<div style='background:{color};border:1px solid #E6E2D8;
                                        border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.75rem;'>
                            <strong>{p_name}</strong>
                            &nbsp; Recall@{report.k}: <code>{rec:.4f}</code>
                            &nbsp; MRR: <code>{mrr_val:.4f}</code>
                            &nbsp; P@{report.k}: <code>{prec:.4f}</code>
                            &nbsp; nDCG@{report.k}: <code>{ndcg:.4f}</code>
                            &nbsp; {pipeline_res.retrieval_latency_ms if False else row.get("retrieval_latency_ms", "?")}&nbsp;ms
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
                    if row.get("error"):
                        st.warning(f"⚠️ {row['error']}")
