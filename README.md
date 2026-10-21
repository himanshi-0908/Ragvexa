# 🌿 Ragvexa — RAG Quality & Evaluation Platform

> **Universal RAG. Any Source. Real Answers.**

Ragvexa is a secure, multi-user **Retrieval-Augmented Generation (RAG) engineering and evaluation platform**. It lets each user:

1. **Ingest** documents (PDF, TXT) and public web pages into a private, tenant-isolated vector index
2. **Chat** with their data using a configurable retrieval strategy
3. **Benchmark** four retrieval pipelines head-to-head on identical datasets and settings
4. **Evaluate** answers with an independent judge model, human-reviewed ground truth, token usage and cost
5. **Diagnose** failures by stage (missing evidence, retrieval, ranking, generation) and export reproducible reports

---

## Architecture

> Full architecture, request flows, data model and the phased roadmap: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

```
                        ┌──────────────────────────────────────────┐
                        │               Streamlit UI               │
                        │  Chat/Ingest │ Benchmark │ Ground Truth  │
                        └──────┬───────────────┬──────────┬────────┘
                               │               │          │
   FastAPI (optional, JWT) ────┤               │          │
                               ▼               ▼          ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  Tenant guard: every vector / DB access is scoped by user_id   │
   │  (retriver.vector_store.tenant_filter, utils.documents)        │
   ├────────────────────────────────────────────────────────────────┤
   │  Retrieval Layer:  Naive │ Hybrid/RRF │ Hybrid+Reranker │ QR   │
   │  ChromaDB (local MiniLM embeddings) · BM25 · cosine rerank     │
   ├────────────────────────────────────────────────────────────────┤
   │  LLM Layer (NVIDIA NIM): answer model │ eval generators │ judge │
   │  llm.llm_handler.invoke_tracked → tokens, latency, cost         │
   ├────────────────────────────────────────────────────────────────┤
   │  Evaluation Layer: Recall@K · Precision@K · MRR · nDCG@K        │
   │  Judge: context relevance · faithfulness · groundedness ·       │
   │         answer relevance · correctness (vs reference answer)    │
   │  Failure stage classifier · reproducible BenchmarkRun records   │
   └────────────────────────────────────────────────────────────────┘
                                   │
   ┌───────────────────────────────▼────────────────────────────────┐
   │ SQLite: users · documents (ownership) · chat_history ·          │
   │         synthetic_eval (origin/reviewed/reference) ·            │
   │         benchmark_runs · usage_events                           │
   │ ChromaDB: chunks tagged {user_id, source, chunk_id}             │
   │ private_uploads/<user_id>/ (0700, deleted after parsing)        │
   └────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Strategies

| Pipeline | Description | Strengths | Weaknesses |
|---|---|---|---|
| **Naive Dense** | Chroma similarity on local MiniLM embeddings | Fast, simple, semantic | Misses keyword-only matches |
| **Hybrid + RRF** | Dense + BM25 fused with Reciprocal Rank Fusion (k_rrf=60) | Best of both worlds | Higher latency |
| **Hybrid + Reranker** | Hybrid/RRF candidates → cosine reranker sorts final top-k | Highest relevance precision | Extra embedding call |
| **Query Rewriting** | LLM rewrites the query → Naive retrieval | Better semantic coverage | LLM call adds latency and cost |

All four pipelines return the same `PipelineResult` schema and are benchmarked with the **same dataset, same K, same prompts, same generator and judge models**.

### Reciprocal Rank Fusion (RRF)

```
score(d) = Σ  1 / (k_rrf + rank_i(d))
           i
```

---

## Evaluation

### Retrieval metrics (need labelled relevant chunk IDs)

| Metric | Formula | Interpretation |
|---|---|---|
| **Recall@K** | `|relevant ∩ retrieved[:K]| / |relevant|` | How many relevant chunks were found |
| **Precision@K** | `|relevant ∩ retrieved[:K]| / K` | Fraction of retrieved that are relevant |
| **MRR** | `1 / rank_of_first_relevant` | How highly ranked the first relevant result is |
| **nDCG@K** | DCG / ideal DCG (binary relevance) | Reward for highly-ranked relevant results |

### Generation metrics (LLM-as-a-judge, optional)

| Metric | Measures |
|---|---|
| **Context Relevance** | Is the retrieved context on-topic for the query? |
| **Faithfulness** | Does the answer avoid contradicting or inventing beyond the context? |
| **Groundedness** | What fraction of the answer's claims trace to a specific context sentence? |
| **Answer Relevance** | Does the answer address the query? |
| **Correctness** | Does the answer agree with the reference answer? (only when a reference exists) |

The judge is a **separate model** (`JUDGE_MODEL`) from the answer model (`LLM_MODEL`) so it is not grading its own output. If you configure them to be the same, the UI and report say so. A judge failure yields "not scored", never a fallback number.

### Latency, tokens and cost

Every LLM call is wrapped by `invoke_tracked`, which records input/output tokens and latency from the API response. Cost is computed **only** when `MODEL_PRICES_JSON` has an entry for that model; otherwise it is reported as `n/a`. Nothing is estimated from guesses.

### Datasets: synthetic vs human-reviewed

* **Synthetic**: at ingest time, each chunk is sent to *every* model in `EVAL_GENERATOR_MODELS`. Each writes questions plus a reference answer. Questions are validated (length, question form, no "this passage" meta-questions) and de-duplicated across models (exact and fuzzy). The generator model is stored with each item.
* **Human**: in **Benchmark → Ground Truth** you can mark synthetic items as *reviewed*, delete bad ones, or import a JSON file of human labels:

```json
[
  {"query": "What is precision at K?",
   "reference_answer": "The fraction of retrieved items that are relevant.",
   "relevant_passages": ["fraction of retrieved items that are relevant"],
   "relevant_chunk_ids": ["optional explicit ids"]}
]
```

Passages are matched against your indexed chunks. A benchmark can be restricted to **reviewed items only**.

### Failure analysis

Each query × pipeline is classified into one stage:

| Stage | Meaning |
|---|---|
| `missing_evidence` | None of the labelled chunks exist in the index any more (deleted document, re-chunked) |
| `retrieval` | Labelled chunks exist but were not in a wider candidate pool (Naive at 4×K) |
| `ranking` | A labelled chunk was in the candidate pool but outside top-K |
| `generation` | Retrieval succeeded but the judged answer scored below 0.5 on faithfulness, groundedness or correctness |
| `ok` | Retrieval (and the answer, if judged) succeeded |

The Query Inspector shows per-pipeline ranks, scores, stage, generated answer, reference answer and judge scores.

### Reproducible reports

Every run stores its configuration (K, pipelines, dataset fingerprint, dataset size and how many items are human/reviewed, generator and judge model ids, embedding model, whether prices were configured, git revision, timestamps) together with per-pipeline aggregates and per-query detail in the `benchmark_runs` table. Runs can be reloaded from **Run History** or downloaded as JSON.

### Limitations (read before quoting numbers)

* Synthetic labels are written by LLMs and are **not** human ground truth. Numbers against them show *relative* differences between pipelines. The report lists how many queries are unreviewed and warns when fewer than 20 queries were used.
* Judge scores are LLM opinions on a 0–1 scale and can be biased; they are not a substitute for human review.
* Cost is unavailable unless you configure prices. Token counts come from the API and are exact.
* Benchmarks use the live index at run time: deleting a document after generating its eval items produces `missing_evidence` failures.

---

## Security model

| Concern | Implementation |
|---|---|
| Authentication | Streamlit: streamlit-authenticator with a signed cookie (`COOKIE_SECRET`, expiry `COOKIE_EXPIRY_DAYS`). FastAPI: JWT bearer tokens (`JWT_SECRET`); the user must still exist. Passwords are bcrypt-hashed on both sides. |
| Tenant isolation | Every ChromaDB read/delete goes through `tenant_filter(user_id)`, which refuses missing or invalid ids. Chat history, eval items, documents, benchmark runs and usage are all keyed by `user_id`. |
| Ownership | The `documents` table is the authoritative list of what a user owns. Deleting a document removes its chunks, eval items and ownership row. Deleting another user's document returns 404 (existence is not revealed). |
| Session handling | Session-state caches (benchmark reports, study guide, inspector state) are purged whenever the signed-in user changes in a browser tab, so one account can never see another's cached results. |
| Upload security | Extension whitelist, size cap (`MAX_UPLOAD_SIZE_MB`, enforced while streaming), magic-byte check, path-traversal-safe random filenames, owner-only (`0700`/`0600`) per-user temp directory, file deleted after parsing. Duplicates are detected by content hash. |
| URL ingestion (SSRF) | The URL and **every redirect hop** are resolved and checked against loopback, private, link-local, multicast, reserved and `.localhost` ranges; literal IPs are judged directly. Only text/HTML content types are accepted and the body is capped at `MAX_URL_FETCH_BYTES`. |
| Quotas | Per-user caps on documents, total chunks, queries per day and benchmarks per day (server-side, in `utils/documents.py`). FastAPI also rate-limits by IP. |
| External data disclosure | Document text, questions and answers are sent to the NVIDIA API. A notice is shown in the sidebar and above ingestion. Embeddings are computed locally. |
| Complete deletion | Per-document delete and "Delete all my data" (index, documents, eval items, chats, runs, usage). |
| Logging | Logs record user ids and counts, never document text, passwords or API keys. |

Known gaps: DNS rebinding between the check and the fetch is not prevented; there is no admin role or audit log; SQLite is single-node.

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd Ragvexa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # run twice: COOKIE_SECRET, JWT_SECRET
```

Fill in `NVIDIA_API_KEY`, `COOKIE_SECRET`, `JWT_SECRET`. Optional: `LLM_MODEL`, `JUDGE_MODEL`, `EVAL_GENERATOR_MODELS`, quotas, and `MODEL_PRICES_JSON` for cost estimates. See `.env.example` for every setting.

### 3. Run

```bash
streamlit run streamlit_app.py          # UI on :8501
uvicorn app:app --reload                # optional API on :8000
```

---

## Running tests

```bash
python -m pytest tests/ -q
```

| Test file | Coverage |
|---|---|
| `test_metrics.py` | Recall@K, Precision@K, MRR, nDCG@K |
| `test_rrf.py` | RRF formula, cosine similarity |
| `test_pipelines.py` | All 4 pipelines, result schema, fallbacks |
| `test_tenant_isolation.py` | Tenant filter on every pipeline, scoped deletion, chat/eval/run isolation, cache keys |
| `test_authorization.py` | FastAPI auth, ownership (404 on foreign documents), quotas, upload validation, SSRF via API |
| `test_ingestion_security.py` | SSRF guard incl. redirects and DNS-to-private, content type and size caps, upload validation, private storage permissions |
| `test_evaluation.py` | Cross-model generation + dedup, ground-truth import, usage/cost tracking, judge independence, failure classification, benchmark runner |

All tests run offline: the vector store, HTTP and LLM calls are mocked.

---

## Reproducible experiment workflow

1. **Ingest** a document with "Generate eval dataset after ingest" (uses every model in `EVAL_GENERATOR_MODELS`).
2. **Review**: Benchmark → Ground Truth. Tick items you have verified, delete bad ones, or import a JSON ground-truth file.
3. **Run**: choose K, pipelines, "Human-reviewed items only" and optionally LLM-judge metrics. The same settings apply to every pipeline.
4. **Read the Limitations box** above the results table before interpreting numbers.
5. **Diagnose** with the failure breakdown and Query Inspector.
6. **Save/share**: download the JSON report; the run is also stored under Run History with its configuration and dataset fingerprint so it can be repeated.

---

## Project structure

```
Ragvexa/
├── ingestion/
│   ├── pdf_loader.py       # PDF → Document
│   ├── text_loader.py      # PDF/TXT dispatch
│   └── web_loader.py       # URL → Document via SSRF-safe fetch
├── retriver/
│   ├── retriver.py         # 4 pipelines + PipelineResult schema (tenant-scoped)
│   └── vector_store.py     # ChromaDB helpers, tenant_filter, scoped delete
├── llm/
│   └── llm_handler.py      # NVIDIA models, invoke_tracked (tokens/cost), judge
├── utils/
│   ├── auth_helper.py      # Registration + credential loading (Streamlit)
│   ├── benchmark.py        # Runner, failure classifier, persisted runs, export
│   ├── chunking.py         # RecursiveCharacterTextSplitter
│   ├── database.py         # SQLAlchemy models + SQLite column migration
│   ├── documents.py        # Ownership, quotas, usage, complete deletion
│   ├── eval_dataset.py     # Cross-model generation, validation, ground truth
│   ├── evaluator.py        # Retrieval metrics + independent judge metrics
│   ├── memory.py           # Per-user chat history
│   ├── rate_limit.py       # API rate limiting
│   ├── routing.py          # FastAPI routes (JWT, ownership, quotas)
│   └── security.py         # SSRF guard, upload validation, private storage
├── tests/
├── streamlit_app.py        # Main UI
├── app.py                  # FastAPI entry point
└── config.py               # Environment config
```
