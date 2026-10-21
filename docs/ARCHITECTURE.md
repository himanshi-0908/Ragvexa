# Ragvexa — Architecture, Flows and Roadmap

This document describes how Ragvexa is built today (after the security and
evaluation upgrade), how a request moves through it, what the data model
guarantees, and what the next phases are.

---

## 1. What Ragvexa is

Ragvexa is a **RAG quality and evaluation platform**, not just a chatbot.
Each user owns a private index, can chat over it with a chosen retrieval
strategy, and can measure retrieval and answer quality with repeatable
experiments. Everything is built on the existing stack: Streamlit, FastAPI,
ChromaDB, SQLite via SQLAlchemy, and NVIDIA NIM models.

Design principles:

1. **Tenant first.** No read or delete reaches a data store without a user id.
2. **No invented numbers.** A metric that cannot be computed is reported as
   unavailable, never estimated.
3. **Same conditions for every pipeline.** One dataset, one K, one prompt set,
   one generator, one judge per run.
4. **Every run is reproducible.** The configuration that produced a number is
   stored next to the number.

---

## 2. System architecture

```mermaid
flowchart TB
    subgraph Clients
        UI[Streamlit UI<br/>streamlit_app.py]
        API[FastAPI<br/>app.py + utils/routing.py]
    end

    subgraph Guard["Tenant & policy layer"]
        TF[tenant_filter<br/>retriver/vector_store.py]
        DOC[Ownership · quotas · usage<br/>utils/documents.py]
        SEC[Upload validation · SSRF-safe fetch<br/>utils/security.py]
    end

    subgraph Retrieval["Retrieval layer (retriver/retriver.py)"]
        N[Naive dense]
        H[Hybrid BM25 + RRF]
        R[Hybrid + cosine rerank]
        Q[Query rewriting]
    end

    subgraph LLM["LLM layer (llm/llm_handler.py)"]
        GEN[Answer model<br/>LLM_MODEL]
        EG[Eval generators<br/>EVAL_GENERATOR_MODELS]
        JUDGE[Judge model<br/>JUDGE_MODEL]
        TRACK[invoke_tracked<br/>tokens · latency · cost]
    end

    subgraph Eval["Evaluation layer"]
        DS[Dataset mgmt<br/>utils/eval_dataset.py]
        MET[Metrics<br/>utils/evaluator.py]
        BM[Benchmark runner<br/>utils/benchmark.py]
    end

    subgraph Storage
        SQL[(SQLite<br/>users · documents · chat_history<br/>synthetic_eval · benchmark_runs · usage_events)]
        VEC[(ChromaDB<br/>chunks tagged user_id · source · chunk_id)]
        FS[(private_uploads/&lt;user&gt;/<br/>0700, temp only)]
    end

    UI --> TF & DOC & SEC
    API --> TF & DOC & SEC
    TF --> Retrieval
    Retrieval --> VEC
    DOC --> SQL
    SEC --> FS
    UI --> GEN
    API --> GEN
    DS --> EG
    BM --> Retrieval
    BM --> MET
    MET --> GEN & JUDGE
    GEN & EG & JUDGE --> TRACK
    DS --> SQL
    BM --> SQL
```

### Component responsibilities

| Component | Responsibility | Must not |
|---|---|---|
| `streamlit_app.py` | Auth UI, ingest, chat, benchmark tabs, ground-truth review, run history | Touch a store without `user_id`; keep cross-user state in `st.session_state` |
| `utils/routing.py` | Same features over JWT for programmatic use | Reveal existence of another user's resources (returns 404) |
| `retriver/vector_store.py` | Chroma access, `tenant_filter`, scoped add/delete/count | Accept a missing or non-integer user id |
| `retriver/retriver.py` | The four pipelines, one `PipelineResult` schema | Query without `tenant_filter`; raise on failure (falls back and sets `error`) |
| `utils/documents.py` | Ownership rows, quotas, usage log, complete deletion | Delete across users; leave eval items or chunks behind |
| `utils/security.py` | SSRF guard on every redirect hop, upload validation, private temp storage | Follow redirects blindly; trust Content-Length |
| `llm/llm_handler.py` | Model factories, `invoke_tracked`, cost from configured prices only | Guess a price |
| `utils/eval_dataset.py` | Multi-model question generation, validation, dedup, human ground truth, review flags | Store a question that fails validation; mix users |
| `utils/evaluator.py` | Retrieval metrics, judge metrics, usage accounting | Substitute a fallback score when the judge fails |
| `utils/benchmark.py` | Run all pipelines under identical settings, classify failures, persist runs, export | Compare runs with different configs silently |

---

## 3. Request flows

### 3.1 Ingest a file

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit / API
    participant SEC as security.py
    participant DOC as documents.py
    participant VS as vector_store.py
    participant EG as eval_dataset.py

    U->>S: upload file (+ "generate eval dataset")
    S->>SEC: validate_upload(name, size, magic bytes)
    S->>DOC: content_hash → document_exists? (dedupe)
    S->>SEC: private_upload_path(user) → write 0600
    S->>S: load_document → chunk_documents
    S->>DOC: check_ingest_quota(user, n_chunks)
    S->>VS: add_to_vector_store(chunks, user_id, source)
    S->>DOC: register_document(user, source, hash, kind, n)
    S->>SEC: remove temp file
    opt eval dataset requested
        S->>EG: generate_eval_dataset(chunks, user, …)
        EG->>EG: for each model: Q/A pairs → validate → fuzzy dedup
        EG->>DOC: persist synthetic items (origin=synthetic, reviewed=false)
    end
```

### 3.2 Chat

```
question → check_daily_quota(user, "query")
         → retrieve_pipeline(question, user_id, strategy, k)   # tenant-scoped
         → generate_response_tracked(answer model, context, history)
         → memory.add_interaction + record_usage(tokens, cost)
         → answer + sources + latency/tokens/cost shown
```

### 3.3 Benchmark

```mermaid
flowchart LR
    A[load_eval_queries<br/>user · reviewed_only] --> B[dataset_fingerprint]
    B --> C{cached?}
    C -- no --> D[check_daily_quota benchmark]
    D --> E[for each query × pipeline<br/>retrieve_pipeline same k]
    E --> F[evaluate_query<br/>Recall P MRR nDCG]
    F --> G{judges on?}
    G -- yes --> H[answer with LLM_MODEL<br/>score with JUDGE_MODEL<br/>track tokens/cost]
    G -- no --> I
    H --> I[classify_failure<br/>missing_evidence · retrieval · ranking · generation · ok]
    I --> J[aggregate per pipeline<br/>+ config snapshot]
    J --> K[(benchmark_runs)]
    J --> L[UI: table · charts · limitations · inspector · JSON export]
```

Failure classification rule, per query and pipeline:

| Stage | Condition |
|---|---|
| `missing_evidence` | No labelled chunk id exists in the user's index |
| `retrieval` | Labelled chunks exist but are absent from a wide candidate pool (Naive at 4×K) |
| `ranking` | A labelled chunk is in the pool but outside top‑K |
| `generation` | Retrieval hit, but faithfulness, groundedness or correctness < 0.5 |
| `ok` | Retrieval hit and the answer (if judged) is acceptable |

### 3.4 Delete

```
delete_document(user, doc_id)
  → get_owned_document            # 404-style NotOwner if not yours
  → delete_user_source(user, src) # Chroma chunks
  → delete synthetic_eval rows for that doc_hash
  → delete documents row

delete_all_user_data(user)
  → clear_user_index + documents + synthetic_eval + chat_history
    + benchmark_runs + usage_events
```

---

## 4. Data model

```mermaid
erDiagram
    users ||--o{ documents : owns
    users ||--o{ chat_history : has
    users ||--o{ synthetic_eval : has
    users ||--o{ benchmark_runs : has
    users ||--o{ usage_events : has

    documents {
        int id PK
        int user_id FK
        string source
        string doc_hash
        string kind "file | url"
        int chunk_count
        int size_bytes
    }
    synthetic_eval {
        int id PK
        int user_id FK
        text query_text
        string chunk_id
        string source_doc
        string doc_hash
        string origin "synthetic | human"
        string generator_model
        text reference_answer
        bool reviewed
    }
    benchmark_runs {
        int id PK
        int user_id FK
        text config_json
        text summary_json
        text report_json
    }
    usage_events {
        int id PK
        int user_id FK
        string kind "query | benchmark | ingest"
        int input_tokens
        int output_tokens
        float cost_usd
    }
```

ChromaDB chunks carry `{user_id, source, chunk_id}` metadata. `chunk_id` is
a content hash, so the same text in two users' documents has the same id but
is still isolated by `user_id`.

---

## 5. What the numbers mean (and what validates them)

| Number | Depends on | Validated by |
|---|---|---|
| Recall@K, Precision@K, MRR, nDCG@K | Index, embedding model, chunking, **labels** | Unit tests prove the formulas; human review proves the labels |
| Faithfulness, groundedness, relevance, correctness | Generator model, judge model, prompts | Independent judge model; **judge calibration against human ratings (Phase 2)** |
| Latency | Hardware, network, model | Measured directly |
| Tokens | API response | Exact |
| Cost | Tokens × configured prices | Only shown when prices are configured |

Changing the generator (for example to another provider) legitimately changes
the generation numbers and leaves retrieval numbers untouched. The run config
records the model ids so runs are only compared like for like.

---

## 6. Roadmap

### Phase 1 — Secure multi-user foundation ✅ (current branch)

Tenant isolation, ownership, quotas, complete deletion, upload and SSRF
hardening, cross-model eval generation, independent judge, token and cost
tracking, failure stages, persisted reproducible runs, 147 offline tests.

### Phase 2 — Trust the numbers (next)

Goal: a reviewer can defend any number in a report.

| Item | Why | Done when |
|---|---|---|
| **Judge calibration** (in progress) | Judge scores are opinions until compared with humans | Agreement per metric shown; warning when low or sample < 5 |
| Human review workflow | Reviewing in a data editor does not scale | Queue of unreviewed items, keyboard accept/reject/edit, reviewer and timestamp stored |
| Confidence intervals | Averages over 20 queries hide variance | Bootstrap 95% CI per metric; "not significant" flag when intervals overlap |
| Dataset versioning | Editing labels silently changes old runs' meaning | Named, immutable dataset snapshots; runs reference a snapshot id |
| Run comparison view | Deciding needs A vs B, not two tables | Side-by-side diff of two runs with per-query deltas |
| Real-API smoke test | All LLM paths are mocked | Opt-in test that runs one judged query against NVIDIA and asserts tokens > 0 |

### Phase 3 — Scale and operate

| Item | Why | Done when |
|---|---|---|
| Postgres + pgvector or Chroma server | SQLite and local Chroma are single-node | `DATABASE_URL` and a vector-store URL switch backends; tests pass on both |
| Background jobs | Benchmarks and eval generation block the UI | Queue (RQ or Celery) with progress polling; cancel support |
| Audit log and admin role | No visibility into who did what | Append-only events for login, ingest, delete, benchmark; admin page |
| SSO / OAuth | Password-only auth | OIDC login alongside local accounts |
| Rate limiting in Streamlit | Only the API is IP-limited | Shared limiter keyed by user id |
| Observability | No metrics or tracing | OpenTelemetry spans per pipeline stage; dashboards for latency and cost |

### Phase 4 — Better retrieval and provider freedom

| Item | Why | Done when |
|---|---|---|
| Cross-encoder reranker | Cosine rerank is weak | `bge-reranker` or NIM reranker as pipeline 5, benchmarked against the four |
| Relevance filter pipeline (e.g. Jev or a small classifier) | Drop noisy chunks before generation | Pipeline with threshold/top-n settings; precision gain vs recall loss reported |
| Multi-query and HyDE | Single query embedding misses phrasing | Two more pipelines behind the same `PipelineResult` schema |
| Chunking experiments | Chunk size dominates retrieval quality | Benchmark can sweep chunk size and overlap on a snapshot |
| Provider-agnostic LLM layer | Users ask for Groq, OpenAI, local models | `LLM_PROVIDER` setting; same `invoke_tracked` contract; prices per provider |
| Per-pipeline cost/latency budget | Choose by value, not just score | Pareto view: quality vs latency vs cost |

### Phase 5 — Product surface

| Item | Why | Done when |
|---|---|---|
| Shareable report pages | Results live inside one account | Read-only report links with expiry |
| CI regression gate | RAG quality regresses silently | CLI `ragvexa bench --snapshot X --fail-below recall=0.7` for pipelines |
| Python SDK / OpenAPI client | API is hand-called | Generated client, examples in README |
| Dataset import/export formats | Teams have labels elsewhere | BEIR-style and CSV import; export of reviewed sets |

---

## 7. Decision log

| Decision | Reason |
|---|---|
| Keep local MiniLM embeddings | No document text leaves the machine for embedding; only generation and judging call NVIDIA |
| Separate judge model by default | A model grading its own answers is biased; the UI warns if they are set equal |
| Cost shown as `n/a` without prices | A wrong cost is worse than none |
| 404 for foreign documents | Prevents enumeration of other users' ids |
| Redirect validation hop by hop | The common SSRF bypass is a public URL that redirects to an internal one |
| Synthetic labels flagged, never hidden | Numbers on unreviewed labels are relative, and the report says so |
| bcrypt directly instead of passlib | passlib is incompatible with bcrypt 5 and broke API login |
