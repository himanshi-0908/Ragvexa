# 🌿 Ragvexa — RAG Quality & Evaluation Platform

> **Universal RAG. Any Source. Real Answers.**

Ragvexa is a secure, multi-user **Retrieval-Augmented Generation (RAG) engineering and evaluation platform**. It lets each user:

1. **Ingest** documents (PDF, TXT) and public web pages into a private, tenant-isolated vector index
2. **Chat** with their data using a configurable retrieval strategy
3. **Benchmark** four retrieval pipelines head-to-head on identical datasets and settings
4. **Evaluate** answers with an independent judge model, human-reviewed ground truth, token usage and cost
5. **Test with their own cases**: upload questions with gold evidence (document, pages, passage) and unanswerable questions, run every strategy on the same snapshot, and see the evidence behind every number with bootstrap confidence intervals
6. **Diagnose** failures by stage (missing evidence, retrieval, ranking, generation) and export reproducible reports

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
   │         test_sets · test_cases · test_case_evidence ·           │
   │         benchmark_runs · usage_events                           │
   │ ChromaDB: chunks tagged {user_id, source, chunk_id,             │
   │           document_id, page_start/end, char_start/end}          │
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

The judge is a **separate model** (`JUDGE_MODEL`) from the answer model (`LLM_MODEL`) so it is not grading its own output. The five judge questions per answer run concurrently, so a judged query costs about one judge call of wall-clock time. Defaults (verified 2026-10-22): answers from `nvidia/nemotron-3-ultra-550b-a55b` (fast, ~40 output tokens), judging by `nvidia/nemotron-3.5-lightning-30b-a3b` (a reasoning model: several seconds and a few hundred tokens per call). Set `JUDGE_MODEL` to the ultra model if you need speed over independence; the UI will warn that the judge is not independent. If you configure them to be the same, the UI and report say so. A judge failure yields "not scored", never a fallback number.

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

## User test sets and evidence-based evaluation

Synthetic questions are generated *from* a chunk, which makes retrieval artificially easy. The most credible ground truth is a set of questions you wrote yourself, each pointing at **where** the answer is in your documents. **Benchmark → Test Sets** takes such a file, validates it, and every strategy is then evaluated on the same cases with metrics that are computed by matching retrieved chunks to your evidence, never by chunk id.

### Test case format (CSV or JSON)

| Field | Required | Description |
|---|---|---|
| `question` | yes | The user question |
| `expected_answer` | no | Reference answer; enables correctness scoring by the judge |
| `answerable` | yes | `true` / `false`. `false` means the answer is **not** in your documents (tests abstention) |
| `gold_evidence` | if `answerable=true` | One or more items `{"document": "policy.pdf", "pages": [4], "passage": "exact or near-exact text"}` |
| `tags` | no | Free labels (`table`, `multi-hop`, …) for per-tag breakdowns |

```json
[
  {"question": "What is the refund window?",
   "expected_answer": "30 days from purchase.",
   "answerable": true,
   "gold_evidence": [{"document": "policy.pdf", "pages": [4],
                      "passage": "Refunds are accepted within 30 days of purchase."}],
   "tags": ["policy"]},
  {"question": "Who is the CEO?", "answerable": false}
]
```

In CSV, `gold_evidence` is a JSON string, or use the columns `evidence_document`, `evidence_pages`, `evidence_passage` for the single-evidence case. `pages` accepts `4`, `4-6` or `[4, 6]`; TXT and URL sources have no pages, so give a passage.

### Validation on upload

The file is parsed as untrusted input (size cap `MAX_TEST_SET_MB`, row cap `MAX_TEST_SET_ROWS`, length limits) and every row is checked before anything is saved: required fields, `answerable=true` must have evidence and `answerable=false` must have none, the `document` must be one of **your** documents (never another user's), a `passage` must actually appear in that document on the given pages (fuzzy token match against the indexed chunks), and identical questions are flagged as duplicates. The preview lists each row with its errors and warnings; you can fix the file or save only the valid rows.

### How retrieval is matched to gold evidence

Strategies use different chunk sizes, so a retrieved chunk is a **hit** for a gold item when:

1. it comes from the same document, and
2. its page range overlaps the gold pages (when both sides have pages), and
3. if the gold item has a passage: at least `overlap_threshold` (default 0.5, `EVIDENCE_OVERLAP_THRESHOLD`, adjustable per run) of the passage's normalised tokens are in the chunk, or the character ranges overlap by that fraction.

Without a passage the document/page match counts and is flagged as a **coarse match**. To make this possible, every chunk records its origin at ingest: `document_id`, `page_start`/`page_end` (PDF) and `char_start`/`char_end` (offsets in the whitespace-normalised document text).

### Metrics

| Group | Metrics | Over |
|---|---|---|
| Retrieval | Hit rate@K, Recall@K (gold items matched in top-K), MRR, nDCG@K | answerable cases |
| Abstention | Correct-abstention rate (the answer says the documents do not contain it), false-answer rate | unanswerable cases (needs generated answers) |
| Answer quality | Faithfulness, groundedness, relevance, correctness vs `expected_answer` | judged cases |

Every mean is reported with a **bootstrap 95% confidence interval** over test cases (percentile bootstrap, fixed seed). Below `MIN_CASES_FOR_CI` cases (default 10) the UI shows *not enough data* instead of an interval and no verdict should be drawn. When cases carry tags, the retrieval metrics are also broken down per tag.

Failure labels per case and strategy: `missing_evidence` (the gold document has no chunks in the index), `retrieval` (evidence indexed but not in the wider candidate pool), `ranking` (in the pool but outside top-K), `generation` (evidence retrieved but the judged answer scored below 0.5, or an unanswerable question was answered), `ok`.

### Evidence view

For every test case and strategy the results show the question, expected answer and answerable flag; the retrieved chunks in rank order with source, page, snippet and score, each marked hit or miss with the gold item it matched and why; the gold evidence text itself; the generated answer with the judge verdicts; and the failure label. The aggregate table of strategies × metrics (with intervals) sits above it, with filters for failed cases only, by tag and by strategy, and a formula-safe CSV download.

### Snapshots and reproducibility

When a run starts, the test set is snapshotted into an immutable JSON copy that is stored with the run configuration (`test_set_snapshot`, plus its fingerprint). All strategies iterate over that same copy, and reloading the run from **Run History** shows exactly the cases it was evaluated on even if the test set or documents changed later. Deleting a document removes the test cases (and their evidence) that cite it; "Delete all my data" removes test sets, cases and evidence.

### Acceptance checklist

- [x] User can upload a valid test set; invalid rows give clear errors (per-row messages in the preview; `tests/test_test_sets.py::TestValidation`).
- [x] All strategies run on the same snapshot and are compared with identical metrics (`TestBenchmarkOnTestSet::test_all_strategies_see_identical_cases_and_snapshot_is_immutable`).
- [x] Recall/MRR/nDCG are computed from evidence matching, not chunk IDs (`utils/evaluator.match_retrieved_chunks`; `TestMatching`, `TestMetrics`).
- [x] Every score can be traced to retrieved chunks and gold evidence in the UI (Evidence View; `get_evidence_view` tested in `test_metrics_failure_labels_and_evidence_view`).
- [x] Unanswerable cases are supported and reported separately (abstention rates; excluded from retrieval metrics).
- [x] Confidence intervals shown; "not enough data" shown for tiny test sets (`bootstrap_ci`, `format_ci`, Limitations box).
- [x] Cross-user access is impossible (tested: foreign documents rejected on upload, test sets and runs scoped by `user_id`); tests pass offline.

---

## Bring your own API key, and automatic model fallback

Two ways to run the LLM side:

| Mode | Who pays | How |
|---|---|---|
| **Demo key** (default) | You, the operator | `NVIDIA_API_KEY` in `.env`. Testers just sign up and use the app. |
| **Own key** | The user | Sidebar → **API key & models** → "My own key". The key is validated against the NVIDIA models endpoint, kept only in that browser session, and never stored or logged. Their document text then goes to NVIDIA under their own account. |

Per-user daily quotas apply in both modes, so nobody can use the server as a free proxy.

**Automatic model fallback.** If the active key cannot use a configured model (retired, not enabled for the account, or forbidden), the app does not fail. It probes a short list of known model ids once per key (cached by key fingerprint for an hour), switches to the first one that answers, and tells you:

- chat shows a warning naming the model that was used instead;
- the benchmark run config records `model_switches` and the Limitations box repeats it;
- the API response carries `model` and `model_switched_from`.

To make a choice permanent, open **Models** in the sidebar, click "Check which models work with this key", and pick the answer, judge and eval-generator models from the ones that answered. Reasoning models (slower, token-heavy) are offered last for answers. The same probe is available from the command line as `python scripts/check_models.py --probe`.

API users can pass their own key per request:

```bash
curl -s -X POST localhost:8000/query -H "Authorization: Bearer TOKEN" -H "X-NVIDIA-API-Key: nvapi-..." -H "Content-Type: application/json" -d '{"question": "..."}'
```

The header is used for that request only. Key problems come back as HTTP 400 with the reason, never as a 500.

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
| Quotas | Per-user caps on documents, total chunks, queries per day, benchmarks per day (server-side, in `utils/documents.py`) and on test sets (file size, rows, sets per user). FastAPI also rate-limits by IP. |
| External data disclosure | Document text, questions and answers are sent to the NVIDIA API. A notice is shown in the sidebar and above ingestion. Embeddings are computed locally. |
| User API keys | Kept in Streamlit session state only (purged on user change or sign-out); API header keys are per request. Cache and reports hold a SHA-256 fingerprint or the last four characters, never the key. |
| Complete deletion | Per-document delete (chunks, eval items, test cases citing it) and "Delete all my data" (index, documents, eval items, test sets, chats, runs, usage). |
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

Fill in `NVIDIA_API_KEY`, `COOKIE_SECRET`, `JWT_SECRET`. Then confirm your key can invoke the configured models, because NVIDIA retires models (HTTP 410) and not every key is entitled to every model (HTTP 404 "not found for account"):

```bash
python scripts/check_models.py --probe
```

It prints one line per configured model with the HTTP status and, if any fail, lists ids that work for your key. Put those in `.env`. Optional: `LLM_MODEL`, `JUDGE_MODEL`, `EVAL_GENERATOR_MODELS`, quotas, and `MODEL_PRICES_JSON` for cost estimates. See `.env.example` for every setting.

### 3. Run

```bash
streamlit run streamlit_app.py          # UI on :8501
uvicorn app:app --reload                # optional API on :8000
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Chat shows `[403] Forbidden Authorization failed`, `[404] ... Not found for account` or `[410]` | The model id is retired or not enabled for your key | `python scripts/check_models.py --probe`, then set working ids in `.env` |
| Tokens column is 0 after a judged benchmark | The model did not return usage metadata | Check `scripts/check_models.py` output shows token counts for that model |
| `COOKIE_SECRET is not set` on startup | `.env` missing or not in the working directory | `cp .env.example .env` and fill it in |

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
| `test_api_keys.py` | Key validation and fingerprinting, probe cache, auto model switch (404/410/403, not 401), benchmark records switches without the key, per-request API header |
| `test_evaluation.py` | Cross-model generation + dedup, ground-truth import, usage/cost tracking, judge independence, failure classification, benchmark runner |
| `test_test_sets.py` | Chunk origin at ingest, test-set upload validation (schema, evidence rules, foreign documents, size/row caps, duplicates), evidence matching across chunk sizes, hand-computed metrics, abstention, bootstrap CI, identical snapshot per strategy, cascade deletion, evidence view data, cross-user isolation |

All tests run offline: the vector store, HTTP and LLM calls are mocked.

---

## Reproducible experiment workflow

1. **Ingest** a document with "Generate eval dataset after ingest" (uses every model in `EVAL_GENERATOR_MODELS`).
2. **Review**: Benchmark → Ground Truth. Tick items you have verified, delete bad ones, or import a JSON ground-truth file. Or upload your own test cases under Benchmark → Test Sets.
3. **Run**: choose the dataset (eval queries or a test set), K, pipelines, "Human-reviewed items only" and optionally LLM-judge metrics. The same settings apply to every pipeline.
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
│   ├── llm_handler.py      # NvidiaChat wrapper: per-user key, auto model fallback, invoke_tracked
│   └── model_registry.py   # Key validation, model probing/cache, suggestions
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
│   ├── security.py         # SSRF guard, upload validation, private storage
│   └── test_sets.py        # User test sets: CSV/JSON import, validation, snapshots, cascade delete
├── tests/
├── scripts/
│   └── check_models.py     # Verify the NVIDIA key can invoke configured models
├── streamlit_app.py        # Main UI
├── app.py                  # FastAPI entry point
└── config.py               # Environment config
```
