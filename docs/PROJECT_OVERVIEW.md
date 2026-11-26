# Ragvexa — Project Overview (presentation and interview guide)

> **One line:** Ragvexa is a secure, multi-user platform for building *and measuring* Retrieval-Augmented Generation (RAG) systems: you upload documents, chat over them with a chosen retrieval strategy, and benchmark four strategies head-to-head on the same test cases with metrics that are traceable to the exact evidence behind every number.

Repository: https://github.com/himanshi-0908/Ragvexa (branch `claude/intelligent-zhukovsky-a8cda3` holds the evidence-based evaluation work on top of `feat/secure-multi-user-eval-platform`).

---

## 1. The 60-second pitch

Most RAG demos stop at "upload a PDF and ask a question". The hard part is knowing **whether the answer is actually right and why**. Ragvexa treats RAG as an engineering problem to be measured:

- **Four retrieval strategies** (dense, hybrid with BM25 + Reciprocal Rank Fusion, hybrid + reranker, LLM query rewriting) share one result schema and are always run under identical conditions.
- **Three kinds of ground truth**: synthetic questions written by several LLMs, human-reviewed labels, and the most credible one, **user-written test cases with gold evidence** (document, pages, passage) plus unanswerable questions that test whether the system abstains.
- **Metrics with evidence**: Hit rate, Recall@K, MRR, nDCG@K computed by matching retrieved chunks to gold evidence (never by chunk id, so strategies with different chunk sizes are comparable), abstention rates, LLM-judge scores from an *independent* judge model, bootstrap 95% confidence intervals, per-tag breakdowns, and a per-stage failure label (missing evidence / retrieval / ranking / generation).
- **Evidence view**: for every question and strategy you can see the retrieved chunks in rank order, which gold item each one matched and why, the gold passage, the generated answer and the judge's verdicts.
- **Production concerns**: real authentication, strict per-user tenant isolation on every data path, server-side quotas, SSRF-safe URL ingestion, upload validation, reproducible saved runs, and 212 offline tests.

---

## 2. The problem it solves

| Pain in typical RAG projects | What Ragvexa does about it |
|---|---|
| "It works on my three questions" | Benchmarks all strategies on the same dataset with the same K, prompts and models, and stores the run so it can be repeated |
| Synthetic eval questions are generated from a chunk, so retrieval looks artificially good | Lets users upload their own questions with the *location* of the answer; metrics come from evidence matching |
| Metrics keyed on chunk ids break when chunking changes | Matches by document + page overlap + passage token overlap, so a 200-token chunker and a 1000-token chunker are scored on equal terms |
| Scores without context are not actionable | Every score is traceable to the chunks that produced it; failures are labelled by stage |
| A model grading its own answers is biased | The judge is a separate model by default; the UI and report warn if it is not |
| Small test sets produce confident-looking noise | Bootstrap intervals, and an explicit "not enough data" below 10 cases |
| Multi-user demos leak data between accounts | Every read and delete goes through a tenant filter that refuses a missing user id; tested |

---

## 3. What a user does (the flow)

1. **Sign in** (streamlit-authenticator with a signed cookie; the FastAPI side uses JWT). Passwords are bcrypt-hashed.
2. **Ingest** a PDF/TXT or a public URL. Text is chunked (recursive splitter, 1000 chars with 200 overlap), embedded **locally** with MiniLM and stored in ChromaDB tagged with the owner's user id, the document id, page range and character offsets.
3. **Chat** over the private index with the chosen strategy; the answer shows retrieved chunks, latency, tokens and cost.
4. **Build ground truth**: generate synthetic questions with several LLMs at ingest time, review them, import human labels, or upload a **test set** (CSV/JSON) of questions with gold evidence and unanswerable questions. Uploads are validated row by row before anything is saved.
5. **Run a benchmark**: choose the dataset, K, strategies, the evidence overlap threshold and whether to generate answers and judge them. The test set is snapshotted so every strategy sees identical cases.
6. **Read the results**: strategies × metrics with confidence intervals, abstention rates, tag breakdown, failure breakdown, and the evidence view with filters (failed only, by tag, by strategy). Download the JSON report or a formula-safe CSV. Reload any past run from Run History.

---

## 4. Architecture

```
                 Streamlit UI (chat, ingest, benchmark, ground truth, test sets, run history)
                 FastAPI (optional JSON API with JWT, same users table)
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │  Tenant guard: tenant_filter(user_id) on every vector    │
        │  read/delete; all SQL filtered by user_id; ownership     │
        │  table for documents (404 for foreign ids)               │
        ├─────────────────────────────────────────────────────────┤
        │  Retrieval: Naive │ Hybrid (BM25+RRF) │ Hybrid+Reranker  │
        │             │ Query Rewriting   → PipelineResult schema  │
        ├─────────────────────────────────────────────────────────┤
        │  LLM layer (NVIDIA NIM): answer model, eval generators,  │
        │  independent judge; invoke_tracked records tokens,       │
        │  latency, cost; automatic model fallback per API key     │
        ├─────────────────────────────────────────────────────────┤
        │  Evaluation: chunk-id metrics for eval queries;          │
        │  evidence matching + metrics for test sets; judge        │
        │  metrics; bootstrap CIs; failure classifier; snapshots   │
        └─────────────────────────────────────────────────────────┘
          SQLite (users, documents, chat_history, synthetic_eval,
                  test_sets, test_cases, test_case_evidence,
                  benchmark_runs, usage_events)
          ChromaDB (chunks: user_id, source, chunk_id, document_id,
                    page_start/end, char_start/end)
```

**Stack:** Python 3.11, Streamlit, FastAPI, LangChain, ChromaDB, sentence-transformers (all-MiniLM-L6-v2, local), rank_bm25, SQLAlchemy + SQLite, NVIDIA NIM (LangChain NVIDIA endpoints), bcrypt, PyJWT, slowapi, pytest.

**Size:** about 6,600 lines of application code and 2,200 lines of tests (212 tests, all offline).

---

## 5. Retrieval strategies

| Strategy | How it works | Trade-off |
|---|---|---|
| Naive dense | Chroma cosine/L2 similarity on MiniLM embeddings | Fast and simple; misses keyword-only matches |
| Hybrid | Dense results + BM25 keyword results fused with Reciprocal Rank Fusion, `score(d) = Σ 1/(k_rrf + rank_i(d))`, k_rrf = 60 | Covers both semantic and lexical matches; higher latency |
| Hybrid + Reranker | Hybrid candidates re-scored by cosine similarity between query and chunk embeddings | Best precision at the top; extra embedding call |
| Query Rewriting | An LLM rewrites the question for retrieval, then dense retrieval | Helps vague queries; adds an LLM call and cost; falls back to the original query if the LLM fails |

All four return the same `PipelineResult` (ranked `RetrievedChunk`s with chunk id, source, text, score, rank, and the chunk's origin fields), so the evaluator and the UI never special-case a strategy.

---

## 6. Evaluation methodology (the core of the project)

### 6.1 Ground truth, from weakest to strongest

1. **Synthetic** — at ingest, each chunk is sent to every configured generator model; each writes questions plus a reference answer. Questions are validated (length, question form, no "this passage" meta-questions) and de-duplicated across models (exact and fuzzy). The generator model is stored with each item. The report always says how many labels are unreviewed.
2. **Human-reviewed** — users tick items they verified, delete bad ones, or import JSON labels matched to indexed passages.
3. **User test sets with gold evidence** — the user writes the question, the expected answer, and *where* the answer lives: `{"document": "policy.pdf", "pages": [4], "passage": "Refunds are accepted within 30 days of purchase."}`. `answerable=false` rows carry no evidence and test abstention.

### 6.2 Chunk origin (prerequisite for evidence matching)

Every chunk records `document_id`, `page_start`/`page_end` (PDF) and `char_start`/`char_end` (offsets in the whitespace-normalised text of the whole document). Offsets are found by locating the normalised chunk text in the normalised document text, scanning forward from the previous chunk so overlapping chunks resolve correctly.

### 6.3 Evidence matching (never chunk ids)

A retrieved chunk is a **hit** for a gold item when:

1. it comes from the same document, and
2. its page range overlaps the gold pages (when both have pages), and
3. if the gold item has a passage, at least `overlap_threshold` (default 0.5) of the passage's normalised tokens appear in the chunk, or the character ranges overlap by that fraction.

Without a passage, the page/document match counts and is flagged as a **coarse match** in the report. Chunks indexed before origin tracking fall back to a source-name comparison.

### 6.4 Metrics

| Metric | Definition | Computed over |
|---|---|---|
| Hit rate@K | 1 if any gold item is matched in the top K | answerable cases |
| Recall@K | matched gold items / all gold items | answerable cases |
| Precision@K | hits in top K / K | answerable cases |
| MRR | 1 / rank of the first hit | answerable cases |
| nDCG@K | DCG / ideal DCG with binary gains; a chunk only gains when it is the first to cover a not-yet-covered gold item, so small-chunk strategies cannot inflate the score by returning the same passage twice | answerable cases |
| Correct-abstention rate | share of unanswerable cases where the answer says the documents do not contain it | unanswerable cases (needs generated answers) |
| False-answer rate | 1 − correct-abstention rate | unanswerable cases |
| Context relevance, faithfulness, groundedness, answer relevance, correctness | 0–1 scores from an independent judge model with structured prompts; a judge failure yields "not scored", never a fallback number | judged cases |

Every mean carries a **bootstrap 95% confidence interval** (percentile bootstrap, 1000 resamples, fixed seed for determinism). Below `MIN_CASES_FOR_CI` (10) cases the UI shows "not enough data" instead of an interval. Metrics are also broken down per tag.

### 6.5 Failure labels (deterministic rules)

| Label | Meaning |
|---|---|
| `missing_evidence` | the gold document has no chunks in the index (deleted or re-chunked) |
| `retrieval` | evidence is indexed but not even in a wider candidate pool (Naive at 4×K) |
| `ranking` | evidence is in the candidate pool but outside the top K |
| `generation` | evidence was retrieved but the judged answer scored below 0.5, or an unanswerable question was answered |
| `ok` | everything else |

### 6.6 Reproducibility

Every run stores its full configuration: K, strategies, dataset fingerprint, dataset composition, generator and judge model ids, whether the judge was independent, model switches that happened, embedding model, git revision, timestamps, and for test-set runs the **immutable snapshot of the cases**. Reloading a run shows exactly the cases it was evaluated on even if the test set or documents changed later.

---

## 7. Security and multi-tenancy

| Concern | Implementation |
|---|---|
| Authentication | streamlit-authenticator with a signed cookie; JWT bearer tokens for the API (the user must still exist). Passwords bcrypt-hashed |
| Tenant isolation | `tenant_filter(user_id)` builds the Chroma `where` clause for every read/delete and raises on a missing or invalid id; every SQL query filters by user id; tested for every pipeline |
| Ownership | The `documents` table is the authoritative list of what a user owns; deleting another user's document returns 404 so ids cannot be enumerated |
| Deletion | Deleting a document removes its chunks, eval items and the test cases citing it; "Delete all my data" removes everything the user owns |
| Uploads | Extension whitelist, streaming size cap, magic-byte check, random filenames in a per-user `0700` directory, file deleted after parsing, duplicates by content hash |
| URL ingestion (SSRF) | Each redirect hop is resolved and checked against loopback, private, link-local, multicast and reserved ranges; only text/HTML; body size capped |
| Test-set uploads | Treated as untrusted: size and row caps, length limits, safe parsing, formula-injection-safe CSV export (`=`, `+`, `-`, `@` prefixed) |
| Quotas and rate limits | Server-side caps on documents, chunks, queries/day, benchmarks/day, test sets; API rate limiting by IP |
| Keys and logs | Users may bring their own NVIDIA key: kept in session state only, never stored or logged; reports hold only a fingerprint. Logs record ids and counts, never document text |
| Session hygiene | Cached reports are purged whenever the signed-in user changes in a browser tab |

Known gaps (say these before anyone asks): DNS rebinding between check and fetch is not prevented; no admin role or audit log; SQLite is single-node; judge scores are LLM opinions, not human truth.

---

## 8. Engineering decisions worth explaining

| Decision | Why |
|---|---|
| Local MiniLM embeddings | Document text never leaves the machine for embedding; only generation and judging call NVIDIA |
| Separate judge model by default | A model grading its own output is biased; the UI warns when judge equals generator |
| Cost reported as `n/a` unless prices are configured | A wrong number is worse than none; token counts come from the API and are exact |
| Register the document row *before* indexing | The document id must be stored on every chunk; if indexing fails the row is removed |
| Null metadata omitted from Chroma | Chroma rejects `None` values, so TXT/URL chunks simply have no page keys |
| Matching by location and text, not chunk id | Chunk ids change with chunking; evidence does not |
| nDCG credits the first chunk per gold item | Prevents small-chunk strategies from inflating scores with duplicates |
| Snapshot the test set into the run | Guarantees all strategies see identical cases and keeps old runs readable |
| Deterministic bootstrap (fixed seed) | The same run always reports the same interval |
| Streamlit session cache keyed by user + fingerprint | Avoids recomputation without leaking across users |

---

## 9. Testing

212 pytest tests run offline in about 5 seconds; the vector store, HTTP and LLM calls are mocked and the embedding model is never downloaded in tests.

| Area | Examples |
|---|---|
| Metrics | Recall/Precision/MRR/nDCG against hand-computed values, RRF formula, cosine similarity |
| Pipelines | All four strategies, result schema, fallbacks when BM25 or the LLM is unavailable |
| Tenant isolation | Every pipeline passes the user filter; scoped deletion; chat, eval and run isolation; cache keys include the user |
| Authorization (API) | JWT, 404 for foreign documents, quotas, upload validation, SSRF via the API |
| Ingestion security | SSRF guard including redirects and DNS-to-private, content type and size caps, private storage permissions |
| API keys | Validation and fingerprinting, probe cache, automatic model switch on 403/404/410 but never on 401 |
| Evaluation | Cross-model generation and de-duplication, ground-truth import, usage and cost tracking, judge independence, failure classification, benchmark runner |
| Test sets | Chunk origin, upload validation (schema, evidence rules, foreign documents, size and row caps, duplicates), evidence matching across chunk sizes, abstention, bootstrap CI and "not enough data", identical snapshot per strategy and immutability, cascade deletion, evidence view data, cross-user access |

---

## 10. Demo script (5 minutes)

1. Sign in; show the sidebar: embedding model (local), answer and judge models, usage and quotas, "My Documents".
2. Upload a short policy document; point out "2 chunks ingested" and that chunks now carry document id and offsets.
3. Ask a question in chat; open "View Retrieved Context" to show chunks, latency, tokens.
4. Benchmark → Test Sets: upload a JSON with answerable and unanswerable cases plus one row that cites a document you don't own and one duplicate. Show the preview rejecting exactly those two with clear messages. Save the valid rows.
5. Run → choose the test set, all four strategies, K=5. Show the summary table, the "not enough data" note for a small set, the tag and failure breakdowns.
6. Evidence View: pick a case; show rank-ordered chunks marked ✅/❌ with "token overlap 100%" versus "text overlap 44% below threshold 50%", the gold passage, and (with judge metrics on) the answer and verdicts; filter to failed cases only.
7. Run History: reload the run; note the immutable snapshot and fingerprint in the configuration.

---

## 11. Likely interview questions and answers

**Why not compare by chunk id?**
Chunk ids are content hashes assigned at ingest. Two strategies may use different chunk sizes, and re-chunking a document changes every id, so id-based recall would punish or reward strategies for reasons unrelated to retrieval quality. Matching by document, page overlap and passage token overlap scores every strategy against the same human-defined evidence.

**How do you avoid a strategy gaming nDCG with many small chunks?**
Gains are binary per gold item: a chunk gains 1 only if it is the first in rank order to cover a gold item nobody earlier covered. The ideal DCG assumes one chunk per gold item at the top.

**How do you know the judge is trustworthy?**
By default the judge is a different model from the generator, the prompts are structured 0–1 scores, failures are reported as "not scored", and the Limitations box states that judge scores are LLM opinions. Calibrating the judge against human ratings is the planned next phase.

**What happens when the test set is tiny?**
The bootstrap interval is withheld below 10 cases and the UI says "not enough data" rather than printing a verdict.

**How is tenant isolation guaranteed rather than hoped for?**
A single helper builds every vector-store filter and raises on a missing user id; all SQL is keyed by user id; the ownership table returns 404 for foreign ids; there is a test that runs every pipeline and asserts the filter was passed on every call.

**What does an unanswerable test case measure?**
Whether the system abstains. The answer prompt says to reply that the context does not contain the answer; a pattern detector classifies the generated answer; the correct-abstention and false-answer rates are reported separately from retrieval metrics, which exclude those cases.

**Why can a run be reproduced months later?**
The run stores K, strategies, model ids, embedding model, git revision, dataset fingerprint and, for test sets, the full snapshot of the cases. Nothing in the report depends on the live state of the test set.

**What would you do next?**
Calibrate judge scores against human ratings; add character-offset gold evidence for non-PDF sources; a cross-encoder reranker; an admin role and audit log; Postgres for multi-node deployment; CI running the offline suite on every push.

---

## 12. Numbers to remember

- 4 retrieval strategies, 1 shared result schema
- 3 kinds of ground truth (synthetic, human-reviewed, evidence-based test sets)
- 5 retrieval metrics, 5 judge metrics and 2 abstention rates, with bootstrap 95% intervals on test-set runs
- 5 failure labels
- 9 SQLite tables, 1 Chroma collection with 8 metadata keys per chunk
- 212 offline tests, ~5 seconds
- Default models: answers from `nvidia/nemotron-3-ultra-550b-a55b`, judging by `nvidia/nemotron-3.5-lightning-30b-a3b`, embeddings by `all-MiniLM-L6-v2` locally
