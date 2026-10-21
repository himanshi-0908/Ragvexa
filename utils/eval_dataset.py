"""
eval_dataset.py — Evaluation dataset management.

Two kinds of items live in the `synthetic_eval` table:

* origin="synthetic": query → chunk_id pairs (plus a reference answer) written
  by one or more LLMs during ingestion. Questions are validated and
  de-duplicated across models before being saved.
* origin="human": ground truth imported or reviewed by a person. Benchmarks
  can be restricted to reviewed items only.

Synthetic items are NOT human-labelled ground truth. Their purpose is to
enable reproducible relative comparisons between pipelines.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from utils.database import SessionLocal, SyntheticEvalItem

logger = logging.getLogger("ragvexa")

_WH_START = re.compile(
    r"^(what|why|how|when|where|who|whom|whose|which|does|do|is|are|can|could|"
    r"should|would|will|did|explain|describe|define|list|name|compare|summari[sz]e)\b",
    re.IGNORECASE,
)
_LEADING_NUMBERING = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|q\d*[:.)])\s*", re.IGNORECASE)

MIN_QUESTION_CHARS = 12
MAX_QUESTION_CHARS = 300
FUZZY_DUPLICATE_RATIO = 0.9


def _file_hash(file_path: str) -> str:
    """Compute MD5 of a file's contents for deduplication."""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception:
        h.update(file_path.encode())
    return h.hexdigest()


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ─────────────────────────────────────────────
# Question validation & de-duplication (pure, no network)
# ─────────────────────────────────────────────

def normalize_question(q: str) -> str:
    q = _LEADING_NUMBERING.sub("", q or "").strip().strip('"\u201c\u201d\'')
    q = re.sub(r"\s+", " ", q)
    return q


def _canonical(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def validate_question(q: str) -> Optional[str]:
    """Return the cleaned question, or None if it is not a usable question."""
    q = normalize_question(q)
    if not (MIN_QUESTION_CHARS <= len(q) <= MAX_QUESTION_CHARS):
        return None
    if not re.search(r"[a-zA-Z]{3,}", q):
        return None
    if not (q.endswith("?") or _WH_START.match(q)):
        return None
    lowered = q.lower()
    # Reject meta-questions that point at "the passage" instead of its content.
    if any(p in lowered for p in ("this passage", "the passage", "the text above", "according to the text")):
        return None
    return q


def is_duplicate(q: str, seen: Iterable[str]) -> bool:
    cq = _canonical(q)
    for other in seen:
        co = _canonical(other)
        if cq == co:
            return True
        if difflib.SequenceMatcher(None, cq, co).ratio() >= FUZZY_DUPLICATE_RATIO:
            return True
    return False


def dedup_questions(questions: List[str]) -> List[str]:
    """Order-preserving exact + fuzzy de-duplication."""
    kept: List[str] = []
    for q in questions:
        if not is_duplicate(q, kept):
            kept.append(q)
    return kept


def parse_qa_block(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse model output of the form:
        Q: ...
        A: ...
    Falls back to one-question-per-line when no Q:/A: markers are present.
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    q_cur: Optional[str] = None
    found_markers = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(?:\*\*)?(q|question)\s*\d*\s*[:.)-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            found_markers = True
            if q_cur:
                pairs.append((q_cur, None))
            q_cur = m.group(2).strip().strip("*")
            continue
        m = re.match(r"^(?:\*\*)?(a|answer)\s*\d*\s*[:.)-]\s*(.+)$", line, re.IGNORECASE)
        if m and q_cur:
            found_markers = True
            pairs.append((q_cur, m.group(2).strip().strip("*")))
            q_cur = None
            continue
    if q_cur:
        pairs.append((q_cur, None))
    if not found_markers:
        pairs = [(ln.strip(), None) for ln in (text or "").splitlines() if ln.strip()]
    return pairs


# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────

def dataset_exists_for_doc(user_id: int, doc_hash: str) -> bool:
    """Return True if eval data already exists for this document."""
    db = SessionLocal()
    try:
        count = (
            db.query(SyntheticEvalItem)
            .filter(
                SyntheticEvalItem.user_id == user_id,
                SyntheticEvalItem.doc_hash == doc_hash,
            )
            .count()
        )
        return count > 0
    finally:
        db.close()


def _existing_questions(db, user_id: int) -> List[str]:
    rows = db.query(SyntheticEvalItem.query_text).filter(SyntheticEvalItem.user_id == user_id).all()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────
# Cross-model synthetic generation
# ─────────────────────────────────────────────

def _generation_prompt(text: str, n: int) -> str:
    return (
        f"Read the passage and write exactly {n} distinct, specific questions a reader "
        "would ask whose answer is DIRECTLY contained in the passage, each followed by "
        "a short answer taken from the passage.\n"
        "Rules: do not refer to 'the passage' or 'the text'; questions must be "
        "self-contained; no numbering, no preamble.\n"
        "Format exactly:\nQ: <question>\nA: <answer>\n\n"
        f"Passage:\n{text[:1200]}\n"
    )


def generate_eval_dataset(
    chunks,
    user_id: int,
    source_name: str,
    doc_hash: str,
    questions_per_chunk: int = 2,
    max_chunks: int = 30,
    generator_models: Optional[List[str]] = None,
    llm_factory=None,
) -> int:
    """
    Generate evaluation questions for each chunk with EVERY configured
    generator model, validate and de-duplicate them, and persist.
    Returns the number of query-chunk pairs saved.

    Args:
        chunks: list of LangChain Document objects (already ingested)
        user_id: authenticated user's DB id
        source_name: filename/URL of the ingested document
        doc_hash: MD5 hash of the source for deduplication
        questions_per_chunk: questions requested per chunk PER MODEL
        max_chunks: cap to avoid excessive LLM calls on very large docs
        generator_models: NIM model ids; defaults to config.EVAL_GENERATOR_MODELS
        llm_factory: callable(model_id) -> llm, injectable for tests
    """
    if dataset_exists_for_doc(user_id, doc_hash):
        return 0  # Already generated, skip

    from config import EVAL_GENERATOR_MODELS
    if llm_factory is None:
        from llm.llm_handler import get_llm as llm_factory  # type: ignore

    models = generator_models or EVAL_GENERATOR_MODELS or []
    llms: Dict[str, object] = {}
    for m in models:
        try:
            llms[m] = llm_factory(m)
        except Exception as e:  # a missing model must not block the others
            logger.warning("Eval generator model %s unavailable: %s", m, e)
    if not llms:
        return 0

    db = SessionLocal()
    saved = 0
    try:
        seen_questions = _existing_questions(db, user_id)
        for chunk in chunks[:max_chunks]:
            chunk_id = chunk.metadata.get("chunk_id", _content_hash(chunk.page_content)[:12])
            text = chunk.page_content.strip()
            if len(text) < 80:
                continue  # skip trivially short chunks

            prompt = _generation_prompt(text, questions_per_chunk)
            candidates: List[Tuple[str, Optional[str], str]] = []
            for model_id, llm in llms.items():
                try:
                    response = llm.invoke(prompt)
                    content = getattr(response, "content", None) or str(response)
                except Exception as e:
                    logger.warning("Eval generation failed for model %s: %s", model_id, e)
                    continue
                taken = 0
                for q, a in parse_qa_block(content):
                    vq = validate_question(q)
                    if not vq:
                        continue
                    candidates.append((vq, (a or "").strip()[:1000] or None, model_id))
                    taken += 1
                    if taken >= questions_per_chunk:
                        break

            for q, a, model_id in candidates:
                if is_duplicate(q, seen_questions):
                    continue
                seen_questions.append(q)
                db.add(
                    SyntheticEvalItem(
                        user_id=user_id,
                        query_text=q,
                        chunk_id=chunk_id,
                        source_doc=source_name,
                        doc_hash=doc_hash,
                        origin="synthetic",
                        generator_model=model_id,
                        reference_answer=a,
                        reviewed=False,
                    )
                )
                saved += 1

        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Eval dataset generation failed for user_id=%s", user_id)
        saved = 0
    finally:
        db.close()

    return saved


# ─────────────────────────────────────────────
# Human ground truth
# ─────────────────────────────────────────────

def import_ground_truth(user_id: int, items: List[Dict], source_doc: str = "human-ground-truth") -> Dict:
    """
    Import human-labelled items. Each item:
        {"query": str,
         "relevant_chunk_ids": [str] (optional),
         "relevant_passages": [str]  (optional; matched against the user's indexed chunks),
         "reference_answer": str (optional),
         "source_doc": str (optional)}
    Items are stored with origin="human", reviewed=True.
    Returns {"saved": n, "skipped": [reasons]}.
    """
    from retriver.vector_store import get_vector_store, tenant_filter

    skipped: List[str] = []
    saved = 0

    # Load the user's chunks once for passage matching.
    passage_index: List[Tuple[str, str, str]] = []  # (chunk_id, text, source)
    try:
        res = get_vector_store().get(where=tenant_filter(user_id))
        for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            passage_index.append(((meta or {}).get("chunk_id", ""), text or "", (meta or {}).get("source", "")))
    except Exception as e:
        logger.warning("Could not load chunks for passage matching: %s", e)
    known_ids = {cid for cid, _, _ in passage_index}

    db = SessionLocal()
    try:
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                skipped.append(f"item {idx}: not an object")
                continue
            q = validate_question(str(item.get("query", "")))
            if not q:
                skipped.append(f"item {idx}: invalid or empty query")
                continue

            chunk_ids = [str(c) for c in (item.get("relevant_chunk_ids") or []) if str(c).strip()]
            for passage in item.get("relevant_passages") or []:
                needle = re.sub(r"\s+", " ", str(passage)).strip().lower()
                if len(needle) < 20:
                    continue
                for cid, text, _src in passage_index:
                    if needle in re.sub(r"\s+", " ", text).lower():
                        chunk_ids.append(cid)
            chunk_ids = list(dict.fromkeys(chunk_ids))
            unknown: List[str] = []
            if known_ids:
                unknown = [c for c in chunk_ids if c not in known_ids]
                chunk_ids = [c for c in chunk_ids if c in known_ids]
            if not chunk_ids:
                if unknown:
                    skipped.append(f"item {idx}: unknown chunk ids {unknown[:3]} (not in your index)")
                else:
                    skipped.append(f"item {idx}: no relevant chunks resolved")
                continue

            ref = item.get("reference_answer")
            src = str(item.get("source_doc") or source_doc)[:200]
            doc_hash = "human-" + _content_hash(q)[:16]
            for cid in chunk_ids:
                db.add(
                    SyntheticEvalItem(
                        user_id=user_id,
                        query_text=q,
                        chunk_id=cid,
                        source_doc=src,
                        doc_hash=doc_hash,
                        origin="human",
                        generator_model=None,
                        reference_answer=(str(ref).strip()[:2000] if ref else None),
                        reviewed=True,
                    )
                )
                saved += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return {"saved": saved, "skipped": skipped}


def list_eval_items(user_id: int, limit: int = 500) -> List[Dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(SyntheticEvalItem)
            .filter(SyntheticEvalItem.user_id == user_id)
            .order_by(SyntheticEvalItem.created_at.desc(), SyntheticEvalItem.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "query": r.query_text,
                "chunk_id": r.chunk_id,
                "source_doc": r.source_doc,
                "origin": r.origin or "synthetic",
                "generator_model": r.generator_model,
                "reference_answer": r.reference_answer,
                "reviewed": bool(r.reviewed),
            }
            for r in rows
        ]
    finally:
        db.close()


def set_reviewed(user_id: int, item_ids: List[int], reviewed: bool = True) -> int:
    """Mark items as human-reviewed. Only the owner's rows are affected."""
    if not item_ids:
        return 0
    db = SessionLocal()
    try:
        n = (
            db.query(SyntheticEvalItem)
            .filter(SyntheticEvalItem.user_id == user_id, SyntheticEvalItem.id.in_(item_ids))
            .update({SyntheticEvalItem.reviewed: reviewed}, synchronize_session=False)
        )
        db.commit()
        return n
    finally:
        db.close()


def delete_eval_items(user_id: int, item_ids: List[int]) -> int:
    if not item_ids:
        return 0
    db = SessionLocal()
    try:
        n = (
            db.query(SyntheticEvalItem)
            .filter(SyntheticEvalItem.user_id == user_id, SyntheticEvalItem.id.in_(item_ids))
            .delete(synchronize_session=False)
        )
        db.commit()
        return n
    finally:
        db.close()


# ─────────────────────────────────────────────
# Loading for benchmarks
# ─────────────────────────────────────────────

def load_eval_queries(user_id: int, limit: int = 50, reviewed_only: bool = False) -> List[Dict]:
    """
    Load eval queries for a user, grouped by query text.
    Returns dicts: query, relevant_chunk_ids, source_doc, reference_answer,
    origin, reviewed, generator_models.
    Ordering is deterministic (newest first, then id) so a re-run with the
    same data yields the same query set.
    """
    db = SessionLocal()
    try:
        q = db.query(SyntheticEvalItem).filter(SyntheticEvalItem.user_id == user_id)
        if reviewed_only:
            q = q.filter(SyntheticEvalItem.reviewed.is_(True))
        items = q.order_by(SyntheticEvalItem.created_at.desc(), SyntheticEvalItem.id.desc()).limit(limit * 3).all()

        query_map: Dict[str, Dict] = {}
        for item in items:
            if item.query_text not in query_map:
                if len(query_map) >= limit:
                    continue
                query_map[item.query_text] = {
                    "query": item.query_text,
                    "relevant_chunk_ids": [],
                    "source_doc": item.source_doc,
                    "reference_answer": item.reference_answer,
                    "origin": item.origin or "synthetic",
                    "reviewed": bool(item.reviewed),
                    "generator_models": [],
                }
            entry = query_map[item.query_text]
            if item.chunk_id not in entry["relevant_chunk_ids"]:
                entry["relevant_chunk_ids"].append(item.chunk_id)
            if item.generator_model and item.generator_model not in entry["generator_models"]:
                entry["generator_models"].append(item.generator_model)
            if not entry["reference_answer"] and item.reference_answer:
                entry["reference_answer"] = item.reference_answer
            entry["reviewed"] = entry["reviewed"] and bool(item.reviewed)
        return list(query_map.values())
    finally:
        db.close()


def dataset_fingerprint(queries: List[Dict]) -> str:
    """Stable hash of the exact query set + labels used in a benchmark run."""
    parts = sorted(f"{q['query']}|{','.join(sorted(q['relevant_chunk_ids']))}" for q in queries)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def get_eval_dataset_stats(user_id: int) -> Dict:
    """Return stats about the eval dataset for UI display."""
    db = SessionLocal()
    try:
        items = (
            db.query(SyntheticEvalItem)
            .filter(SyntheticEvalItem.user_id == user_id)
            .all()
        )
        sources: Dict[str, int] = {}
        models: Dict[str, int] = {}
        for item in items:
            sources[item.source_doc] = sources.get(item.source_doc, 0) + 1
            if item.generator_model:
                models[item.generator_model] = models.get(item.generator_model, 0) + 1
        return {
            "total_queries": len(items),
            "unique_queries": len({i.query_text for i in items}),
            "human_items": sum(1 for i in items if (i.origin or "synthetic") == "human"),
            "reviewed_items": sum(1 for i in items if i.reviewed),
            "reviewed_unique_queries": len({i.query_text for i in items if i.reviewed}),
            "sources": sources,
            "generator_models": models,
        }
    finally:
        db.close()
