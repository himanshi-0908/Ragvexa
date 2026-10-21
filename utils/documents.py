"""
documents.py — Document ownership, per-user quotas, usage accounting and
complete deletion.

Every ingested source gets a Document row owned by exactly one user. All
deletion goes through delete_document()/delete_all_user_data(), which remove
the vector chunks, eval items and ownership row together so nothing is left
behind in any store.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func

from config import (
    MAX_BENCHMARKS_PER_DAY,
    MAX_CHUNKS_PER_USER,
    MAX_DOCS_PER_USER,
    MAX_QUERIES_PER_DAY,
)
from utils.database import (
    BenchmarkRun,
    ChatHistory,
    Document,
    SessionLocal,
    SyntheticEvalItem,
    UsageEvent,
)

logger = logging.getLogger("ragvexa")


class QuotaExceeded(Exception):
    """Raised when a per-user limit would be exceeded."""


class NotOwner(PermissionError):
    """Raised when a user tries to act on a resource they do not own."""


@dataclass
class DocumentInfo:
    id: int
    source: str
    kind: str
    chunk_count: int
    size_bytes: int
    created_at: datetime
    doc_hash: str


def content_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _require_user(user_id) -> int:
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise ValueError("A valid user_id is required")
    return user_id


# ─────────────────────────────────────────────
# Listing / ownership
# ─────────────────────────────────────────────

def list_documents(user_id: int) -> List[DocumentInfo]:
    _require_user(user_id)
    db = SessionLocal()
    try:
        rows = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return [
            DocumentInfo(r.id, r.source, r.kind, r.chunk_count or 0, r.size_bytes or 0, r.created_at, r.doc_hash)
            for r in rows
        ]
    finally:
        db.close()


def get_owned_document(user_id: int, document_id: int) -> Document:
    """Return the Document row if and only if `user_id` owns it."""
    _require_user(user_id)
    db = SessionLocal()
    try:
        row = db.query(Document).filter(Document.id == document_id).first()
        if row is None or row.user_id != user_id:
            # Same error for "missing" and "not yours" so IDs cannot be enumerated.
            raise NotOwner("Document not found")
        db.expunge(row)
        return row
    finally:
        db.close()


def document_exists(user_id: int, doc_hash: str) -> bool:
    _require_user(user_id)
    db = SessionLocal()
    try:
        return (
            db.query(Document)
            .filter(Document.user_id == user_id, Document.doc_hash == doc_hash)
            .count()
            > 0
        )
    finally:
        db.close()


# ─────────────────────────────────────────────
# Quotas
# ─────────────────────────────────────────────

def _count_today(db, user_id: int, kind: str) -> int:
    since = datetime.utcnow() - timedelta(days=1)
    return (
        db.query(func.count(UsageEvent.id))
        .filter(UsageEvent.user_id == user_id, UsageEvent.kind == kind, UsageEvent.created_at >= since)
        .scalar()
        or 0
    )


def check_ingest_quota(user_id: int, new_chunks: int) -> None:
    """Raise QuotaExceeded if adding `new_chunks` would break per-user limits."""
    _require_user(user_id)
    db = SessionLocal()
    try:
        doc_count = db.query(func.count(Document.id)).filter(Document.user_id == user_id).scalar() or 0
        if doc_count >= MAX_DOCS_PER_USER:
            raise QuotaExceeded(
                f"Document limit reached ({MAX_DOCS_PER_USER}). Delete a document before ingesting another."
            )
        chunk_total = (
            db.query(func.coalesce(func.sum(Document.chunk_count), 0))
            .filter(Document.user_id == user_id)
            .scalar()
            or 0
        )
        if chunk_total + new_chunks > MAX_CHUNKS_PER_USER:
            raise QuotaExceeded(
                f"Chunk limit reached ({chunk_total + new_chunks} > {MAX_CHUNKS_PER_USER}). "
                "Delete a document or ingest a smaller file."
            )
    finally:
        db.close()


def check_daily_quota(user_id: int, kind: str) -> None:
    """kind: 'query' | 'benchmark'. Raises QuotaExceeded when the daily cap is hit."""
    _require_user(user_id)
    limits = {"query": MAX_QUERIES_PER_DAY, "benchmark": MAX_BENCHMARKS_PER_DAY}
    limit = limits.get(kind)
    if limit is None:
        return
    db = SessionLocal()
    try:
        used = _count_today(db, user_id, kind)
        if used >= limit:
            raise QuotaExceeded(f"Daily {kind} limit reached ({limit}/24h). Try again later.")
    finally:
        db.close()


def record_usage(
    user_id: int,
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: Optional[float] = None,
) -> None:
    _require_user(user_id)
    db = SessionLocal()
    try:
        db.add(
            UsageEvent(
                user_id=user_id,
                kind=kind,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cost_usd=cost_usd,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to record usage for user_id=%s", user_id)
    finally:
        db.close()


def usage_summary(user_id: int) -> dict:
    """Today's usage counts and all-time token totals for the UI."""
    _require_user(user_id)
    db = SessionLocal()
    try:
        tokens_in, tokens_out, cost = (
            db.query(
                func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                func.sum(UsageEvent.cost_usd),
            )
            .filter(UsageEvent.user_id == user_id)
            .one()
        )
        return {
            "queries_today": _count_today(db, user_id, "query"),
            "benchmarks_today": _count_today(db, user_id, "benchmark"),
            "max_queries_per_day": MAX_QUERIES_PER_DAY,
            "max_benchmarks_per_day": MAX_BENCHMARKS_PER_DAY,
            "input_tokens_total": int(tokens_in or 0),
            "output_tokens_total": int(tokens_out or 0),
            "cost_usd_total": float(cost) if cost is not None else None,
        }
    finally:
        db.close()


# ─────────────────────────────────────────────
# Registration / deletion
# ─────────────────────────────────────────────

def register_document(
    user_id: int,
    source: str,
    doc_hash: str,
    kind: str,
    chunk_count: int,
    size_bytes: int = 0,
) -> int:
    _require_user(user_id)
    db = SessionLocal()
    try:
        row = Document(
            user_id=user_id,
            source=source,
            doc_hash=doc_hash,
            kind=kind,
            chunk_count=chunk_count,
            size_bytes=size_bytes,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()


def delete_document(user_id: int, document_id: int) -> dict:
    """
    Completely delete one document the user owns: vector chunks, eval items
    and the ownership row. Returns counts of what was removed.
    """
    from retriver.vector_store import delete_user_source

    doc = get_owned_document(user_id, document_id)  # raises NotOwner otherwise
    removed_chunks = delete_user_source(user_id, doc.source)

    db = SessionLocal()
    try:
        removed_eval = (
            db.query(SyntheticEvalItem)
            .filter(SyntheticEvalItem.user_id == user_id, SyntheticEvalItem.doc_hash == doc.doc_hash)
            .delete(synchronize_session=False)
        )
        db.query(Document).filter(Document.id == doc.id, Document.user_id == user_id).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    logger.info("Deleted document id=%s for user_id=%s (chunks=%s, eval_items=%s)",
                document_id, user_id, removed_chunks, removed_eval)
    return {"chunks": removed_chunks, "eval_items": removed_eval}


def delete_all_user_data(user_id: int) -> dict:
    """Erase everything the user owns: index, documents, eval data, chats, runs, usage."""
    from retriver.vector_store import clear_user_index

    _require_user(user_id)
    removed_chunks = clear_user_index(user_id)
    db = SessionLocal()
    try:
        counts = {"chunks": removed_chunks}
        for model, label in (
            (SyntheticEvalItem, "eval_items"),
            (Document, "documents"),
            (ChatHistory, "chats"),
            (BenchmarkRun, "benchmark_runs"),
            (UsageEvent, "usage_events"),
        ):
            counts[label] = db.query(model).filter(model.user_id == user_id).delete(synchronize_session=False)
        db.commit()
        logger.info("Deleted all data for user_id=%s: %s", user_id, counts)
        return counts
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
