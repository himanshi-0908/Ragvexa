"""
test_sets.py — User-written test cases with gold evidence.

A test set is a list of questions the user knows the answer to, each pointing
at WHERE the answer lives in their own documents (document, pages, passage).
Unanswerable cases (answerable=false) carry no evidence and test abstention.

Flow:  parse_test_set_file  ->  validate_test_set (preview with per-row
errors, nothing saved)  ->  save_test_set  ->  test_set_snapshot (immutable
JSON a benchmark run stores so every strategy sees identical cases).

Everything is scoped to the signed-in user: evidence may only reference that
user's documents and all reads/writes filter by user_id. Uploaded files are
untrusted: size and row caps, length limits, safe parsing, and
safe_csv_cell() for anything exported back to CSV.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sqlalchemy import func

from config import (
    MAX_TEST_SET_BYTES,
    MAX_TEST_SET_MB,
    MAX_TEST_SET_ROWS,
    MAX_TEST_SETS_PER_USER,
)
from utils.database import Document, SessionLocal, TestCase, TestCaseEvidence, TestSet
from utils.evaluator import normalise_text, token_overlap

logger = logging.getLogger("ragvexa")

MAX_QUESTION_LEN = 1000
MAX_ANSWER_LEN = 4000
MAX_PASSAGE_LEN = 4000
MAX_EVIDENCE_PER_CASE = 10
MAX_TAGS_PER_CASE = 10
MAX_TAG_LEN = 40
MAX_NAME_LEN = 100
# Fraction of passage tokens that must be found in one chunk of the document
# for the upload-time "does this passage really appear here" check.
PASSAGE_MATCH_THRESHOLD = 0.8

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}
_EVIDENCE_KEYS = ("document", "pages", "passage")
_CSV_COLUMNS = {
    "question", "expected_answer", "answerable", "tags", "gold_evidence",
    "evidence_document", "evidence_pages", "evidence_passage",
}


class TestSetError(ValueError):
    """A whole-file problem (format, size, quota); row problems go in the preview."""
    __test__ = False  # keep pytest from collecting it when imported into tests


# ─────────────────────────────────────────────
# Preview structures
# ─────────────────────────────────────────────

@dataclass
class EvidenceItem:
    document: str
    document_id: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    passage: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "document": self.document,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "passage": self.passage,
        }


@dataclass
class RowPreview:
    index: int                      # 1-based row number in the upload
    question: str = ""
    expected_answer: Optional[str] = None
    answerable: Optional[bool] = None
    tags: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class TestSetPreview:
    rows: List[RowPreview]
    source_name: str = ""

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    @property
    def valid_rows(self) -> int:
        return sum(1 for r in self.rows if r.ok)

    @property
    def error_rows(self) -> int:
        return sum(1 for r in self.rows if not r.ok)

    def as_table(self) -> List[Dict]:
        """Rows for a preview table (st.dataframe)."""
        out = []
        for r in self.rows:
            ev = "; ".join(
                f"{e.document}" + (f" p.{e.page_start}" + (f"-{e.page_end}" if e.page_end and e.page_end != e.page_start else "") if e.page_start else "")
                for e in r.evidence
            )
            out.append({
                "row": r.index,
                "status": "ok" if r.ok else "error",
                "question": r.question[:120],
                "answerable": r.answerable,
                "evidence": ev[:160],
                "tags": ", ".join(r.tags),
                "errors": " | ".join(r.errors),
                "warnings": " | ".join(r.warnings),
            })
        return out


# ─────────────────────────────────────────────
# Parsing (untrusted input)
# ─────────────────────────────────────────────

def _clean_str(value, limit: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN from pandas-style CSVs
        return None
    s = str(value).replace("\x00", "").strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    return s[:limit]


def parse_bool(value) -> Optional[bool]:
    """'true'/'false'/'yes'/'no'/'1'/'0' (any case) -> bool; anything else -> None."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def parse_pages(value) -> Tuple[Optional[int], Optional[int]]:
    """
    Accepts 4, "4", "4-6", "4,5,6", [4, 6], "[4]" or None.
    Returns (page_start, page_end) as 1-based inclusive ints. Raises ValueError.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    if isinstance(value, bool):
        raise ValueError("pages must be a number, a range like 4-6, or a list")
    if isinstance(value, int):
        nums = [value]
    elif isinstance(value, float):
        if value != value or value != int(value):
            raise ValueError("pages must be whole numbers")
        nums = [int(value)]
    elif isinstance(value, (list, tuple)):
        nums = []
        for v in value:
            s, e = parse_pages(v)
            if s is not None:
                nums.extend([s, e])
    else:
        s = str(value).strip().strip("[]()")
        nums = []
        for part in re.split(r"[,\s;]+", s):
            if not part:
                continue
            m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
            if m:
                nums.extend([int(m.group(1)), int(m.group(2))])
            elif part.isdigit():
                nums.append(int(part))
            else:
                raise ValueError(f"pages '{value}' is not a page number, range or list")
    if not nums:
        return None, None
    if any(n <= 0 or n > 100000 for n in nums):
        raise ValueError("page numbers must be positive")
    return min(nums), max(nums)


def parse_tags(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                value = json.loads(s)
            except json.JSONDecodeError:
                value = re.split(r"[,;|]", s.strip("[]"))
        else:
            value = re.split(r"[,;|]", s)
    if not isinstance(value, (list, tuple)):
        value = [value]
    tags: List[str] = []
    for t in value:
        t = _clean_str(t, MAX_TAG_LEN)
        if t and t not in tags:
            tags.append(t)
    return tags[:MAX_TAGS_PER_CASE]


def _parse_evidence(raw) -> List[Dict]:
    """gold_evidence as a list of objects, a single object, or a JSON string of either."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            raise ValueError("gold_evidence is not valid JSON")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("gold_evidence must be a list of {document, pages, passage} objects")
    items = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each gold_evidence item must be an object")
        items.append({k: item.get(k) for k in _EVIDENCE_KEYS})
    return items


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise TestSetError("File is not readable text (expected UTF-8).")


def parse_test_set_file(filename: str, data: bytes) -> List[Dict]:
    """
    Turn an uploaded CSV or JSON file into raw row dicts (one per test case).
    Enforces the size and row caps. Raises TestSetError for file-level problems.
    """
    if not data:
        raise TestSetError("File is empty.")
    if len(data) > MAX_TEST_SET_BYTES:
        raise TestSetError(f"File exceeds the {MAX_TEST_SET_MB} MB test-set limit.")
    name = (filename or "").lower()
    text = _decode(data)

    if name.endswith(".json"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise TestSetError(f"Invalid JSON: {e.msg} (line {e.lineno}).")
        if isinstance(payload, dict):
            payload = payload.get("test_cases", payload.get("cases", payload.get("items")))
        if not isinstance(payload, list):
            raise TestSetError('JSON must be a list of test cases (or {"test_cases": [...]}).')
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                rows.append({"_error": "not an object"})
            else:
                rows.append(dict(item))
    elif name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        headers = [h.strip() for h in (reader.fieldnames or []) if h]
        if "question" not in headers:
            raise TestSetError("CSV must have a 'question' column.")
        unknown = [h for h in headers if h not in _CSV_COLUMNS]
        if unknown:
            raise TestSetError(f"Unknown CSV column(s): {', '.join(unknown[:5])}. "
                               f"Allowed: {', '.join(sorted(_CSV_COLUMNS))}.")
        rows = []
        for rec in reader:
            if rec is None:
                continue
            row = {(k or "").strip(): v for k, v in rec.items() if k}
            if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
                continue  # blank line
            rows.append(row)
    else:
        raise TestSetError("Unsupported file type: upload a .csv or .json file.")

    if len(rows) > MAX_TEST_SET_ROWS:
        raise TestSetError(f"Too many rows ({len(rows)}); the limit is {MAX_TEST_SET_ROWS} per test set.")
    if not rows:
        raise TestSetError("No test cases found in the file.")
    return rows


# ─────────────────────────────────────────────
# Validation (nothing is saved here)
# ─────────────────────────────────────────────

def _canonical_question(q: str) -> str:
    return normalise_text(q)


def _match_document(name: str, docs: List[Document]) -> Tuple[Optional[Document], Optional[str]]:
    """Find the user's document by display name (exact, then case-insensitive, then basename)."""
    wanted = (name or "").strip()
    if not wanted:
        return None, "document name is empty"
    exact = [d for d in docs if d.source == wanted]
    if not exact:
        low = wanted.lower()
        exact = [d for d in docs if d.source.lower() == low]
    if not exact:
        base = low.rsplit("/", 1)[-1]
        exact = [d for d in docs if d.source.lower().rsplit("/", 1)[-1] == base]
    if not exact:
        return None, f"document '{wanted[:60]}' is not one of your documents"
    exact.sort(key=lambda d: (d.created_at or 0, d.id), reverse=True)
    warn = f"several of your documents are named '{wanted[:60]}'; the newest was used" if len(exact) > 1 else None
    return exact[0], warn


def default_chunk_loader(user_id: int, source: str) -> List[Dict]:
    """The user's indexed chunks of one source, with origin metadata."""
    from retriver.vector_store import get_vector_store, tenant_filter
    res = get_vector_store().get(where=tenant_filter(user_id, source)) or {}
    chunks = []
    for text, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        meta = meta or {}
        chunks.append({
            "text": text or "",
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
        })
    chunks.sort(key=lambda c: (c["char_start"] is None, c["char_start"] or 0))
    return chunks


def _passage_found(passage: str, page_start, page_end, chunks: List[Dict]) -> Tuple[bool, float, Optional[str]]:
    """
    Fuzzy check that `passage` appears in the document (on the given pages when
    the chunks carry pages). Returns (found, best_overlap, note).
    """
    if not chunks:
        return False, 0.0, "the document has no indexed chunks"
    candidates = chunks
    if page_start is not None:
        paged = [c for c in chunks if c.get("page_start") is not None]
        if paged:
            candidates = [c for c in paged if c["page_start"] <= page_end and c["page_end"] >= page_start]
            if not candidates:
                return False, 0.0, f"the document has no text on pages {page_start}-{page_end}"
    best = max(token_overlap(passage, c["text"]) for c in candidates)
    if best >= PASSAGE_MATCH_THRESHOLD:
        return True, best, None
    # A passage can straddle a chunk boundary: check the joined text too.
    joined = normalise_text(" ".join(c["text"] for c in candidates))
    if normalise_text(passage) in joined:
        return True, 1.0, None
    return False, best, None


def validate_test_set(
    user_id: int,
    rows: List[Dict],
    check_passages: bool = True,
    chunk_loader: Optional[Callable[[int, str], List[Dict]]] = None,
    source_name: str = "",
) -> TestSetPreview:
    """
    Validate raw rows against the signed-in user's documents and return a
    preview with per-row errors and warnings. Nothing is written.
    """
    from utils.documents import _require_user
    _require_user(user_id)
    if chunk_loader is None:
        chunk_loader = default_chunk_loader

    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.user_id == user_id).all()
        for d in docs:
            db.expunge(d)
    finally:
        db.close()

    chunk_cache: Dict[str, List[Dict]] = {}
    seen_questions: Dict[str, int] = {}
    previews: List[RowPreview] = []

    for i, raw in enumerate(rows, start=1):
        row = RowPreview(index=i)
        previews.append(row)
        if not isinstance(raw, dict) or "_error" in raw:
            row.errors.append("row is not an object")
            continue
        unknown = [k for k in raw if k not in _CSV_COLUMNS]
        if unknown:
            row.warnings.append(f"ignored unknown field(s): {', '.join(str(u)[:30] for u in unknown[:3])}")

        q = _clean_str(raw.get("question"), MAX_QUESTION_LEN)
        if not q:
            row.errors.append("question is required")
        elif len(q) < 5:
            row.errors.append("question is too short")
        row.question = q or ""
        if q and len(str(raw.get("question"))) > MAX_QUESTION_LEN:
            row.warnings.append(f"question truncated to {MAX_QUESTION_LEN} characters")

        row.expected_answer = _clean_str(raw.get("expected_answer"), MAX_ANSWER_LEN)

        answerable = parse_bool(raw.get("answerable"))
        if answerable is None:
            row.errors.append("answerable is required (true/false)")
        row.answerable = answerable

        try:
            row.tags = parse_tags(raw.get("tags"))
        except Exception:
            row.errors.append("tags must be a list or comma-separated text")

        # Evidence: gold_evidence (JSON/list) or the single-evidence CSV columns.
        ev_items: List[Dict] = []
        try:
            ev_items = _parse_evidence(raw.get("gold_evidence"))
        except ValueError as e:
            row.errors.append(str(e))
        single_doc = _clean_str(raw.get("evidence_document"), 300)
        if single_doc or _clean_str(raw.get("evidence_passage"), MAX_PASSAGE_LEN):
            ev_items.append({
                "document": single_doc,
                "pages": raw.get("evidence_pages"),
                "passage": raw.get("evidence_passage"),
            })
        if len(ev_items) > MAX_EVIDENCE_PER_CASE:
            row.errors.append(f"at most {MAX_EVIDENCE_PER_CASE} evidence items per case")
            ev_items = ev_items[:MAX_EVIDENCE_PER_CASE]

        if answerable is True and not ev_items:
            row.errors.append("answerable cases need gold_evidence (document and pages or passage)")
        if answerable is False and ev_items:
            row.errors.append("unanswerable cases must not have gold_evidence")
            ev_items = []

        for n, item in enumerate(ev_items, start=1):
            doc_name = _clean_str(item.get("document"), 300)
            if not doc_name:
                row.errors.append(f"evidence {n}: document is required")
                continue
            doc, note = _match_document(doc_name, docs)
            if doc is None:
                row.errors.append(f"evidence {n}: {note or 'document not found'}")
                continue
            if note:
                row.warnings.append(f"evidence {n}: {note}")
            try:
                p_start, p_end = parse_pages(item.get("pages"))
            except ValueError as e:
                row.errors.append(f"evidence {n}: {e}")
                continue
            passage = _clean_str(item.get("passage"), MAX_PASSAGE_LEN)
            if p_start is None and not passage:
                row.errors.append(f"evidence {n}: give pages and/or a passage")
                continue
            if p_start is not None and doc.kind != "file":
                row.warnings.append(f"evidence {n}: pages are ignored for URL sources")
            if passage and len(passage.split()) < 3:
                row.errors.append(f"evidence {n}: passage is too short to match reliably")
                continue
            if passage and check_passages:
                if doc.source not in chunk_cache:
                    try:
                        chunk_cache[doc.source] = chunk_loader(user_id, doc.source)
                    except Exception as e:
                        logger.warning("Could not load chunks for passage check: %s", e)
                        chunk_cache[doc.source] = []
                found, best, note = _passage_found(passage, p_start, p_end, chunk_cache[doc.source])
                if not found:
                    detail = note or f"best token overlap {best:.0%}, need {PASSAGE_MATCH_THRESHOLD:.0%}"
                    row.errors.append(f"evidence {n}: passage not found in '{doc.source[:40]}' ({detail}); fix the passage or pages")
                    continue
            row.evidence.append(EvidenceItem(doc.source, doc.id, p_start, p_end, passage))

        if q:
            key = _canonical_question(q)
            if key in seen_questions:
                row.errors.append(f"duplicate of row {seen_questions[key]}")
            else:
                seen_questions[key] = i

    return TestSetPreview(rows=previews, source_name=source_name)


# ─────────────────────────────────────────────
# Persistence (scoped to user_id)
# ─────────────────────────────────────────────

def save_test_set(user_id: int, name: str, preview: TestSetPreview, skip_invalid: bool = False) -> int:
    """
    Persist a validated preview as a new test set. Refuses when any row has
    errors unless skip_invalid=True (then only valid rows are saved).
    Returns the new test set id.
    """
    from utils.documents import _require_user
    _require_user(user_id)
    name = _clean_str(name, MAX_NAME_LEN) or (preview.source_name[:MAX_NAME_LEN] if preview.source_name else "") or "Test set"
    if preview.error_rows and not skip_invalid:
        raise TestSetError(f"{preview.error_rows} row(s) have errors; fix them or save only the valid rows.")
    rows = [r for r in preview.rows if r.ok]
    if not rows:
        raise TestSetError("There are no valid rows to save.")

    db = SessionLocal()
    try:
        n_sets = db.query(func.count(TestSet.id)).filter(TestSet.user_id == user_id).scalar() or 0
        if n_sets >= MAX_TEST_SETS_PER_USER:
            raise TestSetError(f"Test-set limit reached ({MAX_TEST_SETS_PER_USER}); delete one first.")
        owned = {d.id for d in db.query(Document.id).filter(Document.user_id == user_id).all()}
        ts = TestSet(user_id=user_id, name=name)
        db.add(ts)
        db.flush()
        for r in rows:
            if r.answerable and any(e.document_id not in owned for e in r.evidence):
                raise TestSetError(f"row {r.index}: evidence references a document you do not own")
            tc = TestCase(
                test_set_id=ts.id,
                user_id=user_id,
                question=r.question,
                expected_answer=r.expected_answer,
                answerable=bool(r.answerable),
                tags=json.dumps(r.tags) if r.tags else None,
            )
            db.add(tc)
            db.flush()
            for e in r.evidence:
                db.add(TestCaseEvidence(
                    test_case_id=tc.id,
                    document_id=e.document_id,
                    page_start=e.page_start,
                    page_end=e.page_end,
                    passage=e.passage,
                ))
        db.commit()
        logger.info("Saved test set id=%s for user_id=%s (%s cases)", ts.id, user_id, len(rows))
        return ts.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def list_test_sets(user_id: int) -> List[Dict]:
    from utils.documents import _require_user
    _require_user(user_id)
    db = SessionLocal()
    try:
        sets = db.query(TestSet).filter(TestSet.user_id == user_id).order_by(TestSet.created_at.desc(), TestSet.id.desc()).all()
        out = []
        for ts in sets:
            cases = db.query(TestCase).filter(TestCase.test_set_id == ts.id, TestCase.user_id == user_id).all()
            out.append({
                "id": ts.id,
                "name": ts.name,
                "created_at": ts.created_at,
                "cases": len(cases),
                "answerable": sum(1 for c in cases if c.answerable),
                "unanswerable": sum(1 for c in cases if not c.answerable),
            })
        return out
    finally:
        db.close()


def _case_dict(tc: TestCase, evidence: List[TestCaseEvidence], doc_names: Dict[int, str]) -> Dict:
    try:
        tags = json.loads(tc.tags) if tc.tags else []
    except json.JSONDecodeError:
        tags = []
    return {
        "case_id": tc.id,
        "question": tc.question,
        "expected_answer": tc.expected_answer,
        "answerable": bool(tc.answerable),
        "tags": [str(t) for t in tags],
        "gold_evidence": [
            {
                "document_id": e.document_id,
                "document": doc_names.get(e.document_id, f"document #{e.document_id}"),
                "page_start": e.page_start,
                "page_end": e.page_end,
                "passage": e.passage,
            }
            for e in evidence
        ],
    }


def load_test_set(user_id: int, test_set_id: int) -> Optional[Dict]:
    """The test set with its cases and evidence, or None if missing / not owned."""
    from utils.documents import _require_user
    _require_user(user_id)
    db = SessionLocal()
    try:
        ts = db.query(TestSet).filter(TestSet.id == test_set_id, TestSet.user_id == user_id).first()
        if ts is None:
            return None
        cases = (
            db.query(TestCase)
            .filter(TestCase.test_set_id == ts.id, TestCase.user_id == user_id)
            .order_by(TestCase.id)
            .all()
        )
        case_ids = [c.id for c in cases]
        evidence = (
            db.query(TestCaseEvidence).filter(TestCaseEvidence.test_case_id.in_(case_ids)).order_by(TestCaseEvidence.id).all()
            if case_ids else []
        )
        doc_names = {d.id: d.source for d in db.query(Document).filter(Document.user_id == user_id).all()}
        by_case: Dict[int, List[TestCaseEvidence]] = {}
        for e in evidence:
            by_case.setdefault(e.test_case_id, []).append(e)
        return {
            "test_set_id": ts.id,
            "name": ts.name,
            "created_at": ts.created_at.isoformat() if ts.created_at else None,
            "cases": [_case_dict(c, by_case.get(c.id, []), doc_names) for c in cases],
        }
    finally:
        db.close()


def delete_test_set(user_id: int, test_set_id: int) -> int:
    """Delete one of the user's test sets with its cases and evidence. Returns cases removed."""
    from utils.documents import _require_user
    _require_user(user_id)
    db = SessionLocal()
    try:
        ts = db.query(TestSet).filter(TestSet.id == test_set_id, TestSet.user_id == user_id).first()
        if ts is None:
            return 0
        case_ids = [c.id for c in db.query(TestCase.id).filter(TestCase.test_set_id == ts.id, TestCase.user_id == user_id).all()]
        if case_ids:
            db.query(TestCaseEvidence).filter(TestCaseEvidence.test_case_id.in_(case_ids)).delete(synchronize_session=False)
        n = db.query(TestCase).filter(TestCase.test_set_id == ts.id, TestCase.user_id == user_id).delete(synchronize_session=False)
        db.delete(ts)
        db.commit()
        return n
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_cases_for_documents(db, user_id: int, document_ids: List[int]) -> Dict[str, int]:
    """
    Remove every test case (and all its evidence) that cites one of the given
    documents. Called from utils.documents inside its own transaction.
    """
    if not document_ids:
        return {"test_cases": 0, "evidence": 0}
    case_ids = [
        r[0]
        for r in db.query(TestCaseEvidence.test_case_id)
        .join(TestCase, TestCase.id == TestCaseEvidence.test_case_id)
        .filter(TestCase.user_id == user_id, TestCaseEvidence.document_id.in_(document_ids))
        .distinct()
        .all()
    ]
    if not case_ids:
        return {"test_cases": 0, "evidence": 0}
    n_ev = db.query(TestCaseEvidence).filter(TestCaseEvidence.test_case_id.in_(case_ids)).delete(synchronize_session=False)
    n_tc = db.query(TestCase).filter(TestCase.id.in_(case_ids), TestCase.user_id == user_id).delete(synchronize_session=False)
    return {"test_cases": n_tc, "evidence": n_ev}


def delete_all_test_data(db, user_id: int) -> Dict[str, int]:
    """Remove all of a user's test sets, cases and evidence (inside the caller's transaction)."""
    case_ids = [r[0] for r in db.query(TestCase.id).filter(TestCase.user_id == user_id).all()]
    n_ev = (
        db.query(TestCaseEvidence).filter(TestCaseEvidence.test_case_id.in_(case_ids)).delete(synchronize_session=False)
        if case_ids else 0
    )
    n_tc = db.query(TestCase).filter(TestCase.user_id == user_id).delete(synchronize_session=False)
    n_ts = db.query(TestSet).filter(TestSet.user_id == user_id).delete(synchronize_session=False)
    return {"test_sets": n_ts, "test_cases": n_tc, "evidence": n_ev}


# ─────────────────────────────────────────────
# Snapshots (what a benchmark run stores)
# ─────────────────────────────────────────────

def snapshot_fingerprint(cases: List[Dict]) -> str:
    canon = json.dumps(cases, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def test_set_snapshot(user_id: int, test_set_id: int) -> Optional[Dict]:
    """
    An immutable, JSON-serialisable copy of the test set as it is right now.
    A benchmark run stores this with its configuration, so every strategy in
    the run evaluates exactly these cases and the run stays reproducible even
    if the test set or documents change afterwards.
    """
    data = load_test_set(user_id, test_set_id)
    if data is None:
        return None
    cases = json.loads(json.dumps(data["cases"], default=str))
    return {
        "test_set_id": data["test_set_id"],
        "name": data["name"],
        "created_at": data["created_at"],
        "fingerprint": snapshot_fingerprint(cases),
        "cases": cases,
    }


# ─────────────────────────────────────────────
# Export safety
# ─────────────────────────────────────────────

def safe_csv_cell(value) -> str:
    """Neutralise spreadsheet formula injection: cells starting with = + - @ (or
    a tab/CR) get a leading apostrophe so they are shown as text."""
    if value is None:
        return ""
    s = str(value)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def rows_to_csv(rows: List[Dict], columns: Optional[List[str]] = None) -> str:
    """CSV text with every cell passed through safe_csv_cell."""
    if not rows:
        return ""
    cols = columns or list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([safe_csv_cell(c) for c in cols])
    for r in rows:
        writer.writerow([safe_csv_cell(r.get(c)) for c in cols])
    return buf.getvalue()
