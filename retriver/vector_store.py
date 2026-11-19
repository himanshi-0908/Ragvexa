import hashlib
import re

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DB_DIR

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings_singleton = None


def get_embedding_model_name() -> str:
    return EMBEDDING_MODEL


def get_embeddings():
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_singleton


def get_vector_store() -> Chroma:
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_DIR
    )


def tenant_filter(user_id: int, source: str | None = None) -> dict:
    """
    Build the Chroma `where` clause that scopes every read/delete to one user.

    All vector-store access MUST go through this helper so tenant isolation
    cannot be bypassed by a missing filter. A non-integer / non-positive
    user_id raises instead of silently matching everything.
    """
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise ValueError("tenant_filter requires a positive integer user_id")
    if source is None:
        return {"user_id": user_id}
    return {"$and": [{"user_id": user_id}, {"source": source}]}


def chunk_id_for(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def normalise_document_text(text: str) -> str:
    """Whitespace-normalised text: the coordinate system for char offsets."""
    return re.sub(r"\s+", " ", text or "").strip()


def _page_number(meta: dict):
    """1-based page number from loader metadata (PyPDFLoader stores 0-based `page`)."""
    if not meta:
        return None
    page = meta.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        return None
    return page + 1


def annotate_chunk_origins(chunks, documents, document_id: int | None = None) -> int:
    """
    Record where each chunk came from so evidence matching can work across
    chunking strategies: document_id, page_start/page_end (PDF only) and
    char_start/char_end, offsets in the whitespace-normalised text of the
    whole document (pages joined in order by a single space).

    `documents` are the loader output the chunks were split from, in order.
    Keys whose value would be null (pages of TXT/URL sources, offsets of a
    chunk that cannot be located) are left out because Chroma rejects None
    metadata values; readers treat a missing key as null.
    Returns the number of chunks for which offsets were found.
    """
    pages = []        # (start_offset, end_offset, page_number)
    parts = []
    cursor_total = 0
    for doc in documents or []:
        norm = normalise_document_text(getattr(doc, "page_content", "") or "")
        if not norm:
            continue
        if parts:
            cursor_total += 1  # the joining space
        pages.append((cursor_total, cursor_total + len(norm), _page_number(getattr(doc, "metadata", None))))
        parts.append(norm)
        cursor_total += len(norm)
    full_text = " ".join(parts)

    located = 0
    search_from = 0
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        meta = chunk.metadata
        for key in ("page_start", "page_end", "char_start", "char_end"):
            meta.pop(key, None)
        if document_id is not None:
            meta["document_id"] = int(document_id)

        needle = normalise_document_text(chunk.page_content)
        start = -1
        if needle and full_text:
            start = full_text.find(needle, search_from)
            if start < 0:
                start = full_text.find(needle)
        if start >= 0:
            end = start + len(needle)
            meta["char_start"] = start
            meta["char_end"] = end
            search_from = start
            located += 1
            page_nums = [p for (ps, pe, p) in pages if p is not None and ps < end and pe > start]
            if page_nums:
                meta["page_start"] = min(page_nums)
                meta["page_end"] = max(page_nums)
        else:
            # Fall back to the loader's own page metadata when the text was not found.
            page = _page_number(meta)
            if page is not None:
                meta["page_start"] = meta["page_end"] = page
    return located


def add_to_vector_store(
    chunks,
    user_id: int,
    source: str | None = None,
    document_id: int | None = None,
    documents=None,
) -> int:
    """Add chunks to the vector store tagged with the owner's user_id.

    When `documents` (the loader output the chunks were split from) is given,
    each chunk also records its origin: document_id, page range and character
    offsets (see annotate_chunk_origins). Returns number of chunks added."""
    tenant_filter(user_id)  # validates user_id
    vector_store = get_vector_store()
    if documents is not None or document_id is not None:
        annotate_chunk_origins(chunks, documents or [], document_id)
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata["user_id"] = user_id
        if source:
            chunk.metadata["source"] = source
        chunk.metadata.setdefault("source", "unknown")
        # Stable chunk_id based on content hash
        chunk.metadata["chunk_id"] = chunk_id_for(chunk.page_content)
    if chunks:
        vector_store.add_documents(chunks)
    return len(chunks)


def _delete_where(where: dict) -> int:
    vector_store = get_vector_store()
    res = vector_store.get(where=where)
    ids = res.get("ids", []) if res else []
    if ids:
        vector_store.delete(ids=ids)
    return len(ids)


def clear_user_index(user_id: int) -> int:
    """Delete all chunks for a given user. Returns count deleted."""
    return _delete_where(tenant_filter(user_id))


def delete_user_source(user_id: int, source: str) -> int:
    """Delete every chunk of one source owned by this user. Returns count deleted."""
    return _delete_where(tenant_filter(user_id, source))


def count_user_chunks(user_id: int) -> int:
    vector_store = get_vector_store()
    res = vector_store.get(where=tenant_filter(user_id), include=[])
    return len(res.get("ids", [])) if res else 0


def get_user_doc_sources(user_id: int) -> list[str]:
    """Return unique source filenames indexed for this user."""
    vector_store = get_vector_store()
    res = vector_store.get(where=tenant_filter(user_id))
    sources = set()
    for meta in res.get("metadatas", []):
        src = meta.get("source", "")
        if src:
            sources.add(src)
    return sorted(sources)


def get_user_chunk_ids(user_id: int) -> set[str]:
    """All chunk_ids currently indexed for this user (used by failure analysis)."""
    vector_store = get_vector_store()
    res = vector_store.get(where=tenant_filter(user_id))
    return {m.get("chunk_id", "") for m in res.get("metadatas", []) if m}
