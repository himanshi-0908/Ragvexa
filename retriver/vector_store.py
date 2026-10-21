import hashlib
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


def add_to_vector_store(chunks, user_id: int, source: str | None = None) -> int:
    """Add chunks to the vector store tagged with the owner's user_id.
    Returns number of chunks added."""
    tenant_filter(user_id)  # validates user_id
    vector_store = get_vector_store()
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
