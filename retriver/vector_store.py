import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DB_DIR

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding_model_name() -> str:
    return EMBEDDING_MODEL

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vector_store() -> Chroma:
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_DIR
    )

def add_to_vector_store(chunks, user_id: int) -> int:
    """Add chunks to the vector store. Returns number of chunks added."""
    vector_store = get_vector_store()
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata["user_id"] = user_id
        # Stable chunk_id based on content hash
        chunk.metadata["chunk_id"] = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()[:12]
    vector_store.add_documents(chunks)
    return len(chunks)

def clear_user_index(user_id: int) -> int:
    """Delete all chunks for a given user. Returns count deleted."""
    vector_store = get_vector_store()
    res = vector_store.get(where={"user_id": user_id})
    ids = res.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)
    return len(ids)

def get_user_doc_sources(user_id: int) -> list[str]:
    """Return unique source filenames indexed for this user."""
    vector_store = get_vector_store()
    res = vector_store.get(where={"user_id": user_id})
    sources = set()
    for meta in res.get("metadatas", []):
        src = meta.get("source", "")
        if src:
            sources.add(src)
    return sorted(sources)