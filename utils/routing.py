import logging
import os
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
import bcrypt
from pydantic import BaseModel, Field

from config import (
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MAX_QUESTION_CHARS,
    LOGIN_RATE_LIMIT,
    UPLOAD_RATE_LIMIT,
    QUERY_RATE_LIMIT,
    EXTERNAL_LLM_NOTICE,
)
from ingestion.text_loader import load_document
from ingestion.web_loader import load_web_url
from utils.chunking import chunk_documents
from utils.database import SessionLocal, User
from utils.documents import (
    NotOwner,
    QuotaExceeded,
    check_daily_quota,
    check_ingest_quota,
    content_hash,
    delete_all_user_data,
    delete_document,
    document_exists,
    list_documents,
    record_usage,
    register_document,
    usage_summary,
)
from utils.memory import UserMemory
from utils.rate_limit import limiter
from utils.security import (
    UploadValidationError,
    private_upload_path,
    remove_quietly,
    validate_upload,
)
from retriver.vector_store import add_to_vector_store
from retriver.retriver import retrieve_pipeline, PIPELINE_NAMES
from llm.llm_handler import get_llm, generate_response_tracked

logger = logging.getLogger("ragvexa")

router = APIRouter()


def hash_password(plain: str) -> str:
    """bcrypt hash in the same format streamlit-authenticator produces."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt check. Any malformed stored hash counts as a mismatch."""
    try:
        return bcrypt.checkpw((plain or "").encode("utf-8"), (hashed or "").encode("utf-8"))
    except (ValueError, TypeError):
        return False

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class QueryRequest(BaseModel):
    # Caps prompt size — protects against someone pasting a novel into the
    # question field and blowing up your NVIDIA API cost per call.
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    strategy: str = Field("Naive", max_length=40)
    k: int = Field(5, ge=1, le=10)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)


def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    """Every protected route depends on this — no token, no user_id, no access.
    The user must still exist: a token for a deleted account is rejected."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    db = SessionLocal()
    try:
        if db.query(User.id).filter(User.id == user_id).first() is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    finally:
        db.close()
    return user_id


@router.post("/login")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Same users table the Streamlit app uses — one account works for both.
    Rate-limited to slow down password-guessing attempts."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            logger.warning("Failed login attempt for username=%s", form_data.username)
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        token = create_access_token(user.id)
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()


@router.get("/me/usage")
async def my_usage(user_id: int = Depends(get_current_user_id)):
    return usage_summary(user_id)


@router.get("/disclosure")
async def disclosure():
    return {"notice": EXTERNAL_LLM_NOTICE}


@router.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
):
    filename = file.filename or ""
    file_path = private_upload_path(user_id, filename)

    # Stream to disk in chunks, aborting early if the file exceeds the cap —
    # never trust Content-Length alone, a client can lie about it.
    size = 0
    chunk_size = 1024 * 1024
    head = b""
    hasher_bytes = bytearray()
    try:
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit",
                    )
                if not head:
                    head = chunk[:16]
                hasher_bytes.extend(chunk)
                buffer.write(chunk)

        try:
            validate_upload(filename, size, head)
        except UploadValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        doc_hash = content_hash(bytes(hasher_bytes))
        if document_exists(user_id, doc_hash):
            return {"message": "This document is already in your index", "duplicate": True}

        documents = load_document(file_path)
        chunks = chunk_documents(documents)
        try:
            check_ingest_quota(user_id, len(chunks))
        except QuotaExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))

        source = os.path.basename(filename)[:200]
        n = add_to_vector_store(chunks, user_id=user_id, source=source)
        doc_id = register_document(user_id, source, doc_hash, "file", n, size)
        record_usage(user_id, "ingest")
        return {"message": "Document processed and stored successfully", "document_id": doc_id, "chunks": n}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Upload failed")
    finally:
        remove_quietly(file_path)


@router.post("/ingest-url")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def ingest_url(
    request: Request,
    body: URLRequest,
    user_id: int = Depends(get_current_user_id),
):
    try:
        documents = load_web_url(body.url)
        if not documents:
            raise HTTPException(status_code=400, detail="No readable content found at URL")
        chunks = chunk_documents(documents)
        doc_hash = content_hash("\n".join(d.page_content for d in documents).encode())
        if document_exists(user_id, doc_hash):
            return {"message": "This page is already in your index", "duplicate": True}
        try:
            check_ingest_quota(user_id, len(chunks))
        except QuotaExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))
        source = body.url[:200]
        n = add_to_vector_store(chunks, user_id=user_id, source=source)
        doc_id = register_document(user_id, source, doc_hash, "url", n, 0)
        record_usage(user_id, "ingest")
        return {"message": "URL content processed and stored successfully", "document_id": doc_id, "chunks": n}
    except HTTPException:
        raise
    except ValueError as e:
        # e.g. blocked internal/private URL from the SSRF guard
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("URL ingestion failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="URL ingestion failed")


@router.get("/documents")
async def my_documents(user_id: int = Depends(get_current_user_id)):
    return [
        {
            "id": d.id,
            "source": d.source,
            "kind": d.kind,
            "chunk_count": d.chunk_count,
            "size_bytes": d.size_bytes,
            "created_at": d.created_at,
        }
        for d in list_documents(user_id)
    ]


@router.delete("/documents/{document_id}")
async def remove_document(document_id: int, user_id: int = Depends(get_current_user_id)):
    try:
        removed = delete_document(user_id, document_id)
    except NotOwner:
        # 404, not 403: do not reveal that someone else's document exists
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", **removed}


@router.delete("/me/data")
async def remove_all_my_data(user_id: int = Depends(get_current_user_id)):
    counts = delete_all_user_data(user_id)
    return {"message": "All your data has been deleted", **counts}


@router.post("/query")
@limiter.limit(QUERY_RATE_LIMIT)
async def query_model(
    request: Request,
    body: QueryRequest,
    user_id: int = Depends(get_current_user_id),
):
    if body.strategy not in PIPELINE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy. Choose one of {PIPELINE_NAMES}")
    try:
        check_daily_quota(user_id, "query")
    except QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    try:
        result = retrieve_pipeline(body.question, user_id=user_id, strategy=body.strategy, k=body.k)
        context = "\n\n".join(c.text for c in result.chunks)
        llm = get_llm()
        call = generate_response_tracked(llm, context, body.question)

        memory = UserMemory(user_id=user_id)
        memory.add_interaction(body.question, call.text)
        record_usage(user_id, "query", call.input_tokens, call.output_tokens, call.cost_usd)

        return {
            "answer": call.text,
            "context_used": context,
            "sources": [{"source": c.source, "chunk_id": c.chunk_id, "score": c.score} for c in result.chunks],
            "usage": {"input_tokens": call.input_tokens, "output_tokens": call.output_tokens, "cost_usd": call.cost_usd},
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Query failed")


@router.get("/history")
async def get_history(user_id: int = Depends(get_current_user_id)):
    memory = UserMemory(user_id=user_id)
    return memory.get_history()
