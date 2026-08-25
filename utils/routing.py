import logging
import os
import shutil
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from config import (
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_UPLOAD_EXTENSIONS,
    LOGIN_RATE_LIMIT,
    UPLOAD_RATE_LIMIT,
    QUERY_RATE_LIMIT,
)
from ingestion.text_loader import load_document
from ingestion.web_loader import load_web_url
from utils.chunking import chunk_documents
from utils.database import SessionLocal, User
from utils.memory import UserMemory
from utils.rate_limit import limiter
from utils.security import safe_filename
from retriver.vector_store import add_to_vector_store
from retriver.retriver import retrieve_context
from llm.llm_handler import get_llm, generate_response

logger = logging.getLogger("ragvexa")

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class QueryRequest(BaseModel):
    # Caps prompt size — protects against someone pasting a novel into the
    # question field and blowing up your NVIDIA API cost per call.
    question: str = Field(..., min_length=1, max_length=2000)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)


def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    """Every protected route depends on this — no token, no user_id, no access."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/login")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Same users table the Streamlit app uses — one account works for both.
    Rate-limited to slow down password-guessing attempts."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not pwd_context.verify(form_data.password, user.hashed_password):
            logger.warning("Failed login attempt for username=%s", form_data.username)
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        token = create_access_token(user.id)
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()


@router.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    os.makedirs("temp", exist_ok=True)
    fname = safe_filename(file.filename)
    file_path = os.path.join("temp", fname)

    # Stream to disk in chunks, aborting early if the file exceeds the cap —
    # never trust Content-Length alone, a client can lie about it.
    size = 0
    chunk_size = 1024 * 1024
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit",
                    )
                buffer.write(chunk)

        documents = load_document(file_path)
        chunks = chunk_documents(documents)
        add_to_vector_store(chunks, user_id=user_id)
        return {"message": "Document processed and stored successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/ingest-url")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def ingest_url(
    request: Request,
    body: URLRequest,
    user_id: int = Depends(get_current_user_id),
):
    try:
        documents = load_web_url(body.url)
        chunks = chunk_documents(documents)
        add_to_vector_store(chunks, user_id=user_id)
        return {"message": "URL content processed and stored successfully"}
    except ValueError as e:
        # e.g. blocked internal/private URL from the SSRF guard
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("URL ingestion failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
@limiter.limit(QUERY_RATE_LIMIT)
async def query_model(
    request: Request,
    body: QueryRequest,
    user_id: int = Depends(get_current_user_id),
):
    try:
        context = retrieve_context(body.question, user_id=user_id)
        llm = get_llm()
        answer = generate_response(llm, context, body.question)

        memory = UserMemory(user_id=user_id)
        memory.add_interaction(body.question, answer)

        return {"answer": answer, "context_used": context}
    except Exception as e:
        logger.exception("Query failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_history(user_id: int = Depends(get_current_user_id)):
    memory = UserMemory(user_id=user_id)
    return memory.get_history()
