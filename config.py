import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = None
try:
    NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY")
except Exception:
    pass

if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if NVIDIA_API_KEY:
    NVIDIA_API_KEY = NVIDIA_API_KEY.strip(' "\'\n\r')

CHROMA_DB_DIR = "./chroma_db"

# --- Auth secrets (must come from .env, never hardcode these) ---
COOKIE_NAME = os.getenv("COOKIE_NAME", "ragvexa_auth")
COOKIE_SECRET = os.getenv("COOKIE_SECRET")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

if not COOKIE_SECRET:
    raise ValueError(
        "COOKIE_SECRET is not set. Add it to your .env file, e.g.\n"
        '  COOKIE_SECRET=<run: python -c "import secrets; print(secrets.token_hex(32))">'
    )

if not JWT_SECRET:
    raise ValueError(
        "JWT_SECRET is not set. Add it to your .env file, e.g.\n"
        '  JWT_SECRET=<run: python -c "import secrets; print(secrets.token_hex(32))">'
    )

# --- Upload limits ---
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt"}

# --- Rate limits (used by FastAPI via slowapi) ---
LOGIN_RATE_LIMIT = os.getenv("LOGIN_RATE_LIMIT", "5/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")
QUERY_RATE_LIMIT = os.getenv("QUERY_RATE_LIMIT", "20/minute")

# --- CORS: comma-separated origins allowed to call the FastAPI backend ---
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()
]