import json
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default=None):
    """Read a setting from Streamlit secrets first, then the environment."""
    try:
        val = st.secrets.get(name)
        if val is not None and str(val).strip():
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)


NVIDIA_API_KEY = _secret("NVIDIA_API_KEY")

if NVIDIA_API_KEY:
    NVIDIA_API_KEY = NVIDIA_API_KEY.strip(' "\'\n\r')
    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

    # Guard against accidental placeholder copy-paste
    if "YOUR" in NVIDIA_API_KEY or "HERE" in NVIDIA_API_KEY:
        raise ValueError("It looks like you pasted the placeholder 'nvapi-YOUR-API-KEY-HERE' into Streamlit Cloud. Please replace it with your ACTUAL API key!")
    if not NVIDIA_API_KEY.startswith("nvapi-"):
        raise ValueError(f"NVIDIA API keys must start with 'nvapi-'. Your key starts with: {NVIDIA_API_KEY[:6]}...")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./copilot.db")

COOKIE_NAME = os.getenv("COOKIE_NAME", "ragvexa_auth")
COOKIE_SECRET = _secret("COOKIE_SECRET")
JWT_SECRET = _secret("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
COOKIE_EXPIRY_DAYS = int(os.getenv("COOKIE_EXPIRY_DAYS", "7"))

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

if len(COOKIE_SECRET) < 16 or len(JWT_SECRET) < 16:
    raise ValueError("COOKIE_SECRET and JWT_SECRET must each be at least 16 characters.")

# --- Upload limits ---
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt"}
# Private scratch space for uploads while they are parsed. Files are written
# with owner-only permissions under a per-user sub-directory and deleted as
# soon as parsing finishes.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./private_uploads")

# --- Per-user quotas (enforced server-side in utils/documents.py) ---
MAX_DOCS_PER_USER = int(os.getenv("MAX_DOCS_PER_USER", "25"))
MAX_CHUNKS_PER_USER = int(os.getenv("MAX_CHUNKS_PER_USER", "5000"))
MAX_QUERIES_PER_DAY = int(os.getenv("MAX_QUERIES_PER_DAY", "200"))
MAX_BENCHMARKS_PER_DAY = int(os.getenv("MAX_BENCHMARKS_PER_DAY", "20"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# --- URL ingestion limits ---
MAX_URL_FETCH_BYTES = int(os.getenv("MAX_URL_FETCH_BYTES", str(5 * 1024 * 1024)))
URL_FETCH_TIMEOUT_SECONDS = int(os.getenv("URL_FETCH_TIMEOUT_SECONDS", "15"))
MAX_URL_REDIRECTS = int(os.getenv("MAX_URL_REDIRECTS", "3"))

# --- Rate limits (used by FastAPI via slowapi) ---
LOGIN_RATE_LIMIT = os.getenv("LOGIN_RATE_LIMIT", "5/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")
QUERY_RATE_LIMIT = os.getenv("QUERY_RATE_LIMIT", "20/minute")

# --- CORS: comma-separated origins allowed to call the FastAPI backend ---
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()
]

# --- LLM models (NVIDIA NIM ids) ---
# Chat / answer generation model.
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
# Models used to GENERATE evaluation questions (comma-separated). Using more
# than one model reduces single-model bias in the synthetic dataset.
EVAL_GENERATOR_MODELS = [
    m.strip()
    for m in os.getenv("EVAL_GENERATOR_MODELS", f"{LLM_MODEL},meta/llama-3.1-70b-instruct").split(",")
    if m.strip()
]
# Independent judge model. Should differ from the answer model so the judge
# is not grading its own output.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta/llama-3.1-70b-instruct")

# --- Pricing (USD per 1M tokens) used ONLY to estimate cost. ---
# Format: {"model-id": {"input": 0.0, "output": 0.0}}. When a model has no
# entry, cost is reported as unavailable rather than guessed.
try:
    MODEL_PRICES = json.loads(os.getenv("MODEL_PRICES_JSON", "{}") or "{}")
    if not isinstance(MODEL_PRICES, dict):
        MODEL_PRICES = {}
except json.JSONDecodeError:
    MODEL_PRICES = {}

# --- External data disclosure ---
# Document text, questions and answers are sent to the NVIDIA API for
# generation, eval-question generation and LLM-judge scoring. Embeddings are
# computed locally. This notice is shown in the UI.
EXTERNAL_LLM_NOTICE = (
    "Document text, your questions and generated answers are sent to the NVIDIA "
    "API (third party) for answer generation, evaluation-question generation and "
    "LLM-judge scoring. Embeddings are computed locally. Do not ingest documents "
    "you are not allowed to share with a third-party API."
)
