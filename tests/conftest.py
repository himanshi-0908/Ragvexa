"""
Shared test configuration.

Sets the secrets and storage paths BEFORE any project module is imported so
config.py does not raise and tests never touch the real copilot.db / chroma_db.
"""

import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="ragvexa_test_")

os.environ.setdefault("COOKIE_SECRET", "test-cookie-secret-0123456789abcdef")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-0123456789abcdef")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(_TMP, 'test.db')}")
os.environ.setdefault("CHROMA_DB_DIR", os.path.join(_TMP, "chroma"))
os.environ.setdefault("UPLOAD_DIR", os.path.join(_TMP, "uploads"))
os.environ.setdefault("MAX_DOCS_PER_USER", "3")
os.environ.setdefault("MAX_CHUNKS_PER_USER", "50")
os.environ.setdefault("MAX_QUERIES_PER_DAY", "5")
os.environ.setdefault("MAX_BENCHMARKS_PER_DAY", "2")
os.environ.setdefault("MODEL_PRICES_JSON", '{"priced-model": {"input": 1.0, "output": 2.0}}')
# Never let a developer's real key leak into tests
os.environ.pop("NVIDIA_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402


@pytest.fixture
def fresh_db():
    """Wipe all rows between tests that touch the DB."""
    from utils.database import Base, engine
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def two_users(fresh_db):
    """Create two users and return their ids."""
    from utils.database import SessionLocal, User
    db = SessionLocal()
    try:
        a = User(username="alice", email="alice@example.com", name="Alice", hashed_password="x")
        b = User(username="bob", email="bob@example.com", name="Bob", hashed_password="x")
        db.add_all([a, b])
        db.commit()
        return a.id, b.id
    finally:
        db.close()
