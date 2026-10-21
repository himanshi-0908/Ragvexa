from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Float,
    Boolean, inspect, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

from config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String)

    chats = relationship("ChatHistory", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chats")


class Document(Base):
    """Ownership record for every ingested source (file or URL).

    Chunks in ChromaDB carry user_id + source metadata; this table is the
    authoritative list of what a user owns so deletion and quotas can be
    enforced server-side.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    source = Column(String, nullable=False)        # display name (filename / URL)
    doc_hash = Column(String, nullable=False)      # md5 of content, for dedup
    kind = Column(String, nullable=False)          # "file" | "url"
    chunk_count = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class SyntheticEvalItem(Base):
    __tablename__ = "synthetic_eval"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    query_text = Column(Text, nullable=False)
    chunk_id = Column(String, nullable=False)       # md5 hash assigned at ingest
    source_doc = Column(String, nullable=False)     # filename the chunk came from
    doc_hash = Column(String, nullable=False)       # hash of source file, for dedup
    created_at = Column(DateTime, default=datetime.utcnow)
    # --- evaluation provenance ---
    origin = Column(String, default="synthetic")    # "synthetic" | "human"
    generator_model = Column(String, nullable=True)  # which LLM wrote the question
    reference_answer = Column(Text, nullable=True)   # expected answer (for correctness)
    reviewed = Column(Boolean, default=False)        # human has verified this item


class BenchmarkRun(Base):
    """Persisted benchmark report so experiments are reproducible and auditable."""
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    config_json = Column(Text, nullable=False)     # everything needed to re-run
    summary_json = Column(Text, nullable=False)    # per-pipeline aggregates
    report_json = Column(Text, nullable=False)     # full per-query detail
    created_at = Column(DateTime, default=datetime.utcnow)


class UsageEvent(Base):
    """Per-user usage log used for daily quotas and cost reporting."""
    __tablename__ = "usage_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    kind = Column(String, nullable=False, index=True)   # "query" | "benchmark" | "ingest"
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Create tables
Base.metadata.create_all(bind=engine)


def _migrate_sqlite_columns():
    """Add columns introduced after the first release to existing SQLite DBs.

    SQLite cannot ALTER a column, but ADD COLUMN is safe and idempotent here.
    """
    insp = inspect(engine)
    if "synthetic_eval" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("synthetic_eval")}
    wanted = {
        "origin": "VARCHAR DEFAULT 'synthetic'",
        "generator_model": "VARCHAR",
        "reference_answer": "TEXT",
        "reviewed": "BOOLEAN DEFAULT 0",
    }
    with engine.begin() as conn:
        for col, ddl in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE synthetic_eval ADD COLUMN {col} {ddl}"))


_migrate_sqlite_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
