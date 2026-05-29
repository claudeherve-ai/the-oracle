"""SQLite database setup for The Oracle."""

import os
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("ORACLE_DB", "sqlite:///oracle.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    category = Column(String, nullable=False)
    statement = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, default="")
    sources = Column(JSON, default=[])
    deadline = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    resolution = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)


class SignalRecord(Base):
    __tablename__ = "signals"
    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    entities = Column(JSON, default=[])
    sentiment = Column(Float, default=0.0)
    relevance = Column(Float, default=0.5)
    extra_data = Column(JSON, default={})     # renamed: 'metadata' is reserved in SQLAlchemy
    captured_at = Column(DateTime)


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


init_db()
