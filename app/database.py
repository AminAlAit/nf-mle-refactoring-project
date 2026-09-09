"""
Database setup.
"""

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

# bonus_solution raises when DB_CONN is unset. Falling back to SQLite means the API
# and the tests run without a Postgres container; compose still points at Postgres.
DEFAULT_DATABASE_URL = "sqlite:///./houses.db"
DATABASE_URL = os.getenv("DB_CONN", DEFAULT_DATABASE_URL)

# SQLite rejects connections opened on another thread, which FastAPI's threadpool
# does. Only relax this for SQLite, never Postgres.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    """
    Base class for the models.
    """


def get_db() -> Generator[Session]:
    """
    One session per request.
    """
    with SessionLocal() as session:
        yield session
