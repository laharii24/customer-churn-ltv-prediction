"""
db.py
Uses SQLite instead of PostgreSQL — no separate database install needed.
Creates a single file, churn.db, right in your project folder.
"""

from sqlalchemy import create_engine

DB_URL = "sqlite:///churn.db"

engine = create_engine(DB_URL, echo=False)


def get_engine():
    """Return the shared SQLAlchemy engine."""
    return engine