"""Database engine, session management, and key encryption for EvoArch."""

from __future__ import annotations

import base64
import os

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

load_dotenv()

_DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not _DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. Add it to your .env file."
    )

# psycopg2 requires the postgresql:// scheme; Neon URLs are fine as-is.
_connect_args: dict = {}
engine = create_engine(_DATABASE_URL, echo=False, connect_args=_connect_args)


def create_db_and_tables() -> None:
    """Create all SQLModel tables if they do not already exist."""
    # Import here to ensure the User model is registered before create_all.
    import evoarch.db.models  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    return Session(engine)


# ── Fernet key encryption ────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    """Derive a Fernet cipher from the SECRET_KEY env var."""
    secret = os.environ.get("SECRET_KEY", "")
    if not secret:
        raise RuntimeError("SECRET_KEY is not configured. Add it to your .env file.")
    # Fernet requires a 32-byte URL-safe base64 key; derive one from the hex secret.
    raw = bytes.fromhex(secret)[:32]
    key = base64.urlsafe_b64encode(raw)
    return Fernet(key)


def encrypt_key(plaintext: str) -> str:
    """Encrypt an API key string for storage. Returns a base64-safe ciphertext string."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_key(ciphertext: str) -> str | None:
    """Decrypt a stored API key. Returns None if the token is invalid or tampered."""
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        return None

