"""SQLModel table definitions for EvoArch user accounts."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    """One authenticated Google user, with optional personal API keys and free-trial state."""

    __tablename__ = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    google_sub: str = Field(unique=True, index=True)  # Google's stable subject identifier
    email: str
    name: str
    picture_url: str | None = None
    # API keys are Fernet-encrypted before storage; None means not yet configured.
    gemini_key_enc: str | None = None
    openai_key_enc: str | None = None
    # Free-trial counter: incremented each time the owner's key is used for this user.
    free_runs_used: int = Field(default=0)
    created_at: datetime = Field(default_factory=_utcnow)
    last_login: datetime = Field(default_factory=_utcnow)

