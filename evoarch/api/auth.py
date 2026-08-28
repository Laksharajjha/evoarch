"""Google OAuth 2.0 routes and authentication dependency for EvoArch.

Implements the OAuth 2.0 authorization code flow using plain httpx —
no authlib dependency, avoiding the cryptography/openssl binding issues
in Anaconda environments.
"""

from __future__ import annotations

import logging
import os
import secrets
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel
from sqlmodel import select

from evoarch.db.database import decrypt_key, encrypt_key, get_session
from evoarch.db.models import User

load_dotenv()

LOGGER = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "")
MAX_FREE_RUNS = int(os.environ.get("MAX_FREE_RUNS", "2"))

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

SESSION_COOKIE_NAME = "evoarch_user"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

router = APIRouter()


# ── Signed session helpers ────────────────────────────────────────────────────

def _serializer() -> URLSafeTimedSerializer:
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set.")
    return URLSafeTimedSerializer(SECRET_KEY, salt="evoarch-session")


def _sign_session(user_id: str) -> str:
    return _serializer().dumps(user_id)


def _verify_session(token: str) -> str | None:
    """Return the user_id if the cookie is valid, else None."""
    try:
        return _serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (SignatureExpired, BadSignature, Exception):
        return None


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_current_user(request: Request) -> User:
    """FastAPI dependency: resolve the signed session cookie to a User row.

    Raises HTTP 401 if the user is not logged in.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    user_id = _verify_session(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired"
        )
    with get_session() as db:
        user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return user


# ── Helpers ───────────────────────────────────────────────────────────────────

def _callback_uri(request: Request) -> str:
    """Build the OAuth callback URL, honouring reverse-proxy headers."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}/auth/callback"


def _build_auth_url(redirect_uri: str, state: str) -> str:
    """Construct the Google OAuth 2.0 authorization URL."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)


async def _exchange_code_for_userinfo(
    code: str, redirect_uri: str
) -> dict[str, Any]:
    """Exchange the authorization code for an access token, then fetch the user profile."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Step 1: Exchange code for access token.
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
        access_token: str = token_data.get("access_token", "")
        if not access_token:
            raise ValueError("No access_token in Google token response")

        # Step 2: Fetch user profile.
        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_resp.raise_for_status()
        return userinfo_resp.json()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/auth/login")
async def login(request: Request) -> RedirectResponse:
    """Redirect the browser to Google's OAuth 2.0 consent screen."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )
    state = secrets.token_urlsafe(32)
    redirect_uri = _callback_uri(request)
    auth_url = _build_auth_url(redirect_uri, state)
    response = RedirectResponse(url=auth_url)
    response.set_cookie(
        "oauth_state", state, httponly=True, samesite="lax", max_age=600
    )
    return response


@router.get("/auth/callback")
async def auth_callback(request: Request) -> RedirectResponse:
    """Exchange the Google authorization code for user info and create a session."""
    stored_state = request.cookies.get("oauth_state", "")
    returned_state = request.query_params.get("state", "")
    code = request.query_params.get("code", "")
    error = request.query_params.get("error", "")

    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {error}",
        )
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code.",
        )
    if stored_state != returned_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state mismatch — possible CSRF.",
        )

    redirect_uri = _callback_uri(request)
    try:
        userinfo = await _exchange_code_for_userinfo(code, redirect_uri)
    except Exception as exc:
        LOGGER.warning("OAuth token exchange failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OAuth token exchange failed.",
        ) from exc

    google_sub: str = userinfo.get("sub", "")
    email: str = userinfo.get("email", "")
    name: str = userinfo.get("name", email)
    picture: str | None = userinfo.get("picture")

    if not google_sub or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google did not return a valid user profile.",
        )

    # Upsert the user record.
    with get_session() as db:
        stmt = select(User).where(User.google_sub == google_sub)
        user = db.exec(stmt).first()
        if user is None:
            user = User(
                google_sub=google_sub,
                email=email,
                name=name,
                picture_url=picture,
            )
            db.add(user)
        else:
            user.name = name
            user.email = email
            user.picture_url = picture
            user.last_login = datetime.now(timezone.utc)
            db.add(user)
        db.commit()
        db.refresh(user)
        user_id = str(user.id)

    session_token = _sign_session(user_id)
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=request.headers.get("x-forwarded-proto", "http") == "https",
    )
    response.delete_cookie("oauth_state")
    return response


@router.get("/auth/logout")
async def logout() -> RedirectResponse:
    """Clear the session cookie and redirect to the home page."""
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.get("/auth/me")
async def me(request: Request) -> JSONResponse:
    """Return the current user's profile and trial status. 401 if not logged in."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return JSONResponse({"authenticated": False}, status_code=401)
    user_id = _verify_session(token)
    if not user_id:
        return JSONResponse({"authenticated": False}, status_code=401)
    with get_session() as db:
        user = db.get(User, user_id)
    if user is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    has_own_key = bool(user.gemini_key_enc or user.openai_key_enc)
    return JSONResponse({
        "authenticated": True,
        "name": user.name,
        "email": user.email,
        "picture": user.picture_url,
        "free_runs_used": user.free_runs_used,
        "max_free_runs": MAX_FREE_RUNS,
        "has_own_key": has_own_key,
    })


# ── Key management ────────────────────────────────────────────────────────────

class KeyUpdateRequest(BaseModel):
    provider: str  # "gemini" or "openai"
    api_key: str


@router.post("/api/user/keys")
async def update_user_key(request: Request, body: KeyUpdateRequest) -> JSONResponse:
    """Save or update the authenticated user's own API key (encrypted at rest)."""
    user = get_current_user(request)
    if body.provider not in {"gemini", "openai"}:
        raise HTTPException(status_code=400, detail="provider must be 'gemini' or 'openai'")
    api_key = body.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key must not be blank")

    encrypted = encrypt_key(api_key)
    with get_session() as db:
        db_user = db.get(User, user.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if body.provider == "gemini":
            db_user.gemini_key_enc = encrypted
        else:
            db_user.openai_key_enc = encrypted
        db.add(db_user)
        db.commit()

    return JSONResponse({"status": "saved"})
