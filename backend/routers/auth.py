"""Authentication router — register, login, refresh, me."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, EmailStr

from backend.core.database import get_db
from backend.core.security import decode_token, verify_password
from backend.services.auth_service import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    issue_token_pair,
)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_http_bearer = HTTPBearer()


def _bearer_token(credentials: HTTPAuthorizationCredentials = Depends(_http_bearer)) -> str:
    return credentials.credentials


# ── Request / Response schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: Literal["patient", "doctor", "admin"]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserProfile)
async def register(body: RegisterRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    user_doc = await create_user(db, body.email, body.password, body.role)
    return UserProfile(
        id=str(user_doc["_id"]),
        email=user_doc["email"],
        role=user_doc["role"],
        is_active=user_doc["is_active"],
    )


@router.post("/login", response_model=TokenPair)
async def login(body: LoginRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    user_doc = await get_user_by_email(db, body.email)
    if not user_doc or not verify_password(body.password, user_doc["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    return TokenPair(**issue_token_pair(user_doc))


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token.",
        )
    user_id = payload.get("sub")
    user_doc = await get_user_by_id(db, user_id)
    if not user_doc or not user_doc.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return TokenPair(**issue_token_pair(user_doc))


@router.get("/me", response_model=UserProfile)
async def me(db: AsyncIOMotorDatabase = Depends(get_db), authorization: str = Depends(_bearer_token)):
    try:
        payload = decode_token(authorization)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        )
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not an access token.",
        )
    user_id = payload.get("sub")
    user_doc = await get_user_by_id(db, user_id)
    if not user_doc or not user_doc.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return UserProfile(
        id=str(user_doc["_id"]),
        email=user_doc["email"],
        role=user_doc["role"],
        is_active=user_doc["is_active"],
    )
