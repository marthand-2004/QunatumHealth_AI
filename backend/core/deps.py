"""FastAPI dependencies for JWT authentication and RBAC enforcement.

Requirements: 1.3, 1.5, 1.6
"""
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.database import get_db
from backend.core.security import decode_token
from backend.services.auth_service import get_user_by_id

_http_bearer = HTTPBearer()


def _bearer_token(credentials: HTTPAuthorizationCredentials = Depends(_http_bearer)) -> str:
    return credentials.credentials


async def get_current_user(
    db: AsyncIOMotorDatabase = Depends(get_db),
    token: str = Depends(_bearer_token),
) -> dict:
    """Decode the JWT from the Authorization header and return the user document.

    Raises HTTP 401 if the token is expired, invalid, wrong type, or the user
    is not found / inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except JWTError:
        raise credentials_exception

    if payload.get("type") != "access":
        raise credentials_exception

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise credentials_exception

    user_doc = await get_user_by_id(db, user_id)
    if not user_doc or not user_doc.get("is_active"):
        raise credentials_exception

    return user_doc


def require_role(roles: list[str]) -> Callable:
    """Dependency factory that enforces role membership.

    Returns a FastAPI dependency that calls ``get_current_user`` and raises
    HTTP 403 if the authenticated user's role is not in *roles*.
    """
    async def _dependency(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )
        return current_user

    return _dependency
