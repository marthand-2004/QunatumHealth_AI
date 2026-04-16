"""MongoDB database dependency helpers."""
from fastapi import Request
from motor.motor_asyncio import AsyncIOMotorDatabase


def get_db(request: Request) -> AsyncIOMotorDatabase:
    """FastAPI dependency — returns the Motor database from app state."""
    return request.app.state.db
