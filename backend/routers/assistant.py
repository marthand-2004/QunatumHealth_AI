"""Health assistant chat router.

Endpoints:
  POST /api/assistant/message          — send a message, get a response
  WS   /api/assistant/ws/{session_id}  — streaming LLM responses via WebSocket

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

import logging
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.services.assistant_service import (
    MEDICAL_DISCLAIMER,
    _requires_disclaimer,
    build_system_prompt,
    chat,
    get_or_create_session,
    stream_gemini,
    stream_openai,
    _template_response,
    _MAX_HISTORY,
)
from backend.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None  # optional; creates new session if omitted


class MessageResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str


# ---------------------------------------------------------------------------
# POST /message
# ---------------------------------------------------------------------------


@router.post("/message", response_model=MessageResponse)
async def send_message(
    body: MessageRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Send a chat message and receive a response within 5 seconds.

    Requirements: 8.1, 8.2, 8.3, 8.6
    """
    user_id = str(current_user["_id"])

    # Resolve or create session
    if body.session_id:
        # Validate session belongs to this user
        session_doc = await db["chat_sessions"].find_one({"_id": ObjectId(body.session_id)})
        if not session_doc or str(session_doc.get("user_id")) != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        session_id = body.session_id
    else:
        session_id = await get_or_create_session(db, user_id)

    if not body.message or not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )

    response_text = await chat(db, session_id, user_id, body.message.strip())

    return MessageResponse(
        session_id=session_id,
        response=response_text,
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# WS /ws/{session_id}
# ---------------------------------------------------------------------------


@router.websocket("/ws/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """WebSocket endpoint for streaming LLM responses.

    Protocol:
      Client → sends JSON: {"token": "<jwt>", "message": "<text>"}
      Server → streams text chunks, then sends {"done": true}

    Requirements: 8.1, 8.2, 8.3, 8.6
    """
    await websocket.accept()

    try:
        # First message must contain auth token + user message
        data = await websocket.receive_json()
        token: str = data.get("token", "")
        user_message: str = data.get("message", "").strip()

        if not token:
            await websocket.send_json({"error": "Missing auth token"})
            await websocket.close(code=4001)
            return

        if not user_message:
            await websocket.send_json({"error": "Message cannot be empty"})
            await websocket.close(code=4002)
            return

        # Authenticate via JWT
        from backend.core.security import decode_token
        from backend.services.auth_service import get_user_by_id
        from jose import JWTError

        try:
            payload = decode_token(token)
        except JWTError:
            await websocket.send_json({"error": "Invalid or expired token"})
            await websocket.close(code=4003)
            return

        if payload.get("type") != "access":
            await websocket.send_json({"error": "Invalid token type"})
            await websocket.close(code=4003)
            return

        user_id_str: str | None = payload.get("sub")
        if not user_id_str:
            await websocket.send_json({"error": "Invalid token"})
            await websocket.close(code=4003)
            return

        user_doc = await get_user_by_id(db, user_id_str)
        if not user_doc or not user_doc.get("is_active"):
            await websocket.send_json({"error": "User not found or inactive"})
            await websocket.close(code=4003)
            return

        user_id = str(user_doc["_id"])

        # Validate session ownership
        session_doc = await db["chat_sessions"].find_one({"_id": ObjectId(session_id)})
        if not session_doc or str(session_doc.get("user_id")) != user_id:
            await websocket.send_json({"error": "Session not found"})
            await websocket.close(code=4004)
            return

        # Load history
        cursor = db["chat_messages"].find(
            {"session_id": session_id},
            sort=[("timestamp", 1)],
            limit=_MAX_HISTORY,
        )
        history = []
        async for doc in cursor:
            history.append({"role": doc["role"], "content": doc["content"]})

        # Build system prompt
        system_prompt = await build_system_prompt(db, user_id)

        # Prepend disclaimer if needed
        prefix = MEDICAL_DISCLAIMER if _requires_disclaimer(user_message) else ""

        # Stream response
        full_response = prefix
        if prefix:
            await websocket.send_text(prefix)

        streamed = False

        if settings.GEMINI_API_KEY:
            try:
                async for chunk in stream_gemini(system_prompt, history, user_message):
                    full_response += chunk
                    await websocket.send_text(chunk)
                    streamed = True
            except Exception as exc:
                logger.warning("Gemini streaming failed: %s", exc)
                streamed = False

        if not streamed and settings.OPENAI_API_KEY:
            try:
                async for chunk in stream_openai(system_prompt, history, user_message):
                    full_response += chunk
                    await websocket.send_text(chunk)
                    streamed = True
            except Exception as exc:
                logger.warning("OpenAI streaming failed: %s", exc)
                streamed = False

        if not streamed:
            # Template fallback — send as single chunk
            fallback = _template_response(user_message, system_prompt)
            full_response += fallback
            await websocket.send_text(fallback)

        # Signal completion
        await websocket.send_json({"done": True})

        # Persist messages
        now = datetime.utcnow()
        await db["chat_messages"].insert_one({
            "session_id": session_id,
            "role": "user",
            "content": user_message,
            "timestamp": now,
        })
        await db["chat_messages"].insert_one({
            "session_id": session_id,
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.utcnow(),
        })
        await db["chat_sessions"].update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"updated_at": datetime.utcnow()}},
        )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from session %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for session %s: %s", session_id, exc)
        try:
            await websocket.send_json({"error": "Internal server error"})
            await websocket.close(code=1011)
        except Exception:
            pass
