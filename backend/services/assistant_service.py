"""Health assistant / LLM chat service.

Provides session management, context-aware system prompt construction,
and LLM-backed chat with medical disclaimer injection.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are a health assistant. You provide informational guidance only. "
    "You are NOT a substitute for medical advice. "
    "Always recommend consulting a licensed physician for any medical decisions."
)

MEDICAL_DISCLAIMER = (
    "⚠️ IMPORTANT DISCLAIMER: I am an AI health assistant and cannot provide a "
    "definitive medical diagnosis. The information I provide is for educational "
    "purposes only and is NOT a substitute for professional medical advice, "
    "diagnosis, or treatment. Please consult a licensed physician for any "
    "health concerns.\n\n"
)

# Keywords that trigger the medical disclaimer (Requirement 8.6)
DIAGNOSIS_KEYWORDS = [
    "diagnose",
    "diagnosis",
    "do i have",
    "am i sick",
    "do i suffer",
    "is it cancer",
    "is this cancer",
    "have diabetes",
    "have cvd",
    "have ckd",
    "have heart disease",
    "have kidney disease",
    "what disease",
    "what condition",
    "tell me if i have",
    "confirm i have",
]

# Max messages to include in context window
_MAX_HISTORY = 20

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


async def get_or_create_session(db: AsyncIOMotorDatabase, user_id: str) -> str:
    """Return the active session_id for user_id, creating one if none exists.

    Requirements: 8.1, 8.3
    """
    existing = await db["chat_sessions"].find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)],
    )
    if existing:
        return str(existing["_id"])

    now = datetime.utcnow()
    result = await db["chat_sessions"].insert_one({
        "user_id": user_id,
        "created_at": now,
        "updated_at": now,
    })
    return str(result.inserted_id)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


async def build_system_prompt(db: AsyncIOMotorDatabase, user_id: str) -> str:
    """Build a context-aware system prompt for the given user.

    Fetches the latest prediction and lab values from MongoDB and injects
    them into the prompt so the assistant can reference patient-specific data.

    Requirements: 8.3, 8.4, 8.5
    """
    context_lines: list[str] = [SYSTEM_PROMPT_BASE, ""]

    # Fetch latest prediction
    prediction = await db["predictions"].find_one(
        {"user_id": user_id},
        sort=[("timestamp", -1)],
    )
    if prediction:
        risk_scores: dict = prediction.get("risk_scores", {})
        if risk_scores:
            context_lines.append("Patient's latest disease risk scores:")
            for disease, score in risk_scores.items():
                context_lines.append(f"  - {disease.capitalize()}: {score:.1f}%")
            context_lines.append("")

    # Fetch latest verified document for lab values
    document = await db["documents"].find_one(
        {"user_id": user_id, "verified": True},
        sort=[("verified_at", -1)],
    )
    if document:
        lab_params: list[dict] = document.get("lab_parameters", [])
        if lab_params:
            context_lines.append("Patient's latest lab values:")
            for param in lab_params:
                name = param.get("name", "unknown")
                value = param.get("value", "N/A")
                unit = param.get("unit", "")
                abnormal = " ⚠️ ABNORMAL" if param.get("is_abnormal") else ""
                context_lines.append(f"  - {name}: {value} {unit}{abnormal}")
            context_lines.append("")

    context_lines.append(
        "Use the above patient data when answering questions. "
        "Always remind the patient to consult a physician for medical decisions."
    )

    return "\n".join(context_lines)


# ---------------------------------------------------------------------------
# Disclaimer detection
# ---------------------------------------------------------------------------


def _requires_disclaimer(message: str) -> bool:
    """Return True if the message contains diagnosis-related keywords."""
    lower = message.lower()
    return any(kw in lower for kw in DIAGNOSIS_KEYWORDS)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


def _call_gemini(system_prompt: str, history: list[dict], user_message: str) -> str:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_prompt,
    )

    # Build conversation history for Gemini
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message)
    return response.text.strip()


def _call_openai(system_prompt: str, history: list[dict], user_message: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def _template_response(message: str, system_prompt: str) -> str:
    """Fallback template response when no LLM API key is configured."""
    lower = message.lower()

    if any(kw in lower for kw in ["glucose", "blood sugar", "hba1c"]):
        return (
            "Blood glucose and HbA1c are key markers for diabetes risk. "
            "Fasting glucose above 7.0 mmol/L or HbA1c above 6.5% may indicate diabetes. "
            "Please consult your physician for a proper evaluation."
        )
    if any(kw in lower for kw in ["cholesterol", "triglyceride", "heart", "cvd", "cardiovascular"]):
        return (
            "Elevated cholesterol and triglycerides are risk factors for cardiovascular disease. "
            "A heart-healthy diet, regular exercise, and medication (if prescribed) can help manage these levels. "
            "Consult your cardiologist for personalized advice."
        )
    if any(kw in lower for kw in ["creatinine", "kidney", "ckd", "renal"]):
        return (
            "Creatinine is a key marker of kidney function. Elevated levels may indicate reduced kidney function. "
            "Staying hydrated and avoiding nephrotoxic substances can help. "
            "Please consult a nephrologist for a thorough evaluation."
        )
    if any(kw in lower for kw in ["exercise", "diet", "lifestyle", "weight", "bmi"]):
        return (
            "Lifestyle modifications are among the most effective ways to reduce disease risk. "
            "Aim for at least 150 minutes of moderate exercise per week, a balanced diet rich in vegetables and whole grains, "
            "and maintaining a healthy BMI. Small consistent changes make a big difference."
        )
    if any(kw in lower for kw in ["risk", "score", "prediction", "result"]):
        return (
            "Your risk scores reflect the probability of developing certain conditions based on your lab values and lifestyle. "
            "A higher score means greater risk, but it is not a diagnosis. "
            "Use these scores as motivation to make positive health changes and discuss them with your doctor."
        )

    return (
        "I'm your AI health assistant. I can help you understand your lab values, risk scores, "
        "and provide lifestyle guidance. For specific medical advice or diagnosis, "
        "please consult a licensed physician."
    )


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------


async def chat(
    db: AsyncIOMotorDatabase,
    session_id: str,
    user_id: str,
    message: str,
) -> str:
    """Process a user message and return the assistant's response.

    - Loads session message history from MongoDB
    - Builds context-aware system prompt with patient data
    - Calls LLM (Gemini → OpenAI → template fallback)
    - Prepends medical disclaimer if diagnosis keywords detected
    - Persists both user message and assistant response to chat_messages

    Requirements: 8.2, 8.3, 8.4, 8.5, 8.6
    """
    now = datetime.utcnow()

    # Load recent message history (Requirement 8.3)
    cursor = db["chat_messages"].find(
        {"session_id": session_id},
        sort=[("timestamp", 1)],
        limit=_MAX_HISTORY,
    )
    history = []
    async for doc in cursor:
        history.append({"role": doc["role"], "content": doc["content"]})

    # Build system prompt with patient context
    system_prompt = await build_system_prompt(db, user_id)

    # Call LLM
    response_text: Optional[str] = None

    if settings.GEMINI_API_KEY:
        try:
            response_text = _call_gemini(system_prompt, history, message)
        except Exception as exc:
            logger.warning("Gemini chat failed: %s", exc)

    if response_text is None and settings.OPENAI_API_KEY:
        try:
            response_text = _call_openai(system_prompt, history, message)
        except Exception as exc:
            logger.warning("OpenAI chat failed: %s", exc)

    if response_text is None:
        response_text = _template_response(message, system_prompt)

    # Prepend disclaimer if diagnosis-related (Requirement 8.6)
    if _requires_disclaimer(message):
        response_text = MEDICAL_DISCLAIMER + response_text

    # Persist user message
    await db["chat_messages"].insert_one({
        "session_id": session_id,
        "role": "user",
        "content": message,
        "timestamp": now,
    })

    # Persist assistant response
    await db["chat_messages"].insert_one({
        "session_id": session_id,
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.utcnow(),
    })

    # Update session updated_at
    await db["chat_sessions"].update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": datetime.utcnow()}},
    )

    return response_text


# ---------------------------------------------------------------------------
# Streaming helpers (for WebSocket endpoint)
# ---------------------------------------------------------------------------


async def stream_gemini(
    system_prompt: str,
    history: list[dict],
    user_message: str,
):
    """Async generator yielding text chunks from Gemini streaming API."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_prompt,
    )

    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat_session = model.start_chat(history=gemini_history)
    response = chat_session.send_message(user_message, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


async def stream_openai(
    system_prompt: str,
    history: list[dict],
    user_message: str,
):
    """Async generator yielding text chunks from OpenAI streaming API."""
    from openai import AsyncOpenAI  # type: ignore

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    async with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
