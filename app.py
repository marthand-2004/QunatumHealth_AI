"""
QuantumHealthAI — FastAPI entry point.
Replaces the legacy Flask RAG prototype.
"""
import os
from contextlib import asynccontextmanager

import certifi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from backend.core.config import settings
from backend.routers import auth, onboarding, ocr, documents, predict, explain, recommendations, assistant, clinical, reports


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    # Use certifi CA bundle for Atlas TLS — fixes SSL handshake errors
    app.state.mongo_client = AsyncIOMotorClient(
        settings.MONGODB_URL,
        tlsCAFile=certifi.where(),
    )
    app.state.db = app.state.mongo_client[settings.MONGODB_DB]
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    app.state.mongo_client.close()


app = FastAPI(
    title="QuantumHealthAI",
    version="1.0.0",
    description="Quantum ML-powered health risk prediction platform",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,            prefix="/api/auth",            tags=["auth"])
app.include_router(onboarding.router,      prefix="/api/onboarding",      tags=["onboarding"])
app.include_router(ocr.router,             prefix="/api/ocr",             tags=["ocr"])
app.include_router(documents.router,       prefix="/api/documents",       tags=["documents"])
app.include_router(predict.router,         prefix="/api/predict",         tags=["predict"])
app.include_router(explain.router,         prefix="/api/explain",         tags=["explain"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(assistant.router,       prefix="/api/assistant",       tags=["assistant"])
app.include_router(clinical.router,        prefix="/api/clinical",        tags=["clinical"])
app.include_router(reports.router,         prefix="/api/reports",         tags=["reports"])


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
