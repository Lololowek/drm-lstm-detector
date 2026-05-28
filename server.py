"""
server.py
FastAPI DRM licence server — HTTP layer only.
Feature engineering and model inference live in inference.py.
State is in-memory (single process, no DB).

Usage:
    uvicorn server:app --port 8000 --reload
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import inference

SEQ_LEN = 10


# ── Pydantic models ───────────────────────────────────────────────────────────

class LicenceRequest(BaseModel):
    session_id: str
    licence_id: str
    device_id:  str
    timestamp:  float | None = None


class EventResponse(BaseModel):
    session_id:     str
    position:       int
    classified:     bool
    classification: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_sessions:   int
    n_classified: int


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    det = inference.load_detector()
    app.state.detector        = det
    app.state.sessions        = {}   # session_id -> list[dict]
    app.state.classifications = {}   # session_id -> classification dict
    print(f"Detector loaded — threshold={det.threshold:.2f}, device={det.device}")
    yield


app = FastAPI(title="DRM Licence Server", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/licence", response_model=EventResponse)
def post_licence(req: LicenceRequest):
    ts   = req.timestamp if req.timestamp is not None else time.time()
    buf  = app.state.sessions.setdefault(req.session_id, [])
    clfs = app.state.classifications

    if len(buf) < SEQ_LEN:
        buf.append({
            "licence_id": req.licence_id,
            "device_id":  req.device_id,
            "timestamp":  ts,
            "position":   len(buf),
        })
        if len(buf) == SEQ_LEN:
            clfs[req.session_id] = app.state.detector.classify_session(buf)

    return EventResponse(
        session_id=req.session_id,
        position=len(buf) - 1,
        classified=req.session_id in clfs,
        classification=clfs.get(req.session_id),
    )


@app.get("/sessions")
def list_sessions():
    return [
        {
            "session_id":     sid,
            "n_events":       len(events),
            "classification": app.state.classifications.get(sid, "pending"),
        }
        for sid, events in app.state.sessions.items()
    ]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return {
        "session_id":     session_id,
        "events":         app.state.sessions[session_id],
        "classification": app.state.classifications.get(session_id, "pending"),
    }


@app.get("/events/recent")
def recent_events(limit: int = 50):
    flat = []
    for sid, events in app.state.sessions.items():
        clf = app.state.classifications.get(sid)
        for ev in events:
            flat.append({
                **ev,
                "session_id":            sid,
                "classification_status": clf or "pending",
            })
    flat.sort(key=lambda e: e["timestamp"], reverse=True)
    return flat[:limit]


@app.delete("/sessions")
def clear_sessions():
    app.state.sessions.clear()
    app.state.classifications.clear()
    return {"cleared": True}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(app.state, "detector"),
        n_sessions=len(app.state.sessions),
        n_classified=len(app.state.classifications),
    )
