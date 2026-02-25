import os
import time
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="EVE API")

# ---------- CORS ----------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").strip()
origins = [o.strip().rstrip("/") for o in ALLOWED_ORIGINS.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DeepSeek config ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_ENDPOINT = f"{DEEPSEEK_BASE_URL}/chat/completions"  # per DeepSeek docs  [oai_citation:1‡api-docs.deepseek.com](https://api-docs.deepseek.com/)

# ---------- Abuse protection ----------
# Simple in-memory per-IP limiter:
# - max N requests per window seconds (good enough for portfolio demo)
RL_MAX = int(os.getenv("RL_MAX", "12"))         # requests
RL_WINDOW = int(os.getenv("RL_WINDOW", "60"))   # seconds
_ip_hits: Dict[str, List[float]] = {}


def _rate_limit(ip: str):
    now = time.time()
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if now - t < RL_WINDOW]
    if len(hits) >= RL_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in a minute. ({RL_MAX}/{RL_WINDOW}s)",
        )
    hits.append(now)
    _ip_hits[ip] = hits


class ChatIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    objects: List[str] = Field(default_factory=list, max_length=30)
    face_cues: List[str] = Field(default_factory=list, max_length=30)
    # optional: keep a short chat history for nicer UX
    history: Optional[List[dict]] = None


class ChatOut(BaseModel):
    answer: str


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    return {
        "ok": True,
        "cors_origins": origins,
        "deepseek_base_url": DEEPSEEK_BASE_URL,
        "model": DEEPSEEK_MODEL,
        "has_key": bool(DEEPSEEK_API_KEY),
        "rate_limit": {"max": RL_MAX, "window_s": RL_WINDOW},
    }


def _deepseek_chat(messages: List[Dict[str, Any]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="Missing DEEPSEEK_API_KEY")

    payload = {
        "model": DEEPSEEK_MODEL,  # deepseek-chat recommended for cheap assistant  [oai_citation:2‡api-docs.deepseek.com](https://api-docs.deepseek.com/)
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 220,  # keep answers short = cheap
    }

    try:
        r = requests.post(
            DEEPSEEK_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",  # Bearer auth per docs  [oai_citation:3‡api-docs.deepseek.com](https://api-docs.deepseek.com/)
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {e}")

    if r.status_code != 200:
        # bubble up a readable error
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise HTTPException(status_code=r.status_code, detail=err)

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected DeepSeek response format")


@app.post("/api/chat", response_model=ChatOut)
async def chat(payload: ChatIn, req: Request):
    # rate limit by IP (Render sets headers; Request.client works fine for demo)
    ip = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (req.client.host if req.client else "unknown")
    _rate_limit(ip)

    # Build a safe, honest assistant prompt:
    # - We do not claim medical certainty about emotions.
    # - We describe observable cues.
    system = (
        "You are EVE, a friendly camera assistant on a portfolio demo site. "
        "You answer concisely (2–5 sentences). "
        "If asked about feelings/emotions, you MUST be cautious: describe observable facial cues and uncertainty; "
        "do NOT provide medical or diagnostic claims."
    )

    objects = [o.strip() for o in payload.objects if o and o.strip()][:20]
    cues = [c.strip() for c in payload.face_cues if c and c.strip()][:20]

    context_lines = []
    if objects:
        context_lines.append(f"Detected objects: {', '.join(objects)}.")
    if cues:
        context_lines.append(f"Observed face cues: {', '.join(cues)}.")
    context = " ".join(context_lines) if context_lines else "No detections available."

    user = (
        f"{context}\n"
        f"User question: {payload.question}\n"
        f"Answer as EVE."
    )

    messages = [{"role": "system", "content": system}]

    # optional short history (keep cheap)
    if payload.history:
        for m in payload.history[-6:]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content[:600]})

    messages.append({"role": "user", "content": user})

    answer = _deepseek_chat(messages).strip()
    if not answer:
        answer = "I couldn’t generate a response that time. Try again?"

    return ChatOut(answer=answer)