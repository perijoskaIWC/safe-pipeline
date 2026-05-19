# models/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models = C# record / DTO classes.
# They validate incoming JSON automatically — no manual model binding needed.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Incoming request ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=4000)

    # Source documents for grounding — your RAG chunks go here.
    # The model should only answer from these docs.
    # If empty, Layer 3 groundedness check is skipped.
    source_documents: List[str] = Field(default_factory=list)

    # Optional system prompt override
    system_prompt: Optional[str] = Field(
        default="You are a helpful assistant. Answer only from the provided documents.",
        max_length=2000
    )


# ── Layer results (used internally and surfaced in the response) ───────────────

class SafetyCheckResult(BaseModel):
    passed: bool
    layer: str                          # "content_safety" | "guardrail" | "groundedness"
    blocked_reason: Optional[str] = None
    category: Optional[str] = None      # e.g. "Hate", "Violence"
    severity: Optional[int] = None      # 0-6


# ── Final response ────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    answer: str
    layers_passed: List[SafetyCheckResult]  # all 3 checks that ran
    grounded: Optional[bool] = None         # None = no docs provided


# ── Error response (returned when a layer blocks the request) ─────────────────

class BlockedResponse(BaseModel):
    blocked: bool = True
    blocked_by: str                     # which layer stopped it
    reason: str
    category: Optional[str] = None
    severity: Optional[int] = None