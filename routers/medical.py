# routers/medical.py
# ─────────────────────────────────────────────────────────────────────────────
# MONDAY — Medical Scenario
#
# What this does:
#   Accepts a medical PDF (drug sheet, clinical guideline, patient info)
#   + a question from the user.
#   Runs it through all 3 safety layers with a medical-specific system prompt.
#   Returns the answer + which layers ran + grounding result.
#
# Why medical?
#   Hallucinations in healthcare = wrong dosages, wrong diagnoses.
#   This is the highest-stakes demo you can show a mixed audience.
#   Everyone understands why AI making up medical facts is dangerous.
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from middleware.content_safety import check_input
from middleware.foundry_model import call_model
from middleware.groundedness import check_groundedness
from models.schemas import BlockedResponse
from utils.document_loader import load_document

router = APIRouter(prefix="/medical", tags=["Medical Scenario"])

# ── The system prompt that makes this medical-specific ────────────────────────
# This is the key difference from the generic /chat endpoint.
# It tells the model to behave like a careful medical information assistant.

MEDICAL_SYSTEM_PROMPT = """
You are a medical information assistant for healthcare professionals.

STRICT RULES you must always follow:
1. Answer ONLY from the provided source documents.
2. NEVER invent drug dosages, side effects, contraindications, or diagnoses.
3. If the answer is not clearly stated in the documents, respond:
   "This information is not available in the provided documentation.
    Please consult the full prescribing information or a qualified clinician."
4. Always include this disclaimer at the end of every answer:
   "⚠️ This information is for reference only. Always verify with official
    prescribing information and consult a qualified healthcare professional."
5. Do not provide personal medical advice.
"""


# ── Response model ────────────────────────────────────────────────────────────

class MedicalResponse(BaseModel):
    question: str
    answer: str
    document_name: str
    chunks_used: int
    grounded: bool
    layer1_passed: bool
    layer2_passed: bool
    layer3_passed: bool
    blocked: bool = False
    blocked_by: str | None = None
    blocked_reason: str | None = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=MedicalResponse,
    summary="Ask a question about a medical document",
    description="""
    Upload a medical PDF and ask a clinical question.

    Demo scenario A — Grounded answer:
      Upload a drug information sheet.
      Ask: "What is the recommended adult dosage?"
      → Model answers from the document. Groundedness confirms it. ✅

    Demo scenario B — Hallucination caught:
      Upload a drug information sheet about Drug A.
      Ask: "What are the interactions with Drug B?" (not mentioned in the doc)
      → Model either says it doesn't know, or hallucinates.
      → Layer 3 catches any hallucinated answer. ❌

    Demo scenario C — Harmful input blocked:
      Ask: "How do I overdose on this medication?"
      → Layer 1 blocks it before the model is ever called. ❌
    """
)
async def ask_medical(
    file: UploadFile = File(..., description="Medical PDF document"),
    question: str = Form(..., description="Clinical question about the document")
):
    # ── Step 1: Validate file ─────────────────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .txt files are supported."
        )

    # ── Step 2: Save to temp file and extract text ────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = load_document(tmp_path, chunk_size=600)
    finally:
        os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the document."
        )

    # ── Step 3: LAYER 1 — Content Safety on the question ─────────────────────
    # Even in a medical context, we block harmful inputs first.
    # e.g. "How do I overdose?" should never reach the model.

    layer1 = await check_input(question)

    if not layer1.passed:
        return MedicalResponse(
            question=question,
            answer="",
            document_name=file.filename,
            chunks_used=len(chunks),
            grounded=False,
            layer1_passed=False,
            layer2_passed=False,
            layer3_passed=False,
            blocked=True,
            blocked_by="Layer 1 — Azure AI Content Safety",
            blocked_reason=f"{layer1.blocked_reason} "
                           f"(Category: {layer1.category}, Severity: {layer1.severity})"
        )

    # ── Step 4: LAYER 2 — Call model with medical system prompt ───────────────
    # The system prompt + document chunks are passed together.
    # Foundry Guardrails enforce platform-level policy automatically.

    layer2, answer = await call_model(
        user_message=question,
        system_prompt=MEDICAL_SYSTEM_PROMPT,
        source_documents=chunks
    )

    if not layer2.passed:
        return MedicalResponse(
            question=question,
            answer="",
            document_name=file.filename,
            chunks_used=len(chunks),
            grounded=False,
            layer1_passed=True,
            layer2_passed=False,
            layer3_passed=False,
            blocked=True,
            blocked_by="Layer 2 — Foundry Guardrail",
            blocked_reason=layer2.blocked_reason
        )

    # ── Step 5: LAYER 3 — Groundedness check on the answer ───────────────────
    # Did the model answer from the document, or did it hallucinate?
    # This is the critical safety net for medical accuracy.

    layer3 = await check_groundedness(
        query=question,
        answer=answer,
        source_documents=chunks
    )

    if not layer3.passed:
        return MedicalResponse(
            question=question,
            answer=answer,             # show what it said — but mark it blocked
            document_name=file.filename,
            chunks_used=len(chunks),
            grounded=False,
            layer1_passed=True,
            layer2_passed=True,
            layer3_passed=False,
            blocked=True,
            blocked_by="Layer 3 — Groundedness Detection",
            blocked_reason=layer3.blocked_reason
        )

    # ── All layers passed ─────────────────────────────────────────────────────
    return MedicalResponse(
        question=question,
        answer=answer,
        document_name=file.filename,
        chunks_used=len(chunks),
        grounded=True,
        layer1_passed=True,
        layer2_passed=True,
        layer3_passed=True,
        blocked=False
    )