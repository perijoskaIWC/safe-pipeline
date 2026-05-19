# main.py
# ─────────────────────────────────────────────────────────────────────────────
# The entry point — wires all 3 layers together.
#
# .NET analogy:
#   Program.cs + a single controller with one POST endpoint.
#   FastAPI's @app.post() decorator = [HttpPost] attribute.
#   The function parameters with type hints = model binding.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models.schemas import ChatRequest, ChatResponse, BlockedResponse, SafetyCheckResult
from middleware.content_safety import check_input
from middleware.foundry_model import call_model
from middleware.groundedness import check_groundedness

# ── App setup ─────────────────────────────────────────────────────────────────
# Like builder.Services + app.Build() in one line
app = FastAPI(
    title="Multi-layer Content Safety Pipeline",
    description="Demo: Azure AI Content Safety + Foundry Guardrails + Groundedness Detection",
    version="1.0.0"
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,          # returned when all layers pass
    responses={
        400: {"model": BlockedResponse},  # returned when a layer blocks
        500: {"description": "Internal error"}
    }
)
async def chat(request: ChatRequest):
    """
    Processes a chat message through 3 safety layers:

    1. Azure AI Content Safety  — scans the raw user input
    2. Foundry Guardrails       — enforced automatically by the platform
    3. Groundedness Detection   — checks if the answer is based in fact

    Returns the safe answer or a 400 with details of what was blocked and why.
    """

    layers_passed: list[SafetyCheckResult] = []

    # ── LAYER 1: Input scan ───────────────────────────────────────────────────
    # Runs BEFORE the model sees anything.
    # If this blocks, we never make a model call (saves money + latency).

    layer1 = await check_input(request.user_message)
    layers_passed.append(layer1)

    if not layer1.passed:
        return JSONResponse(
            status_code=400,
            content=BlockedResponse(
                blocked_by="Layer 1 — Azure AI Content Safety",
                reason=layer1.blocked_reason,
                category=layer1.category,
                severity=layer1.severity
            ).model_dump()
        )

    # ── LAYER 2: Model call + Guardrails ──────────────────────────────────────
    # Sends to Foundry. Guardrails run on the platform automatically.
    # A BadRequestError with code "content_filter" = Guardrail fired.

    layer2, model_answer = await call_model(
        user_message=request.user_message,
        system_prompt=request.system_prompt,
        source_documents=request.source_documents
    )
    layers_passed.append(layer2)

    if not layer2.passed:
        return JSONResponse(
            status_code=400,
            content=BlockedResponse(
                blocked_by="Layer 2 — Foundry Guardrail",
                reason=layer2.blocked_reason,
                category=layer2.category
            ).model_dump()
        )

    # ── LAYER 3: Groundedness check ───────────────────────────────────────────
    # Runs on the model's OUTPUT to catch hallucinations.
    # Skipped automatically if no source_documents were provided.

    layer3 = await check_groundedness(
        query=request.user_message,
        answer=model_answer,
        source_documents=request.source_documents
    )
    layers_passed.append(layer3)

    if not layer3.passed:
        return JSONResponse(
            status_code=422,
            content=BlockedResponse(
                blocked_by="Layer 3 — Groundedness Detection",
                reason=layer3.blocked_reason
            ).model_dump()
        )

    # ── All layers passed — return the safe answer ─────────────────────────────

    return ChatResponse(
        answer=model_answer,
        layers_passed=layers_passed,
        grounded=True if request.source_documents else None
    )


# ── Run directly with: python main.py ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)