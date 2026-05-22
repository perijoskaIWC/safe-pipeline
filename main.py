# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from models.schemas import ChatRequest, ChatResponse, BlockedResponse, SafetyCheckResult
from middleware.content_safety import check_input
from middleware.foundry_model import call_model
from middleware.groundedness import check_groundedness
from routers import medical, redteam

app = FastAPI(
    title="Responsible AI Showcase",
    description="Multi-layer Content Safety: Azure AI Content Safety + Foundry Guardrails + Groundedness",
    version="1.0.0"
)

# ── Register routers ──────────────────────────────────────────────────────────
# .NET analogy: app.MapControllers() — registers all route groups
app.include_router(medical.router)
app.include_router(redteam.router)

# ── Serve the frontend ────────────────────────────────────────────────────────
# Serves static/index.html at http://localhost:8000
# .NET analogy: app.UseStaticFiles()
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Main chat endpoint ────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": BlockedResponse},
        500: {"description": "Internal error"}
    }
)
async def chat(request: ChatRequest):
    layers_passed: list[SafetyCheckResult] = []

    # LAYER 1 — Content Safety
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

    # LAYER 2 — Foundry model + Guardrails
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

    # LAYER 3 — Groundedness
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

    return ChatResponse(
        answer=model_answer,
        layers_passed=layers_passed,
        grounded=True if request.source_documents else None
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
