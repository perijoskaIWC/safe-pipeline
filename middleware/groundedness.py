# middleware/groundedness.py
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Groundedness Detection
#
# What it does:
#   After the model responds, checks whether the answer is actually
#   grounded in the source documents you provided.
#   Catches HALLUCINATIONS — facts the model invented that aren't in the docs.
#
# Only runs when source_documents were provided in the request.
# If no docs → no grounding check → layer is skipped.
#
# This uses the Azure AI Content Safety REST API directly
# because the Python SDK for groundedness is still maturing.
#
# .NET analogy:
#   A response filter / output validator that runs after your service returns.
# ─────────────────────────────────────────────────────────────────────────────

import httpx

from config import settings
from models.schemas import SafetyCheckResult


async def check_groundedness(
    query: str,
    answer: str,
    source_documents: list[str]
) -> SafetyCheckResult:
    """
    Calls the Groundedness Detection API.

    Args:
        query           — the original user question
        answer          — what the model just said
        source_documents — the docs you fed into the system prompt

    Returns SafetyCheckResult:
        passed=True  → answer is grounded in the source documents
        passed=False → model hallucinated content not in the docs
    """

    # Skip if no source documents were provided
    if not source_documents:
        return SafetyCheckResult(
            passed=True,
            layer="groundedness",
            blocked_reason="Skipped — no source documents provided"
        )

    # Build URL robustly (avoid double-slash or missing slash)
    url = (
        f"{settings.groundedness_endpoint.rstrip('/')}"
        f"/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": settings.groundedness_key,
        "Content-Type": "application/json"
    }

    # The API expects grounding sources as a plain list of strings (not objects)
    # Docs: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness

    payload = {
        "domain": "Generic",
        "task": "QnA",
        "qna": {
            "query": query
        },
        "text": answer,                        # the model's answer to check
        "groundingSources": source_documents,  # plain list of strings
        "reasoning": False                     # set True for explanation (slower + costs more)
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            # If the API returns an error status, capture and surface it instead
            if response.status_code >= 400:
                # Try parse JSON body for a clear message
                try:
                    err_body = response.json()
                except Exception:
                    err_body = response.text

                return SafetyCheckResult(
                    passed=False,
                    layer="groundedness",
                    blocked_reason=(
                        f"Groundedness API error {response.status_code}: {err_body}"
                    ),
                )

            data = response.json()

        except httpx.RequestError as e:
            return SafetyCheckResult(
                passed=False,
                layer="groundedness",
                blocked_reason=f"Groundedness request failed: {str(e)}"
            )
        except Exception as e:
            return SafetyCheckResult(
                passed=False,
                layer="groundedness",
                blocked_reason=f"Unexpected error calling Groundedness API: {str(e)}"
            )

    # ungroundedDetected = True means the model made something up
    ungrounded = data.get("ungroundedDetected", False)
    ungrounded_percentage = data.get("ungroundedPercentage", 0)

    if ungrounded:
        return SafetyCheckResult(
            passed=False,
            layer="groundedness",
            blocked_reason=(
                f"Response contains ungrounded content "
                f"({ungrounded_percentage:.0%} ungrounded)"
            )
        )

    return SafetyCheckResult(
        passed=True,
        layer="groundedness"
    )