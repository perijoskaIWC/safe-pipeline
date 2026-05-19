# middleware/content_safety.py
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Azure AI Content Safety
#
# What it does:
#   Scans the raw user message BEFORE it ever reaches the model.
#   Checks for: Hate, Violence, Sexual, Self-Harm content.
#   Returns a severity score 0–6 per category.
#   We block if ANY category hits our threshold.
#
# .NET analogy:
#   This is like an ActionFilter that runs before your controller method.
# ─────────────────────────────────────────────────────────────────────────────

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from config import settings
from models.schemas import SafetyCheckResult


def _get_client() -> ContentSafetyClient:
    """
    Creates the Content Safety client.
    Using a factory function keeps it lazy — only created when called.
    In production you'd use dependency injection (FastAPI's Depends()).
    """
    return ContentSafetyClient(
        endpoint=settings.content_safety_endpoint,
        credential=AzureKeyCredential(settings.content_safety_key)
    )


async def check_input(user_message: str) -> SafetyCheckResult:
    """
    Runs the user message through Azure AI Content Safety.

    Returns a SafetyCheckResult with:
      - passed = True  → message is safe, continue to Layer 2
      - passed = False → message is harmful, block immediately
    """
    client = _get_client()

    try:
        # Send text to the API — it checks all 4 categories at once
        response = client.analyze_text(
            AnalyzeTextOptions(
                text=user_message,
                categories=[
                    TextCategory.HATE,
                    TextCategory.VIOLENCE,
                    TextCategory.SEXUAL,
                    TextCategory.SELF_HARM,
                ]
            )
        )

        # Loop through all 4 category results
        # Each has a .severity between 0 (safe) and 6 (most severe)
        for result in response.categories_analysis:
            if result.severity >= settings.safety_threshold:
                # Found a violation — return blocked result immediately
                return SafetyCheckResult(
                    passed=False,
                    layer="content_safety",
                    blocked_reason=f"Content flagged as unsafe",
                    category=result.category,         # e.g. "Hate"
                    severity=result.severity           # e.g. 4
                )

        # All categories passed
        return SafetyCheckResult(
            passed=True,
            layer="content_safety"
        )

    except HttpResponseError as e:
        # API call failed — fail safe means we block rather than let it through
        raise RuntimeError(f"Content Safety API error: {e.message}")