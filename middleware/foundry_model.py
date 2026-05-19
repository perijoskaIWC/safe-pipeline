# middleware/foundry_model.py
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Azure AI Foundry model + Guardrails
#
# What it does:
#   Sends the message to your GPT-4o deployment in Foundry.
#   Foundry Guardrails run AUTOMATICALLY on the platform side —
#   you don't write detection code here, you just handle the error
#   when a guardrail fires and returns HTTP 400.
#
# What Guardrails catch that Layer 1 didn't:
#   - Prompt injection / jailbreak attempts
#   - Content that slipped through at low severity
#   - Policy violations specific to your guardrail configuration
#
# .NET analogy:
#   Calling an HttpClient endpoint and handling specific error codes.
# ─────────────────────────────────────────────────────────────────────────────

from openai import AzureOpenAI, BadRequestError

from config import settings
from models.schemas import SafetyCheckResult


def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=settings.foundry_endpoint,
        api_key=settings.foundry_key,
        api_version="2024-08-01-preview"
    )


async def call_model(
    user_message: str,
    system_prompt: str,
    source_documents: list[str]
) -> tuple[SafetyCheckResult, str]:
    """
    Calls the Foundry model and returns:
      - SafetyCheckResult (passed or blocked by guardrail)
      - The model's answer text (empty string if blocked)

    If you have source documents, we inject them into the system prompt
    so the model answers from them (enables groundedness check in Layer 3).
    """
    client = _get_client()

    # Build the system prompt
    # If source docs were provided, append them so the model can reference them
    full_system_prompt = system_prompt
    if source_documents:
        docs_text = "\n\n---\n\n".join(source_documents)
        full_system_prompt += f"\n\nSource documents:\n\n{docs_text}"

    try:
        response = client.chat.completions.create(
            model=settings.foundry_deployment,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        return (
            SafetyCheckResult(passed=True, layer="guardrail"),
            answer
        )

    except BadRequestError as e:
        # ─────────────────────────────────────────────────────────────────
        # THIS is how you detect a Foundry Guardrail block.
        # The platform returns HTTP 400 with error code "content_filter".
        # You don't call any extra API — Azure does the work, you catch it.
        # ─────────────────────────────────────────────────────────────────
        error_body = e.body or {}
        error_code = error_body.get("code", "")

        if error_code == "content_filter" or "content management policy" in str(e).lower():
            return (
                SafetyCheckResult(
                    passed=False,
                    layer="guardrail",
                    blocked_reason="Foundry Guardrail policy violation",
                    category="Policy"
                ),
                ""
            )

        # Different kind of bad request — re-raise
        raise RuntimeError(f"Foundry model error: {str(e)}")