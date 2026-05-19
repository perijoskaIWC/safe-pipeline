# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Think of this as your appsettings.json + IOptions<T> in one place.
# pydantic-settings reads from your .env file automatically.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Layer 1 — Azure AI Content Safety
    content_safety_endpoint: str
    content_safety_key: str

    # Layer 2 — Azure AI Foundry model deployment
    foundry_endpoint: str
    foundry_key: str
    foundry_deployment: str = "gpt-4o"

    # Layer 3 — Groundedness Detection
    groundedness_endpoint: str
    groundedness_key: str

    # App
    # Severity scale: 0=safe  2=low  4=medium  6=high
    # We block anything AT or ABOVE this threshold.
    safety_threshold: int = 2
    app_env: str = "development"

    class Config:
        env_file = ".env"          # reads from your .env file
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere (like a singleton IOptions)
settings = Settings()
