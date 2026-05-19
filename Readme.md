# Multi-layer Content Safety Pipeline
### Azure AI Content Safety + Foundry Guardrails + Groundedness Detection

---

## What this project does

A FastAPI backend that passes every user message through 3 safety layers
before returning a response. Each layer catches different threats:

| Layer | Technology | Catches |
|-------|-----------|---------|
| 1 | Azure AI Content Safety API | Hate, violence, sexual, self-harm in raw input |
| 2 | Azure AI Foundry Guardrails | Jailbreaks, prompt injections, policy violations |
| 3 | Groundedness Detection | Hallucinations — answers not based in your source docs |

---

## Project structure

```
safe-pipeline/
├── main.py                     ← FastAPI app + single /chat endpoint
├── config.py                   ← Settings (reads from .env)
├── requirements.txt
├── .env.example                ← Copy to .env and fill in your keys
├── test_requests.http          ← Demo scenarios (VS Code REST Client)
├── middleware/
│   ├── content_safety.py       ← Layer 1
│   ├── foundry_model.py        ← Layer 2
│   └── groundedness.py         ← Layer 3
└── models/
    └── schemas.py              ← Request/Response DTOs (like C# records)
```

---

## Step-by-step setup

### STEP 1 — Python environment

Python's virtual environment = a project-scoped NuGet package folder.
It keeps dependencies isolated per project.

```bash
# Create a virtual environment called "venv"
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# You'll see (venv) in your terminal prompt — means it's active
```

### STEP 2 — Install dependencies

```bash
# Like "dotnet restore" — reads requirements.txt and installs everything
pip install -r requirements.txt
```

### STEP 3 — Create your .env file

```bash
# Copy the template
cp .env.example .env
```

Then open `.env` and fill in your Azure keys and endpoints.

**Where to find them:**

- **Content Safety endpoint + key:**
  Azure Portal → your Content Safety resource → Keys and Endpoint

- **Foundry endpoint + key:**
  AI Foundry portal → your project → Settings → Connection details

- **Groundedness endpoint + key:**
  Same as Content Safety if it's in the same resource
  (Groundedness is part of Content Safety)

### STEP 4 — Configure Foundry Guardrails in the portal

This is the ONLY configuration you need to do in the portal for Layer 2.
You do NOT write code for it — you configure it, and Azure enforces it.

1. Go to AI Foundry portal → your project
2. Left menu → **Guardrails & Controls**
3. Click **+ Create guardrail**
4. Enable: Hate, Violence, Sexual, Self-Harm → set to **Low**
5. Enable: **Prompt Shields** (catches jailbreaks)
6. Click **Save**
7. Go to your GPT-4o deployment → **Edit** → assign your new guardrail

### STEP 5 — Run the app

```bash
# Starts FastAPI on http://localhost:8000
# reload=True means it restarts automatically when you save a file (like hot reload)
python main.py
```

### STEP 6 — Open the auto-generated Swagger UI

FastAPI generates Swagger automatically (like Swashbuckle in ASP.NET).

Open: http://localhost:8000/docs

You'll see the /chat endpoint with full schema — you can test it right there.

### STEP 7 — Run the demo scenarios

Open `test_requests.http` in VS Code with the REST Client extension.
Or use the Swagger UI at /docs.

Run each scenario in order to show all 3 layers working.

---

## Demo script (for presentations)

### Scenario 1 — Normal request
Send request #2 from test_requests.http.
Show the response: answer returned, all 3 layers show passed=true.

### Scenario 2 — Layer 1 blocks (harmful input)
Send request #3.
Show: 400 response, blocked_by="Layer 1", category="Violence", severity=4.
Point out: the model was NEVER called. Layer 1 stopped it cheaply.

### Scenario 3 — Layer 2 blocks (jailbreak)
Send request #4.
Show: 400, blocked_by="Layer 2 — Foundry Guardrail".
Point out: you wrote ZERO detection code for this. Azure did it.

### Scenario 4 — Layer 3 blocks (hallucination)
Send request #5.
Show: 422, blocked_by="Layer 3 — Groundedness Detection".
Point out: the model answered, but the answer wasn't in the source doc.

---

## Key concepts explained

### Why 3 layers and not just 1?

Each layer has a different cost, speed, and capability:

- Layer 1 is FAST and CHEAP — runs before the model call, no LLM needed
- Layer 2 is AUTOMATIC — zero code, configured in the portal
- Layer 3 is SMART — only an LLM can judge if an answer is grounded

Together they form "defense in depth" — a concept from security architecture.
If one layer misses something, the next one catches it.

### Why does Layer 2 cost nothing extra in code?

Foundry Guardrails are enforced at the PLATFORM level.
When a guardrail fires, Azure returns HTTP 400 before your code runs.
You just need to catch that specific error — that's all the "integration" needed.

### Severity levels (Layer 1)

| Value | Meaning |
|-------|---------|
| 0 | Safe |
| 2 | Low risk |
| 4 | Medium risk |
| 6 | High risk |

We set SAFETY_THRESHOLD=2 in .env, so we block anything at "Low" or above.
In a children's app you'd keep it at 2. For a developer tool you might raise to 4.