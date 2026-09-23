# 🔀 Laya → Claude Router

> **Let AI decide which AI to use.**
> Laya is a local, open-source decision model that classifies every prompt and routes it to the most efficient Claude model — saving tokens and cost without sacrificing quality.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL%20%7C%20macOS-lightgrey)
![Backend](https://img.shields.io/badge/Backend-Azure%20Foundry%20%7C%20Anthropic%20API%20%7C%20OpenAI-orange)

---

## What It Does

Instead of blindly sending every prompt to your most expensive model, this router uses **[Laya](https://huggingface.co/convaiinnovations/laya)** — a lightweight 421M parameter System One model — to classify your prompt locally (no API cost) across four dimensions:

| Dimension | Options |
|---|---|
| **Model** | `sonnet` · `opus` · `fable` (or GPT equivalents) |
| **Effort** | `low` · `medium` · `high` → maps to thinking budget |
| **Domain** | `coding` · `reasoning` · `creative` · `general` |
| **Complexity** | `0.0 – 2.0` score |

Then it routes to the right model with the right token budget.

```
Your prompt
    │
    ▼
Laya (local · CPU · zero API cost)
    │  classifies in one forward pass
    ▼
Right model + right effort
    │
    ▼
Response + token comparison + savings tracker
```

---

## Features

- 🧠 **Smart routing** — Laya picks the cheapest model that can do the job
- 💰 **Token comparison** — see what each model would have cost per prompt
- 📊 **Session stats** — live spend, savings, and model breakdown
- 🖥️ **CLI mode** — interactive terminal with `stats` command
- 🌐 **Web UI** — clean dark interface at `localhost:7860`
- 🔌 **Multi-backend** — works with Azure Foundry, Anthropic API, or OpenAI

---

## Supported Backends

### Option A — Azure AI Foundry *(recommended for enterprise)*
Uses your organization's Foundry resource. Data stays in your Azure tenant.

### Option B — Anthropic API
Direct API access. Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

### Option C — OpenAI
Use GPT models instead of or alongside Claude.

---

## Prerequisites

- Python 3.10+
- Linux / macOS / Windows (WSL2 recommended on Windows)
- One of the backends above
- ~3 GB disk space for Laya model weights

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/laya-claude-router.git
cd laya-claude-router
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS / WSL
# venv\Scripts\activate         # Windows CMD
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Laya model (~2.4 GB)

```bash
pip install huggingface_hub
hf download convaiinnovations/laya --local-dir ./laya
```

---

## Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

### Option A — Azure AI Foundry

```bash
export CLAUDE_CODE_USE_FOUNDRY=1
export ANTHROPIC_FOUNDRY_RESOURCE="your-resource-name"
export ANTHROPIC_FOUNDRY_API_KEY="your-api-key"
```

**Finding your Foundry credentials:**
1. Go to [ai.azure.com](https://ai.azure.com) → your project
2. Home page → copy **API key** and **Project endpoint**
3. Resource name is the subdomain: `https://<resource-name>.services.ai.azure.com/`

**Required Claude deployments in Foundry:**

| Role | Recommended deployment |
|---|---|
| Fast / everyday | `claude-sonnet-4-6` |
| Complex reasoning | `claude-opus-4-8` |
| Safety-critical | `claude-fable-5` |

---

### Option B — Anthropic API

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Update `MODELS` in `laya_router_1.py`:

```python
MODELS = {
    "sonnet": "claude-sonnet-5",
    "opus":   "claude-opus-5-5",
    "fable":  "claude-fable-5-1",
}
```

Update `call_claude()` to use the standard client:

```python
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
```

---

### Option C — OpenAI

```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```

Update `MODELS` and `call_claude()`:

```python
MODELS = {
    "sonnet": "gpt-4o-mini",
    "opus":   "gpt-4o",
    "fable":  "o1",
}
```

---

## Usage

### CLI mode

```bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES="" python3 laya_router_1.py
```

```
You › Write a Python script to parse JSON logs

[laya] routing … → SONNET / effort=medium
  Model: sonnet | Effort: medium | Domain: coding | Complexity: 0.94/2

[claude] thinking …
────────────────────────────────────────────
... Claude's response ...
────────────────────────────────────────────

[tokens used — sonnet ✓ chosen by Laya]
  Model          Input   Output    Est. Cost
  ──────────── ─────── ──────── ────────────
  sonnet            53      469   $0.007194  ← Laya
  opus              53      469   $0.035970
  fable             53      469   $0.035970

  💰 Saved ~$0.028776 vs routing everything to Opus

  Session: 1 requests · $0.0072 spent · $0.0288 saved
```

**CLI commands:**

| Command | Action |
|---|---|
| Any prompt | Route and respond |
| `stats` | Full session summary with model breakdown |
| `exit` / `quit` | Exit |

### Web UI mode

```bash
CUDA_VISIBLE_DEVICES="" python3 laya_router_1.py --web
```

Open `http://localhost:7860` in your browser.

**Web UI features:**
- Live session bar — requests · tokens · spent · saved
- Token comparison table per response
- Savings bar showing % saved vs worst-case routing
- `✓ Laya` tag on the chosen model row

Custom port:
```bash
python3 laya_router_1.py --web --port 8080
```

---

## How Laya Routes

| Model | Effort | Thinking tokens | Best for |
|---|---|---|---|
| sonnet | low | 1,024 | Simple Q&A, lookups, translations |
| sonnet | medium | 5,000 | Code, drafts, summaries |
| opus | medium | 5,000 | Analysis, research, long writing |
| opus | high | 10,000 | Architecture, deep reasoning |
| fable | any | varies | Safety-critical, adversarial inputs |

---

## Cost Model

Approximate prices per 1M tokens (update `MODEL_COSTS` in the script as pricing changes):

| Model | Input | Output |
|---|---|---|
| Sonnet | $3.00 | $15.00 |
| Opus | $15.00 | $75.00 |
| Fable | $15.00 | $75.00 |

---

## Project Structure

```
laya-claude-router/
├── laya_router_1.py      # Main router — CLI + Web UI
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Excludes weights, venv, secrets
└── README.md             # This file

# Not committed (too large / secret):
# laya/                   # Model weights ~2.4 GB — download separately
# venv/                   # Virtual environment
# .env                    # Your actual credentials
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `module 'laya' has no attribute 'load'` | Activate venv: `source venv/bin/activate` |
| Triton / CUDA error | Run with `CUDA_VISIBLE_DEVICES="" python3 ...` |
| `401 Unauthorized` | Check key length: `echo -n "$ANTHROPIC_FOUNDRY_API_KEY" \| wc -c` — should be 84 |
| `DeploymentNotFound` | Check deployment names match your Foundry portal exactly |
| `api_not_supported` | Ensure base URL ends with `/anthropic` not `/anthropic/v1` |
| `PermissionDenied` | Ask your Azure admin for **Cognitive Services User** role |
| Web UI blank | `pip install flask` |

---

## Contributing

PRs welcome. Ideas:

- [ ] OpenAI backend out of the box
- [ ] Streaming responses in web UI
- [ ] Persistent session history across restarts
- [ ] Cost config file separate from code
- [ ] Docker container

---

## License

MIT — free to use, modify, and distribute.

---

## Credits

- **[Laya](https://huggingface.co/convaiinnovations/laya)** by Convai Innovations — open-source System One decision model (Apache 2.0)
- **[Claude](https://anthropic.com)** by Anthropic
- **[Azure AI Foundry](https://ai.azure.com)** by Microsoft
