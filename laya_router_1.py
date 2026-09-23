"""
Laya → Claude Router (Azure Foundry backend)
─────────────────────────────────────────────
Laya classifies every prompt across 4 dimensions:
  • model      → sonnet | opus | fable  (Foundry deployment names)
  • effort     → low | medium | high    (maps to thinking budget tokens)
  • domain     → coding | reasoning | creative | general
  • complexity → simple | moderate | complex

New features:
  • Token comparison — shows what each model would have used vs what Laya chose
  • Session stats   — cumulative tokens saved, cost saved, requests routed
  • Savings tracker — CLI and web UI both show live savings

Usage
─────
  CLI :  python laya_router.py
  Web :  python laya_router.py --web   (opens on http://localhost:7860)

Requirements
────────────
  pip install laya anthropic flask
  export ANTHROPIC_FOUNDRY_RESOURCE="gt255-mhavwnuy-eastus2"
  export ANTHROPIC_FOUNDRY_API_KEY="<your-foundry-api-key>"
  export CLAUDE_CODE_USE_FOUNDRY=1

Set LAYA_PATH if your local checkpoint is not at ~/laya:
  export LAYA_PATH="/home/ha260853/laya"
"""

import os
import sys
import argparse
import warnings

# ── configuration ─────────────────────────────────────────────────────────────

LAYA_PATH        = os.environ.get("LAYA_PATH", os.path.expanduser("~/laya"))
FOUNDRY_RESOURCE = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE", "gt255-mhavwnuy-eastus2")
FOUNDRY_API_KEY  = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE_URL = f"https://{FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic"

# Foundry deployment names — update these to match your portal exactly
MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-8",
    "fable":  "claude-fable-5",
}

# Approximate cost per 1M tokens (input / output) in USD — update as needed
MODEL_COSTS = {
    "sonnet": {"input": 3.00,  "output": 15.00},
    "opus":   {"input": 15.00, "output": 75.00},
    "fable":  {"input": 15.00, "output": 75.00},
}

# Thinking budget tokens per effort level
EFFORT_TOKENS = {
    "low":    1024,
    "medium": 5000,
    "high":   10000,
}

# ── session stats (in-memory) ──────────────────────────────────────────────────

SESSION = {
    "requests":        0,
    "total_in":        0,
    "total_out":       0,
    "actual_cost":     0.0,
    "worst_cost":      0.0,   # cost if every prompt went to opus
    "saved_cost":      0.0,
    "model_counts":    {"sonnet": 0, "opus": 0, "fable": 0},
}

def _cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    c = MODEL_COSTS.get(model_key, MODEL_COSTS["opus"])
    return (input_tokens * c["input"] + output_tokens * c["output"]) / 1_000_000

def update_session(routing: dict, usage: dict):
    mk = routing["model_key"]
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    actual  = _cost(mk, inp, out)
    worst   = _cost("opus", inp, out)
    SESSION["requests"]     += 1
    SESSION["total_in"]     += inp
    SESSION["total_out"]    += out
    SESSION["actual_cost"]  += actual
    SESSION["worst_cost"]   += worst
    SESSION["saved_cost"]   += max(0.0, worst - actual)
    SESSION["model_counts"][mk] = SESSION["model_counts"].get(mk, 0) + 1

def token_comparison(routing: dict, usage: dict) -> dict:
    """
    Show what each model would cost for this prompt's actual token usage,
    vs what Laya chose.
    """
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    chosen = routing["model_key"]
    rows = {}
    for mk in ["sonnet", "opus", "fable"]:
        cost = _cost(mk, inp, out)
        rows[mk] = {
            "model_id": MODELS[mk],
            "input_tokens":  inp,
            "output_tokens": out,
            "cost_usd": cost,
            "chosen": mk == chosen,
        }
    return rows

# ── load laya once ────────────────────────────────────────────────────────────

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    import laya as _laya_mod
    _agent = _laya_mod.load(LAYA_PATH, device="cpu")

print(f"[laya] loaded from {LAYA_PATH}", file=sys.stderr)

# ── routing logic ─────────────────────────────────────────────────────────────

def route(prompt: str) -> dict:
    state = {"prompt": prompt}
    questions = {
        "model": {
            "type": "choice",
            "instructions": (
                "Which Claude model should handle this prompt most efficiently? "
                "Choose the least powerful model that can still do the job well."
            ),
            "criteria": {
                "sonnet": (
                    "Fast everyday tasks: summarisation, Q&A, short drafts, "
                    "translation, simple code, data lookup. Prefer when the task "
                    "is clear and does not need deep reasoning."
                ),
                "opus": (
                    "Hard multi-step tasks: complex analysis, long-form writing, "
                    "nuanced judgment, detailed code review, research synthesis. "
                    "Use when sonnet would likely miss subtleties."
                ),
                "fable": (
                    "Tasks with high-stakes safety or alignment requirements, "
                    "adversarial inputs, or content that needs extra moderation "
                    "and careful handling beyond normal accuracy."
                ),
            },
        },
        "effort": {
            "type": "choice",
            "instructions": (
                "How much thinking / reasoning budget should the model use? "
                "Match effort to genuine task difficulty, not just length."
            ),
            "criteria": {
                "low":    "Straightforward: one clear answer, no ambiguity, minimal steps.",
                "medium": "Moderate: requires a few reasoning steps or trade-off evaluation.",
                "high":   "Complex: deep reasoning, multiple competing considerations, or novel problem-solving.",
            },
        },
        "domain": {
            "type": "choice",
            "instructions": "What is the primary domain of this prompt?",
            "criteria": {
                "coding":    "Writing, debugging, explaining, or reviewing code or systems.",
                "reasoning": "Logic, math, analysis, research, planning, or argumentation.",
                "creative":  "Story, poetry, marketing copy, brainstorming, or imaginative content.",
                "general":   "Conversation, factual Q&A, translation, summarisation, or other.",
            },
        },
        "complexity": {
            "type": "score",
            "instructions": "How complex is this prompt overall?",
            "criteria": ["simple", "moderate", "complex"],
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _agent.predict(state, questions)

    answers       = result["answers"]
    model_key     = answers["model"]["choice"]
    effort_key    = answers["effort"]["choice"]
    domain        = answers["domain"]["choice"]
    complexity    = answers["complexity"]["score"]
    model_id      = MODELS[model_key]
    thinking_tokens = EFFORT_TOKENS[effort_key]

    rationale = (
        f"Model: {model_key} ({answers['model']['probabilities']}) | "
        f"Effort: {effort_key} ({answers['effort']['probabilities']}) | "
        f"Domain: {domain} | Complexity: {complexity:.2f}/2"
    )

    return {
        "model_key":       model_key,
        "model_id":        model_id,
        "effort":          effort_key,
        "thinking_tokens": thinking_tokens,
        "domain":          domain,
        "complexity":      complexity,
        "rationale":       rationale,
        "laya_raw":        result,
    }


# ── call claude ───────────────────────────────────────────────────────────────

def call_claude(prompt: str, routing: dict) -> tuple[str, dict]:
    try:
        import anthropic
    except ImportError:
        return "[error] anthropic not installed. Run: pip install anthropic", {}

    if not FOUNDRY_API_KEY:
        return "[error] ANTHROPIC_FOUNDRY_API_KEY not set.", {}

    client = anthropic.Anthropic(
        api_key=FOUNDRY_API_KEY,
        base_url=FOUNDRY_BASE_URL,
        default_headers={
            "x-api-key": FOUNDRY_API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        response = client.messages.create(
            model=routing["model_id"],
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": routing["thinking_tokens"]},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        err = str(e)
        if any(x in err.lower() for x in ["thinking", "budget", "422"]):
            response = client.messages.create(
                model=routing["model_id"],
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise

    text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    usage = {
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read":    getattr(response.usage, "cache_read_input_tokens", 0),
    }
    return text, usage


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli_loop():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      Laya → Claude Router  (type 'exit')         ║")
    print("║      'stats' → session summary                   ║")
    print("╚══════════════════════════════════════════════════╝\n")

    while True:
        try:
            prompt = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break
        if prompt.lower() == "stats":
            _print_session_stats()
            continue

        print("\n[laya] routing …", end=" ", flush=True)
        routing = route(prompt)
        print(f"→ {routing['model_key'].upper()} / effort={routing['effort']}")
        print(f"  {routing['rationale']}\n")

        print("[claude] thinking …\n")
        text, usage = call_claude(prompt, routing)

        print("─" * 60)
        print(text)
        print("─" * 60)

        if usage:
            comparison = token_comparison(routing, usage)
            update_session(routing, usage)

            print(f"\n[tokens used — {routing['model_key']} ✓ chosen by Laya]")
            print(f"  {'Model':<12} {'Input':>8} {'Output':>8} {'Est. Cost':>12}  {'':>4}")
            print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*12}")
            for mk, row in comparison.items():
                marker = " ← Laya" if row["chosen"] else ""
                print(f"  {mk:<12} {row['input_tokens']:>8} {row['output_tokens']:>8} "
                      f"  ${row['cost_usd']:>9.6f}{marker}")

            saved = _cost("opus", usage["input_tokens"], usage["output_tokens"]) - \
                    _cost(routing["model_key"], usage["input_tokens"], usage["output_tokens"])
            if saved > 0:
                print(f"\n  💰 Saved ~${saved:.6f} vs routing everything to Opus")
            print(f"\n  Session: {SESSION['requests']} requests · "
                  f"${SESSION['actual_cost']:.4f} spent · "
                  f"${SESSION['saved_cost']:.4f} saved\n")
        else:
            print()


def _print_session_stats():
    print("\n╔══════════════════════════════════════════╗")
    print("║            Session Summary               ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Requests routed : {SESSION['requests']}")
    print(f"  Total input     : {SESSION['total_in']:,} tokens")
    print(f"  Total output    : {SESSION['total_out']:,} tokens")
    print(f"  Actual cost     : ${SESSION['actual_cost']:.4f}")
    print(f"  Worst-case cost : ${SESSION['worst_cost']:.4f}  (if all → Opus)")
    print(f"  Total saved     : ${SESSION['saved_cost']:.4f}")
    print(f"\n  Model breakdown:")
    for mk, cnt in SESSION["model_counts"].items():
        pct = (cnt / SESSION["requests"] * 100) if SESSION["requests"] else 0
        bar = "█" * int(pct / 5)
        print(f"    {mk:<8} {cnt:>4} requests  {pct:>5.1f}%  {bar}")
    print()


# ── web UI ────────────────────────────────────────────────────────────────────

WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Laya → Claude Router</title>
<style>
  :root {
    --bg:      #0d0d0f;
    --surface: #16161a;
    --surface2:#1e1e26;
    --border:  #2a2a35;
    --accent:  #7c6af7;
    --accent2: #a78bfa;
    --text:    #e8e8f0;
    --muted:   #6b6b80;
    --sonnet:  #34d399;
    --opus:    #60a5fa;
    --fable:   #f472b6;
    --low:     #fbbf24;
    --medium:  #fb923c;
    --high:    #f87171;
    --green:   #34d399;
    --radius:  10px;
    --font:    'Inter', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── header ── */
  header {
    width: 100%;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px;
  }
  header h1 { font-size: 16px; font-weight: 600; letter-spacing: -0.02em; }

  /* ── session bar ── */
  #session-bar {
    margin-left: auto;
    display: flex;
    gap: 20px;
    font-size: 12px;
    color: var(--muted);
  }
  .stat-item { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }
  .stat-val  { font-size: 13px; font-weight: 600; color: var(--text); }
  .stat-val.green { color: var(--green); }

  /* ── layout ── */
  main {
    width: 100%;
    max-width: 900px;
    padding: 24px 20px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  #history { display: flex; flex-direction: column; gap: 24px; }
  .turn    { display: flex; flex-direction: column; gap: 10px; }

  /* ── bubbles ── */
  .bubble-user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    padding: 10px 16px;
    border-radius: var(--radius) var(--radius) 2px var(--radius);
    max-width: 72%;
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
  }
  .bubble-claude {
    align-self: flex-start;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 18px;
    border-radius: 2px var(--radius) var(--radius) var(--radius);
    max-width: 100%;
    font-size: 14px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── routing badges ── */
  .routing-badge {
    align-self: flex-start;
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    font-size: 11px;
  }
  .badge {
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
  }
  .badge-sonnet { background:#0d2e22; color:var(--sonnet); border-color:#1a4a36; }
  .badge-opus   { background:#0d1e3a; color:var(--opus);   border-color:#1a3060; }
  .badge-fable  { background:#2e0d22; color:var(--fable);  border-color:#4a1a36; }
  .badge-low    { background:#2e220d; color:var(--low);    border-color:#4a3a1a; }
  .badge-medium { background:#2e1a0d; color:var(--medium); border-color:#4a2a1a; }
  .badge-high   { background:#2e0d0d; color:var(--high);   border-color:#4a1a1a; }
  .badge-neutral{ background:var(--surface); color:var(--muted); border-color:var(--border); }

  /* ── token comparison table ── */
  .token-panel {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    font-size: 12px;
    align-self: flex-start;
    width: 100%;
  }
  .token-panel-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
  }
  .token-table {
    width: 100%;
    border-collapse: collapse;
  }
  .token-table th {
    text-align: right;
    color: var(--muted);
    font-weight: 500;
    padding: 3px 8px;
    font-size: 11px;
  }
  .token-table th:first-child { text-align: left; }
  .token-table td {
    text-align: right;
    padding: 4px 8px;
    color: var(--text);
    border-top: 1px solid var(--border);
  }
  .token-table td:first-child { text-align: left; font-weight: 600; }
  .token-table tr.chosen td { background: #1a1a2e; }
  .chosen-tag {
    display: inline-block;
    margin-left: 6px;
    font-size: 10px;
    padding: 1px 6px;
    background: var(--accent);
    color: #fff;
    border-radius: 10px;
    font-weight: 600;
  }
  .savings-line {
    margin-top: 10px;
    font-size: 11.5px;
    color: var(--green);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .bar-wrap { flex: 1; background: var(--border); border-radius: 4px; height: 5px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--green); border-radius: 4px; transition: width .4s; }

  /* ── input ── */
  .input-row {
    display: flex;
    gap: 10px;
    position: sticky;
    bottom: 0;
    padding: 16px 0 8px;
    background: linear-gradient(transparent, var(--bg) 30%);
  }
  textarea {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    padding: 12px 16px;
    font-family: var(--font);
    font-size: 14px;
    resize: none;
    outline: none;
    min-height: 52px;
    max-height: 200px;
    transition: border-color .2s;
  }
  textarea:focus { border-color: var(--accent); }
  #send {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    padding: 0 22px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .15s;
    min-width: 80px;
  }
  #send:disabled { opacity: .45; cursor: not-allowed; }
  #send:hover:not(:disabled) { opacity: .85; }
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #ffffff44; border-top-color: #fff;
    border-radius: 50%; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #f87171; font-size: 13px; }
</style>
</head>
<body>
<header>
  <div class="logo">L</div>
  <h1>Laya → Claude Router</h1>
  <div id="session-bar">
    <div class="stat-item">
      <span class="stat-label">Requests</span>
      <span class="stat-val" id="s-requests">0</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Tokens in / out</span>
      <span class="stat-val" id="s-tokens">0 / 0</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Spent</span>
      <span class="stat-val" id="s-spent">$0.0000</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Saved vs Opus</span>
      <span class="stat-val green" id="s-saved">$0.0000</span>
    </div>
  </div>
</header>
<main>
  <div id="history"></div>
  <div class="input-row">
    <textarea id="prompt" rows="1" placeholder="Write your prompt… (Laya picks the model)" autofocus></textarea>
    <button id="send">Send</button>
  </div>
</main>

<script>
const historyEl = document.getElementById('history');
const promptEl  = document.getElementById('prompt');
const sendBtn   = document.getElementById('send');

// session accumulators
let sRequests = 0, sTotalIn = 0, sTotalOut = 0, sSpent = 0, sSaved = 0;

const MODEL_COSTS = {
  sonnet: { input: 3.00,  output: 15.00 },
  opus:   { input: 15.00, output: 75.00 },
  fable:  { input: 15.00, output: 75.00 },
};

function costOf(mk, inp, out) {
  const c = MODEL_COSTS[mk] || MODEL_COSTS.opus;
  return (inp * c.input + out * c.output) / 1_000_000;
}

promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
sendBtn.addEventListener('click', submit);
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + 'px';
});

function badge(cls, text) {
  return `<span class="badge ${cls}">${escHtml(text)}</span>`;
}

function buildTokenPanel(data) {
  const { model_key, usage, comparison } = data;
  const inp = usage.input_tokens, out = usage.output_tokens;
  const worstCost = costOf('opus', inp, out);
  const actualCost = costOf(model_key, inp, out);
  const saved = Math.max(0, worstCost - actualCost);
  const savePct = worstCost > 0 ? (saved / worstCost * 100) : 0;

  let rows = '';
  for (const [mk, row] of Object.entries(comparison)) {
    const chosen = row.chosen;
    const tag = chosen ? '<span class="chosen-tag">✓ Laya</span>' : '';
    rows += `
      <tr class="${chosen ? 'chosen' : ''}">
        <td>${mk}${tag}</td>
        <td>${row.input_tokens.toLocaleString()}</td>
        <td>${row.output_tokens.toLocaleString()}</td>
        <td>$${row.cost_usd.toFixed(6)}</td>
      </tr>`;
  }

  const saveLine = saved > 0
    ? `<div class="savings-line">
        💰 Saved ~$${saved.toFixed(6)} vs Opus
        <div class="bar-wrap"><div class="bar-fill" style="width:${savePct.toFixed(1)}%"></div></div>
        ${savePct.toFixed(0)}%
       </div>`
    : '';

  return `
    <div class="token-panel">
      <div class="token-panel-title">Token usage · model comparison</div>
      <table class="token-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Input</th>
            <th>Output</th>
            <th>Est. cost</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      ${saveLine}
    </div>`;
}

function updateSessionBar() {
  document.getElementById('s-requests').textContent = sRequests;
  document.getElementById('s-tokens').textContent =
    `${sTotalIn.toLocaleString()} / ${sTotalOut.toLocaleString()}`;
  document.getElementById('s-spent').textContent = `$${sSpent.toFixed(4)}`;
  document.getElementById('s-saved').textContent = `$${sSaved.toFixed(4)}`;
}

async function submit() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  sendBtn.disabled = true;
  sendBtn.innerHTML = '<span class="spinner"></span>';
  promptEl.value = '';
  promptEl.style.height = 'auto';

  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `<div class="bubble-user">${escHtml(prompt)}</div>`;
  historyEl.appendChild(turn);
  turn.scrollIntoView({ behavior: 'smooth', block: 'end' });

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();

    if (data.error) {
      turn.innerHTML += `<div class="bubble-claude error">${escHtml(data.error)}</div>`;
    } else {
      const mb = data.model_key, eb = data.effort;
      const inp = data.usage.input_tokens, out = data.usage.output_tokens;

      // update session
      sRequests++;
      sTotalIn  += inp;
      sTotalOut += out;
      sSpent    += costOf(mb, inp, out);
      sSaved    += Math.max(0, costOf('opus', inp, out) - costOf(mb, inp, out));
      updateSessionBar();

      turn.innerHTML += `
        <div class="routing-badge">
          ${badge('badge-' + mb, mb.toUpperCase())}
          ${badge('badge-' + eb, 'effort: ' + eb)}
          ${badge('badge-neutral', data.domain)}
          ${badge('badge-neutral', 'complexity ' + data.complexity.toFixed(1) + '/2')}
        </div>
        <div class="bubble-claude">${escHtml(data.reply)}</div>
        ${buildTokenPanel(data)}
      `;
    }
  } catch (err) {
    turn.innerHTML += `<div class="bubble-claude error">Network error: ${escHtml(String(err))}</div>`;
  }

  sendBtn.disabled = false;
  sendBtn.textContent = 'Send';
  turn.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


# ── flask app ─────────────────────────────────────────────────────────────────

def run_web(port: int = 7860):
    try:
        from flask import Flask, request, jsonify, Response
    except ImportError:
        print("[error] Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(WEB_HTML, mimetype="text/html")

    @app.route("/chat", methods=["POST"])
    def chat():
        data   = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "empty prompt"}), 400
        try:
            routing      = route(prompt)
            reply, usage = call_claude(prompt, routing)
            comparison   = token_comparison(routing, usage)
            update_session(routing, usage)
            return jsonify({
                "model_key":  routing["model_key"],
                "model_id":   routing["model_id"],
                "effort":     routing["effort"],
                "domain":     routing["domain"],
                "complexity": routing["complexity"],
                "rationale":  routing["rationale"],
                "reply":      reply,
                "usage":      usage or {"input_tokens": 0, "output_tokens": 0, "cache_read": 0},
                "comparison": comparison,
                "session":    SESSION,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/session")
    def session():
        return jsonify(SESSION)

    print(f"\n[web] http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laya → Claude Router")
    parser.add_argument("--web",  action="store_true", help="Launch web UI")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port (default 7860)")
    args = parser.parse_args()

    if args.web:
        run_web(args.port)
    else:
        cli_loop()
