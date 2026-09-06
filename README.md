# DeepSeek Free API

> A pure Go proxy for DeepSeek with Cloudflare bypass and OpenAI-compatible interface — no paid API key required.

![Go](https://img.shields.io/badge/Go-1.26+-blue?style=flat-square)
![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green?style=flat-square)
![Cloudflare Bypass](https://img.shields.io/badge/Cloudflare-Bypass-purple?style=flat-square)
![Free API](https://img.shields.io/badge/API-Free-orange?style=flat-square)

---

## Features

- Full DeepSeek API implementation (pure Go, zero runtime dependencies)
- Cloudflare protection detection with automatic retry
- Proof of Work (PoW) challenge solving — WASM run in-process via wazero
- OpenAI-compatible proxy server
- **Real model registry** — `deepseek-v4-flash` (default) and `deepseek-v4-pro`; the selected model changes the actual upstream request (`model_type: "default"` vs `"expert"`) and gates capabilities (pro is reasoning-only, no web search)
- Cookie management (loads `cookies.json`)
- Streaming and non-streaming responses
- Threaded conversation support
- **Session garbage collector** — with history disabled, every request runs on a throwaway chat session that is deleted on DeepSeek right after use, so session IDs never accumulate on the account
- **Async session-pool mode (default)** — a standing batch of 3 pre-made sessions is kept warm at all times; stateless requests grab one instantly instead of paying per-request creation latency, and each consumed session is deleted upstream + replaced the moment its response is fully processed. `--sync-mode` restores the legacy synchronous flow
- **Graceful shutdown** — CTRL+C drains in-flight requests, then clears every remaining pooled session on DeepSeek before exiting (a second CTRL+C force-exits)
- **Agent mode** (`--agent-mode` / `AGENT_MODE=true`) — OpenAI function/tool calling translated into a single role-tagged prompt; model tool-call blocks are parsed back into OpenAI `tool_calls`
- **Debug mode** (`--debug` / `DEBUG=true`) — prints every request/response headers and bodies in both directions, PoW challenges, SSE frames and session IDs

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/izaart95-jpg/DeepseekFreeAPI.git DeepRouter
cd DeepRouter
go mod tidy
```

### 2. Build (requires Go 1.26+)

```bash
go build -o deepseek-proxy .
```

### 3. Obtain your DeepSeek token

1. Navigate to [chat.deepseek.com](https://chat.deepseek.com) and sign in
2. Open browser DevTools (F12) and go to the Console tab
3. Run the following snippet:

```js
JSON.parse(localStorage.getItem("userToken")).value
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your token. The proxy loads `.env` automatically at startup — first from its working directory, then from the directory next to the binary. Variables already present in the environment always win, so you can still export them directly:

```bash
# Linux / macOS
export DEEPSEEK_TOKEN=your_token_here

# Windows (PowerShell)
$env:DEEPSEEK_TOKEN="your_token_here"
```

---

## Running

### OpenAI-compatible proxy server

```bash
# requires Go 1.26+
go build -o deepseek-proxy .
DEEPSEEK_TOKEN=<token> ./deepseek-proxy proxy

# debug mode (verbose HTTP dumps)
DEEPSEEK_TOKEN=<token> DEBUG=true ./deepseek-proxy proxy
# or: ./deepseek-proxy --debug proxy

# agent mode (OpenAI tool calling via prompt protocol)
DEEPSEEK_TOKEN=<token> AGENT_MODE=true ./deepseek-proxy proxy
# or: ./deepseek-proxy --agent-mode proxy
```

### Sync vs async session flow

By default, stateless traffic (`/history` disabled) runs through the **async** flow:

- At startup the proxy **pre-makes a standing batch of 5 chat sessions** so completion requests never wait on per-request session creation.
- Each request takes a ready session from the batch instantly; multiple concurrent requests are served in parallel up to the batch size.
- Only after a response has been **fully written and processed** is that consumed session deleted upstream (`POST /chat_session/delete`) and a replacement created immediately — the batch refills itself for as long as the app runs.
- If a burst exhausts the batch, extra requests wait up to `SESSION_ACQUIRE_TIMEOUT` seconds (default 10) and then create a session directly instead of stalling; those still go through the garbage collector afterwards.

```bash
# tune the async flow (optional)
SESSION_POOL_SIZE=5            # standing ready-session batch size
SESSION_ACQUIRE_TIMEOUT=10     # seconds to wait for a pooled session (0 = forever)
```

For backward compatibility, `--sync-mode` (or `SYNC_MODE=true`) restores the legacy synchronous flow: every request creates its own session first, then completes, then the session is garbage-collected — one request at a time per client, no pre-warming.

### Graceful shutdown

Pressing CTRL+C (or sending SIGTERM) stops the proxy respectfully: it stops accepting new connections, lets in-flight responses finish (10s drain deadline), prints `clearing all sessions...`, deletes every remaining pooled session on DeepSeek so nothing is left behind, and only then exits. A second CTRL+C force-exits immediately.

### Agent mode

Enable with `--agent-mode` or `AGENT_MODE=1|true|yes|on`. The proxy rewrites the whole OpenAI `messages` array (system/user/assistant/`tool` roles) plus the `tools` definitions into one role-tagged prompt ending in a `[TOOL CONTRACT]`. When the model wants to call a tool it emits a block like:

```
<<<TOOL_CALL>>>
{"name":"get_weather","arguments":{"city":"Paris"}}
<<<END_TOOL_CALL>>>
```

These blocks never reach your client as text:

- **Non-streaming** → parsed into OpenAI `tool_calls` on the assistant message, `finish_reason: "tool_calls"`; send the tool result back as a `role:"tool"` message.
- **Streaming** → content deltas flow normally, each parsed call becomes a `delta.tool_calls` chunk, and the stream ends with `finish_reason: "tool_calls"`.

The prompt pins the exact `{"name","arguments"}` schema, and the parser additionally tolerates the flat payload shapes models sometimes emit anyway (e.g. `{"tool":"bash","command":"ls"}` — tool name under `tool`/`tool_name`/`function`, parameters as the remaining top-level keys, or `parameters`/`args` in place of `arguments`), folding them back into proper `tool_calls` instead of leaking the block to the client as text.

**Anti-loop guardrails** — long agent sessions used to drift into re-issuing identical tool calls (or thinking indefinitely) because the prompt replayed history but never told the model which calls were already made. The prompt now carries an `<already_called>` section: a deduplicated, order-preserving list of every call already issued, placed late in the prompt (right before `<current_task>`) where recency weight is highest, plus `<system>` rules (PROGRESS mandate, NEVER REPEAT, task-done → plain-text answer) and a `NO REPEATS` line in the `<output_rules>` final reminder — so the model always has a hard, current state to check against before emitting a call.

Web search is forced off in this mode so tool answers stay deterministic.

### Debug mode

Enable with `--debug` or `DEBUG=1|true|yes|on`. Every exchange is printed to stderr: client requests (method/path/headers/body), upstream requests to DeepSeek (URL/headers/cookies/body), upstream responses (status/headers/body), streaming SSE frames both ways, PoW challenge + solution, and parsed agent tool calls. Bodies longer than 4 KB are truncated.

> ⚠️ Debug logs contain your DeepSeek token and cookies unredacted — don't paste them publicly.

---

## API Reference

The proxy runs at `http://localhost:3000`. All endpoints require the bearer token `Waguri`.

### `POST /history` — Toggle conversation history

```bash
# Enable
curl -X POST http://localhost:3000/history \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{"enable": true}'

# Disable
curl -X POST http://localhost:3000/history \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{"enable": false}'
```

> 🧹 **Garbage collector:** with history disabled, each request gets a throwaway chat session. As soon as the response is done, the proxy asynchronously deletes that session upstream (`POST /chat_session/delete`) — so your DeepSeek account doesn't fill up with dead session IDs. In the default async mode the deleted session is instantly replaced from a pre-made batch; sessions created in history-enabled mode are never deleted, and rotating via `POST /new` also collects the session it replaces.

### `POST /new` — Create a new session

```bash
curl -X POST http://localhost:3000/new \
  -H "Authorization: Bearer Waguri"
```

### Models

The proxy serves two models. The choice is real configuration: it selects the `model_type` sent to DeepSeek's `/chat/completion` and gates what the request may use.

| model | sent upstream as | web search | reasoning (`reasoning` / `reasoning_effort`) |
|---|---|---|---|
| `deepseek-v4-flash` *(default)* | `"model_type": "default"` | ✅ | ✅ |
| `deepseek-v4-pro` | `"model_type": "expert"` | ❌ refused (400) | ✅ |

Rules:

- Omitting `model` resolves to `deepseek-v4-flash`.
- Any other model id is rejected with `400 model_not_found`.
- `deepseek-v4-pro` with `"search": true` is rejected with `400 model_capability` instead of being silently downgraded.
- When thinking is enabled, the reasoning trace is returned separately as `reasoning_content` (streaming: `delta.reasoning_content`; non-streaming: `message.reasoning_content`) — it never mixes into `content`.

### `POST /v1/chat/completions` — Chat completions (OpenAI format)

Thinking mode stays **off** unless the request payload contains `"reasoning": {"enabled": true}` or a `"reasoning_effort"` value — the model name alone never enables it.

**Non-streaming with thinking + search (flash):**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is the latest news about AI?"}],
    "reasoning": {"enabled": true},
    "search": true,
    "stream": false
  }'
```

**Streaming with the reasoning model (pro):**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "reasoning_effort": "high",
    "stream": true
  }'
```

### Multi-turn conversation example

Enable history first, then send messages sequentially — the model retains context across requests.

```bash
# Step 1: Enable history
curl -X POST http://localhost:3000/history \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{"enable": true}'

# Step 2: First message
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "My name is John"}],
    "search": false,
    "stream": false
  }'

# Step 3: Follow-up — model should remember the name
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is my name?"}],
    "search": false,
    "stream": false
  }'
```

> The second request should return "John" — history is preserved across calls.

---

## Development

The application lives in `internal/dsproxy` (`main.go` is a thin entry point); tests live in `tests/`:

```bash
go test ./...
```

---

## Acknowledgements

- [github.com/xtekky/deepseek4free](https://github.com/xtekky/deepseek4free)
