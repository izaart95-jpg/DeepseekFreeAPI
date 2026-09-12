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
- **Real model registry** — `deepseek-v4.1-flash` (default, thinking + web search); the id maps to the actual upstream request (`model_type: "default"` — the v4.1 backend has no other model class)
- Cookie management (loads `cookies.json`)
- Streaming and non-streaming responses
- Threaded conversation support
- **Stealth session-reuse flow (default)** — with history disabled, all stateless traffic rides ONE lazily-created chat session; every request re-edits the same user message through `POST /chat/edit_message` (DeepSeek's own edit feature), so no conversation context ever accumulates and there is no per-request session create/delete churn — the traffic pattern that was getting accounts suspended for "bot-like behavior". The session rotates (deleted upstream, one fresh session lazily created) only when its edit budget (~6 edits per message) runs out. `--legacy-pool` / `SESSION_FLOW=pool` restores the old pre-warmed batch flow, `--sync-mode` / `SESSION_FLOW=sync` the legacy one-session-per-request flow
- **Graceful shutdown** — CTRL+C drains in-flight requests, then clears every remaining session on DeepSeek before exiting (a second CTRL+C force-exits)
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

### Stateless session flow (history disabled)

By default, stateless traffic (`/history` disabled) runs through the **stealth reuse** flow:

- Nothing is created at startup. The first request lazily creates ONE chat session — indistinguishable from a user opening a new chat.
- Every following request POSTs `chat/edit_message`, re-editing the *first* user message of that session. Each edit replaces the previous prompt, so the conversation stays a single root exchange forever: the model has no memory of earlier requests (exactly the stateless semantics `/history=false` promises) while the account never sees session create/delete churn per request.
- Requests are serialized (one generation in flight, like a browser user); concurrent requests queue briefly behind the current one.
- DeepSeek caps edits of one message (observed budget: 6). When the budget runs out — the stream reports `ban_edit`, or an edit is rejected — the used session is deleted upstream and the **next** request lazily creates a fresh one. Again: a single, human-paced call, never a burst.
- If an edit fails because the session no longer exists server-side, the flow rotates automatically and retries on a fresh session, so a stale session can never poison later requests.

```bash
# the stealth flow is the default; nothing to configure
SESSION_FLOW=stealth   # explicit (same as default)
```

#### Legacy flows (backward compatibility)

The two older behaviors remain fully available via flags or `SESSION_FLOW`:

- **`--legacy-pool`** (or `SESSION_FLOW=pool`) — the pre-warmed session batch: a standing batch of sessions is created at startup (boot burst), each request takes one, and each consumed session is deleted + replaced after its response completes. Tunables: `SESSION_POOL_SIZE=5`, `SESSION_ACQUIRE_TIMEOUT=10`.
- **`--sync-mode`** (or `SESSION_FLOW=sync`, `SYNC_MODE=true`) — the original synchronous flow: every request creates its own session, completes, then the session is garbage-collected.

```bash
./deepseek-proxy --legacy-pool          # or SESSION_FLOW=pool
./deepseek-proxy --sync-mode            # or SESSION_FLOW=sync / SYNC_MODE=true

# legacy pool tuning (pool flow only)
SESSION_POOL_SIZE=5            # standing ready-session batch size
SESSION_ACQUIRE_TIMEOUT=10     # seconds to wait for a pooled session (0 = forever)
```

> ⚠️ The legacy pool flow recreates the exact traffic pattern DeepSeek was flagging (boot-time burst of session creations + per-request create/delete churn). It is kept only for backward compatibility.

### Graceful shutdown

Pressing CTRL+C (or sending SIGTERM) stops the proxy respectfully: it stops accepting new connections, lets in-flight responses finish (10s drain deadline), prints `clearing all sessions...`, deletes every remaining session on DeepSeek (the reuse flow's live session and any pooled sessions) so nothing is left behind, and only then exits. A second CTRL+C force-exits immediately.

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

**Streaming is UTF-8 safe** — the interceptor holds back a small window of un-flushed text (so a tool-call marker split across chunks can't leak). That window used to be cut at a raw byte offset, which could slice a multi-byte character in two; each half then went through JSON encoding, which replaces invalid UTF-8 with U+FFFD — CJK text came out garbled (乱码) in streaming mode while non-streaming stayed clean. The cut now lands on a rune boundary, so every emitted delta is valid UTF-8 (covered by regression tests in `internal/dsproxy/agent_utf8_test.go`).

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

> 🧹 **Stateless semantics:** with history disabled, requests are stateless by design — the stealth reuse flow implements this via `chat/edit_message` (each request replaces the previous prompt on a shared session, so no context accumulates), while the legacy flows used throwaway sessions deleted after each request (`POST /chat_session/delete`). Sessions created in history-enabled mode are never deleted, and rotating via `POST /new` also collects the session it replaces.

### `POST /new` — Create a new session

```bash
curl -X POST http://localhost:3000/new \
  -H "Authorization: Bearer Waguri"
```

### Models

The proxy serves a single model, `deepseek-v4.1-flash` (the DeepSeek v4.1 backend no longer has model classes — `model_type` is always `"default"`). The id is real configuration: it selects the `model_type` sent to DeepSeek's `/chat/completion` and gates what the request may use.

| model | sent upstream as | web search | reasoning (`reasoning` / `reasoning_effort`) |
|---|---|---|---|
| `deepseek-v4.1-flash` *(default)* | `"model_type": "default"` | ✅ | ✅ |

Rules:

- Omitting `model` resolves to `deepseek-v4.1-flash`.
- Any other model id (including the retired `deepseek-v4-flash` / `deepseek-v4-pro`) is rejected with `400 model_not_found`.
- When thinking is enabled, the reasoning trace is returned separately as `reasoning_content` (streaming: `delta.reasoning_content`; non-streaming: `message.reasoning_content`) — it never mixes into `content`.

### `POST /v1/chat/completions` — Chat completions (OpenAI format)

Thinking mode stays **off** unless the request payload contains `"reasoning": {"enabled": true}` or a `"reasoning_effort"` value — the model name alone never enables it.

**Non-streaming with thinking + search:**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "What is the latest news about AI?"}],
    "reasoning": {"enabled": true},
    "search": true,
    "stream": false
  }'
```

**Streaming with thinking:**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash",
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
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "My name is John"}],
    "search": false,
    "stream": false
  }'

# Step 3: Follow-up — model should remember the name
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer Waguri" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash",
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
