package dsproxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ProxyServer is the OpenAI-compatible proxy (port of proxy.py).
type ProxyServer struct {
	log      *log.Logger
	proxyKey string

	mu     sync.Mutex
	api    *DeepSeekAPI
	apiErr error
	apiSet bool

	hl         sync.Mutex
	histChatID string
	histParID  any
	useHistory bool

	// sessionFlow selects how stateless (history=false) requests get a chat
	// session upstream. Exactly one of the three mode handles is non-nil.
	sessionFlow SessionFlow

	// activeLease is the reuse lease whose stream events are currently in
	// flight (stealth flow). Requests are serialized by the reuse manager,
	// so one at a time owns this.
	activeLease *Lease

	// poolWait bounds how long a request waits for a pooled session before
	// falling back to creating one directly (0 waits forever). Legacy pool
	// flow only.
	poolWait time.Duration
}

// SessionFlow is the per-mode state for stateless session handling.
type SessionFlow struct {
	// Stealth is the anti-detection reuse flow (default): one lazily created
	// session, reused via /chat/edit_message, rotated after the edit budget.
	Stealth *ReuseManager
	// Pool is the legacy pre-warmed session pool (--legacy-pool).
	Pool *SessionPool
	// Sync is the legacy one-session-per-request flow (--sync-mode).
	Sync bool
}

// NewProxyServer builds the server; AttachReuseManager / AttachSessionPool
// select the stateless flow afterwards (exactly one, set once by Run).
func NewProxyServer(logger *log.Logger, proxyKey string) *ProxyServer {
	return &ProxyServer{log: logger, proxyKey: proxyKey, poolWait: defaultPoolWait}
}

// AttachReuseManager switches the history=false path to the stealth reuse
// flow. Must be called before the server starts serving.
func (s *ProxyServer) AttachReuseManager(m *ReuseManager) {
	s.sessionFlow.Stealth = m
}

// AttachSessionPool switches the history=false path to the legacy pre-warmed
// session-pool flow. Must be called before the server starts serving.
func (s *ProxyServer) AttachSessionPool(p *SessionPool) {
	s.sessionFlow.Pool = p
}

// EnableSyncFlow restores the legacy synchronous per-request flow.
func (s *ProxyServer) EnableSyncFlow() {
	s.sessionFlow.Sync = true
}

func (s *ProxyServer) getAPI() (*DeepSeekAPI, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.apiSet {
		return s.api, s.apiErr
	}
	token := os.Getenv("DEEPSEEK_TOKEN")
	if token == "" {
		s.apiErr = errors.New("DEEPSEEK_TOKEN not set.\n  export DEEPSEEK_TOKEN=<chat.deepseek.com localStorage -> userToken -> value>")
	} else {
		s.api, s.apiErr = NewDeepSeekAPI(token)
		if s.apiErr == nil {
			s.log.Printf("DeepSeekAPI ready")
		}
	}
	s.apiSet = true
	return s.api, s.apiErr
}

func (s *ProxyServer) newHist() (string, error) {
	api, err := s.getAPI()
	if err != nil {
		return "", err
	}
	cid, err := api.CreateChatSession(context.Background())
	if err != nil {
		return "", err
	}
	s.hl.Lock()
	old := s.histChatID
	s.histChatID, s.histParID = cid, nil
	s.hl.Unlock()
	// The replaced history session is no longer referenced anywhere; collect
	// it so /new rotations don't leak sessions server-side either.
	if old != "" && old != cid {
		s.gcSessions("rotated-out-history", old)
	}
	s.log.Printf("New history session: %s", cid)
	return cid, nil
}

func (s *ProxyServer) getHist() (string, any) {
	s.hl.Lock()
	defer s.hl.Unlock()
	return s.histChatID, s.histParID
}

func (s *ProxyServer) setHistPar(mid any) {
	s.hl.Lock()
	defer s.hl.Unlock()
	s.histParID = mid
}

func (s *ProxyServer) setUseHistory(v bool) {
	s.hl.Lock()
	defer s.hl.Unlock()
	s.useHistory = v
}

func (s *ProxyServer) getUseHistory() bool {
	s.hl.Lock()
	defer s.hl.Unlock()
	return s.useHistory
}

// gcSessions asynchronously deletes used-up chat sessions on DeepSeek so
// their IDs don't accumulate on the account. This mirrors the Kimi reference's
// deleteChat-after-use behavior (references/main.go): sessions are temporary
// by design and are removed right after their last use, in a background
// goroutine so response latency is unaffected. Deletion is best-effort —
// failures are logged and otherwise ignored.
func (s *ProxyServer) gcSessions(reason string, sessionIDs ...string) {
	ids := make([]string, 0, len(sessionIDs))
	for _, id := range sessionIDs {
		if id != "" {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return
	}
	go func() {
		api, err := s.getAPI()
		if err != nil {
			s.log.Printf("[GC:%s] skip delete %v: %v", reason, ids, err)
			return
		}
		// Own context: the triggering request may already be gone by now.
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := api.DeleteChatSession(ctx, ids...); err != nil {
			s.log.Printf("[GC:%s] failed to delete session(s) %v: %v", reason, ids, err)
			return
		}
		s.log.Printf("[GC:%s] deleted chat session(s): %v", reason, ids)
	}()
}

func (s *ProxyServer) checkAuth(r *http.Request) bool {
	if s.proxyKey == "" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.proxyKey
}

func (s *ProxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cors := func() {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	}

	if r.Method == http.MethodOptions {
		cors()
		w.WriteHeader(200)
		return
	}
	cors()

	if debugMode {
		debugDumpClientRequest(r, nil) // headers only; handlers dump the body
	}

	if !s.checkAuth(r) {
		writeError(w, "Bad token", "authentication_error", "invalid_api_key", 401)
		return
	}

	path := r.URL.Path
	if r.Method == http.MethodGet {
		switch path {
		case "/v1/models", "/models":
			writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": ProxyModels})
			return
		case "/history":
			q := r.URL.Query()
			v := q.Get("enable")
			if v == "" {
				v = q.Get("value")
			}
			enabled := strings.EqualFold(v, "true")
			s.setUseHistory(enabled)
			cid, _ := s.getHist()
			writeJSON(w, http.StatusOK, map[string]any{"use_history": enabled, "chat_id": cid})
			return
		}
		writeError(w, "Not Found", "invalid_request_error", "not_found", 404)
		return
	}

	if r.Method == http.MethodPost {
		switch path {
		case "/new":
			if _, err := s.getAPI(); err != nil {
				writeError(w, err.Error(), "authentication_error", "missing_token", 401)
				return
			}
			cid, err := s.newHist()
			if err != nil {
				writeError(w, err.Error(), "upstream_error", nil, 502)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"message": "New session", "chat_id": cid})
			return
		case "/history":
			var body map[string]any
			if err := decodeBody(w, r, &body); err != nil {
				return
			}
			var enabled bool
			if v, ok := body["enable"]; ok {
				enabled = toBool(v)
			} else if v, ok := body["value"]; ok {
				enabled = toBool(v)
			}
			s.setUseHistory(enabled)
			writeJSON(w, http.StatusOK, map[string]any{"use_history": enabled})
			return
		case "/v1/chat/completions", "/chat/completions":
			s.handleChat(w, r)
			return
		}
		writeError(w, "Not Found", "invalid_request_error", "not_found", 404)
		return
	}

	writeError(w, "Not Found", "invalid_request_error", "not_found", 404)
}

// chatMessage is one OpenAI-style message. Content stays raw so both string
// and typed-part arrays are accepted.
type chatMessage struct {
	Role       string              `json:"role"`
	Content    json.RawMessage     `json:"content"`
	ToolCallID string              `json:"tool_call_id,omitempty"`
	ToolCalls  []assistantToolCall `json:"tool_calls,omitempty"`
	Name       string              `json:"name,omitempty"`
}

// reasoningOptions mirrors an OpenAI-style "reasoning" object; only
// "enabled": true toggles thinking.
type reasoningOptions struct {
	Enabled bool `json:"enabled"`
}

// ChatRequest mirrors the OpenAI-style request body. "model" resolves against
// the registry in models.go — "deepseek-v4.1-flash" (default); anything else
// is a 400. Thinking is NOT inferred from the model name or legacy flags: it
// is enabled only when the payload carries "reasoning": {"enabled": true} or
// a "reasoning_effort" string.
type ChatRequest struct {
	Messages        []chatMessage     `json:"messages"`
	Model           string            `json:"model"`
	Stream          bool              `json:"stream"`
	Reasoning       *reasoningOptions `json:"reasoning,omitempty"`
	ReasoningEffort *string           `json:"reasoning_effort,omitempty"`
	Search          bool              `json:"search"`
	Tools           []openAITool      `json:"tools,omitempty"`
}

// ThinkingRequested reports whether the client payload opted into thinking:
// "reasoning": {"enabled": true} or any "reasoning_effort": "<string>".
func ThinkingRequested(req ChatRequest) bool {
	if req.Reasoning != nil && req.Reasoning.Enabled {
		return true
	}
	return req.ReasoningEffort != nil
}

func (s *ProxyServer) handleChat(w http.ResponseWriter, r *http.Request) {
	var body ChatRequest
	rawBody, err := decodeBodyRaw(w, r, &body)
	if err != nil {
		return
	}
	if debugMode {
		debugDumpClientRequest(r, rawBody)
	}

	// Resolve the requested model against the registry. The model is real
	// configuration: it decides the "model_type" sent upstream and gates the
	// capabilities the request may use.
	model, resErr := ResolveModel(body.Model)
	if resErr != nil {
		writeError(w, resErr.Error(), "invalid_request_error", "model_not_found", 400)
		return
	}
	stream := body.Stream
	// Thinking only turns on when explicitly requested via "reasoning" or
	// "reasoning_effort"; the model name never implies it.
	thinking := ThinkingRequested(body)
	search := body.Search

	prompt := ""
	if agentMode {
		// Agent mode folds every message plus the tool contract into one
		// role-aware prompt so the model cannot silently use tools outside
		// the caller's OpenAI tool contract (web search stays off).
		prompt = buildAgentPrompt(body.Messages, body.Tools)
		search = false
		debugf("agent prompt (%d chars):\n%s", len(prompt), prompt)
	} else if len(body.Messages) > 0 {
		raw := body.Messages[len(body.Messages)-1].Content
		var s string
		if json.Unmarshal(raw, &s) == nil {
			prompt = s
		} else {
			var parts []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(raw, &parts) == nil {
				texts := []string{}
				for _, p := range parts {
					if p.Type == "text" {
						texts = append(texts, p.Text)
					}
				}
				prompt = strings.Join(texts, "\n")
			}
		}
	}
	if prompt == "" {
		prompt = " "
	}

	// Capability gate: reject feature combinations the model does not support
	// before any upstream session or PoW work is spent.
	if capErr := model.ValidateCapabilities(search); capErr != nil {
		writeError(w, capErr.Error(), "invalid_request_error", "model_capability", 400)
		return
	}

	// Streaming needs a flusher; bail before touching any upstream state if
	// this connection can't stream (e.g. wrapped in a non-flushing writer).
	if stream {
		if _, ok := w.(http.Flusher); !ok {
			writeError(w, "Streaming unsupported", "server_error", nil, 500)
			return
		}
	}

	api, err := s.getAPI()
	if err != nil {
		writeError(w, err.Error(), "authentication_error", "missing_token", 401)
		return
	}

	// ── session acquisition (history=false path picks a flow) ──────────────
	var (
		chatID        string
		parID         any
		reuseLease    *Lease // stealth flow: the checked-out reusable session
		reuseIsEdit   bool   // stealth flow: true when the request rides an edit
		reusePromptID any    // stealth flow: message_id the edit targets
		pooled        bool   // legacy pool flow: retire via pool.Release
	)
	useHistory := s.getUseHistory()
	switch {
	case useHistory:
		chatID, parID = s.getHist()
		if chatID == "" {
			chatID, err = s.newHist()
			parID = nil
			if err != nil {
				writeError(w, err.Error(), "upstream_error", nil, 502)
				return
			}
		}
	case s.sessionFlow.Stealth != nil:
		lease, acqErr := s.sessionFlow.Stealth.Begin(r.Context())
		if acqErr != nil {
			if errors.Is(acqErr, ErrReuseShuttingDown) {
				writeError(w, "server is shutting down", "server_error", "shutting_down", 503)
				return
			}
			if r.Context().Err() != nil {
				return // client gone; nothing to answer
			}
			writeError(w, acqErr.Error(), "upstream_error", nil, 502)
			return
		}
		chatID = lease.SessionID()
		reuseLease = lease
		if mid := lease.MessageIDForEdit(); mid != nil {
			// The session already carries the first user message: this
			// request re-edits it instead of appending (the anti-bot reuse
			// path — no context accumulates, no session churn).
			reuseIsEdit = true
			reusePromptID = mid
		}
		// else: brand-new session — its first message rides /chat/completion
		// like a human opening a chat.
		parID = nil
	case s.sessionFlow.Pool != nil:
		// Legacy pool flow: take a pre-made session from the standing batch.
		id, acqErr := s.sessionFlow.Pool.Acquire(r.Context(), s.poolWait)
		switch {
		case acqErr == nil:
			chatID, pooled = id, true
			s.log.Printf("Pooled stateless session: %s (%d/%d ready)", chatID, len(s.sessionFlow.Pool.ready), s.sessionFlow.Pool.size)
		case errors.Is(acqErr, ErrPoolTimeout):
			chatID, err = api.CreateChatSession(context.Background())
			if err != nil {
				writeError(w, err.Error(), "upstream_error", nil, 502)
				return
			}
			s.log.Printf("Stateless session (pool busy, created on demand): %s", chatID)
		default: // ErrPoolClosing, or the request's client went away
			if errors.Is(acqErr, context.Canceled) || r.Context().Err() != nil {
				return // client gone; nothing to answer
			}
			writeError(w, "server is shutting down", "server_error", "shutting_down", 503)
			return
		}
		parID = nil
	default:
		chatID, err = api.CreateChatSession(context.Background())
		parID = nil
		if err != nil {
			writeError(w, err.Error(), "upstream_error", nil, 502)
			return
		}
		s.log.Printf("Stateless session: %s", chatID)
	}

	s.log.Printf("-> model=%-18s type=%-7s think=%-5v search=%-5v history=%-5v agent=%v",
		model.ID, model.Type, thinking, search, useHistory, agentMode)
	if reuseLease != nil {
		mode := "first completion"
		if reuseIsEdit {
			mode = "edit_message"
		}
		s.log.Printf("   reuse session %s: %s (edits left: %d)",
			chatID, mode, reuseLease.EditsRemaining())
	}

	// ── run the completion ─────────────────────────────────────────────
	// The wire-format writers below are shared by every flow; the runner
	// closure picks the upstream call (completion vs edit_message).
	var run upstreamRunner
	if reuseLease != nil {
		runner, err2 := s.reuseRunner(api, reuseLease, reuseIsEdit, reusePromptID, prompt, thinking, search, model)
		if err2 != nil {
			reuseLease.Abort()
			writeError(w, err2.Error(), "upstream_error", nil, 502)
			return
		}
		run = runner
		s.activeLease = reuseLease // observeStreamChunk routes events to it
	} else {
		params := ChatParams{
			ChatSessionID:   chatID,
			Prompt:          prompt,
			ParentMessageID: parID,
			ThinkingEnabled: thinking,
			SearchEnabled:   search,
			ModelType:       string(model.Type),
		}
		run = func(ctx context.Context, onChunk func(Chunk) bool) error {
			return api.ChatCompletion(ctx, params, onChunk)
		}
	}

	var streamErr error
	if stream {
		streamErr = s.streamResponse(w, r, run, model.ID)
	} else {
		streamErr = s.blockResponse(w, r, run, model.ID, prompt)
	}
	s.activeLease = nil

	// ── session retirement ──────────────────────────────────────────────
	// Stealth flow: Complete on success (consume one edit; rotate if the
	// budget is exhausted or the stream reported ban_edit), Abort otherwise.
	// A mid-stream client hangup (errWriteFailed) still counts as success —
	// the upstream edit completed; only its relay to the client died.
	// Legacy pool flow: pool-owned sessions are retired through the pool;
	// sessions created on demand and sync-mode sessions keep the legacy GC.
	if reuseLease != nil {
		switch {
		case streamErr == nil, errors.Is(streamErr, errWriteFailed):
			reuseLease.Complete()
		default:
			reuseLease.Abort()
		}
		return
	}
	if !useHistory && chatID != "" {
		if pooled {
			s.sessionFlow.Pool.Release(chatID)
		} else {
			s.gcSessions("stateless", chatID)
		}
	}
}

// upstreamRunner is one stateless completion against the DeepSeek backend:
// it streams chunks to onChunk (which returns true to stop early) and
// reports the terminal error. Both /chat/completion and /chat/edit_message
// implement it, so the OpenAI wire-format writers stay flow-agnostic.
type upstreamRunner func(ctx context.Context, onChunk func(Chunk) bool) error

// reuseRunner builds the upstream runner for the stealth reuse flow. The
// very first request on a fresh session uses /chat/completion (a human's
// first message); every later request re-edits the root message through
// /chat/edit_message (a human editing their own message — the sanctioned
// way to reuse a session without persisting context).
func (s *ProxyServer) reuseRunner(api *DeepSeekAPI, lease *Lease, isEdit bool, msgID any, prompt string, thinking, search bool, model Model) (upstreamRunner, error) {
	if !isEdit {
		params := ChatParams{
			ChatSessionID:   lease.SessionID(),
			Prompt:          prompt,
			ParentMessageID: nil,
			ThinkingEnabled: thinking,
			SearchEnabled:   search,
			ModelType:       string(model.Type),
		}
		return func(ctx context.Context, onChunk func(Chunk) bool) error {
			err := api.ChatCompletion(ctx, params, onChunk)
			if isSessionDeadError(err) {
				lease.MarkDead()
			}
			return err
		}, nil
	}
	if msgID == nil {
		return nil, APIError{Msg: "reuse: no message id to edit"}
	}
	params := EditChatParams{
		ChatSessionID:   lease.SessionID(),
		MessageID:       msgID,
		Prompt:          prompt,
		ThinkingEnabled: thinking,
		SearchEnabled:   search,
	}
	return func(ctx context.Context, onChunk func(Chunk) bool) error {
		err := api.EditChatMessage(ctx, params, onChunk)
		if isSessionDeadError(err) {
			lease.MarkDead()
		}
		return err
	}, nil
}

// ── helpers ──────────────────────────────────────────────────────────────────

// errWriteFailed marks a client-side write failure mid-stream (the client
// hung up): the response is unrecoverable, but the upstream run was healthy.
var errWriteFailed = errors.New("client went away mid-stream")

// streamResponse writes the OpenAI streaming wire format for one upstream
// run. It is flow-agnostic: the runner decides whether the upstream call is
// /chat/completion or /chat/edit_message. (handleChat pre-checked that the
// ResponseWriter can flush.)
func (s *ProxyServer) streamResponse(w http.ResponseWriter, r *http.Request, run upstreamRunner, model string) error {
	flusher := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "close")
	if debugMode {
		debugDumpClientResponse(200, w.Header())
	}
	w.WriteHeader(200)

	rid := "chatcmpl-" + randomHex(16)
	created := time.Now().Unix()

	sse := func(obj map[string]any) bool {
		raw, err := json.Marshal(obj)
		if err != nil {
			return false
		}
		debugf("sse -> %s", raw)
		if _, err := w.Write(append([]byte("data: "), append(raw, []byte("\n\n")...)...)); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	contentChunk := func(text string, finish any) map[string]any {
		return map[string]any{
			"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": text}, "finish_reason": finish}},
		}
	}

	// role-establishing chunk (required by most clients)
	sse(contentChunkRole(rid, created, model))

	// agent mode: incrementally split model text from tool-call blocks
	var interceptor *AgentStreamInterceptor
	if agentMode {
		interceptor = &AgentStreamInterceptor{}
	}
	emittedToolCall := false

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	err := run(ctx, func(ch Chunk) bool {
		s.observeStreamChunk(ch)
		// Thinking deltas ride alongside content as reasoning_content,
		// matching DeepSeek's own OpenAI-compatible field name.
		if ch.Type == "reasoning" && ch.Content != "" {
			if !sse(map[string]any{
				"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"reasoning_content": ch.Content}, "finish_reason": nil}},
			}) {
				return true
			}
			return false
		}
		if ch.Type != "content" || ch.Content == "" {
			return false
		}
		if interceptor != nil {
			parsed := interceptor.Feed(ch.Content)
			if parsed.Content != "" && !sse(contentChunk(parsed.Content, nil)) {
				return true
			}
			for _, call := range parsed.ToolCalls {
				emittedToolCall = true
				debugf("agent tool call #%d: %s(%s)", call["index"], call["function"].(map[string]any)["name"], call["function"].(map[string]any)["arguments"])
				if !sse(map[string]any{
					"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
					"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{call}}, "finish_reason": nil}},
				}) {
					return true
				}
			}
			return false
		}
		return !sse(contentChunk(ch.Content, nil))
	})

	if err != nil {
		etype, _ := errorType(err)
		sse(map[string]any{"error": map[string]any{"message": err.Error(), "type": etype}})
		return err
	}

	if interceptor != nil {
		final := interceptor.Finish()
		if final.Content != "" && !sse(contentChunk(final.Content, nil)) {
			return errWriteFailed
		}
		for _, call := range final.ToolCalls {
			emittedToolCall = true
			debugf("agent tool call #%d: %s(%s) [final]", call["index"], call["function"].(map[string]any)["name"], call["function"].(map[string]any)["arguments"])
			if !sse(map[string]any{
				"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{call}}, "finish_reason": nil}},
			}) {
				return errWriteFailed
			}
		}
	}

	finishReason := "stop"
	if emittedToolCall {
		finishReason = "tool_calls"
	}
	sse(map[string]any{
		"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finishReason}},
	})
	w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	return nil
}

// observeStreamChunk routes non-content stream events to whoever needs them
// today: the reuse lease (message ids, ban_edit) and the history mode's
// parent-message tracking.
func (s *ProxyServer) observeStreamChunk(ch Chunk) {
	switch ch.Type {
	case "ready":
		if s.getUseHistory() && ch.ResponseMessageID != nil {
			s.setHistPar(ch.ResponseMessageID)
		}
		if s.activeLease != nil {
			s.activeLease.ObserveReady(ch.RequestMessageID, ch.ResponseMessageID)
		}
	case "ban_edit":
		if s.activeLease != nil {
			s.activeLease.ObserveBanEdit()
		}
	}
}

// contentChunkRole is the role-establishing first SSE chunk (required by most
// clients).
func contentChunkRole(rid string, created int64, model string) map[string]any {
	return map[string]any{
		"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant", "content": ""}, "finish_reason": nil}},
	}
}

// blockResponse writes the OpenAI non-streaming wire format for one upstream
// run. Flow-agnostic, like streamResponse.
func (s *ProxyServer) blockResponse(w http.ResponseWriter, r *http.Request, run upstreamRunner, model, prompt string) error {
	ctx := r.Context()
	var parts []string
	var reasoning []string

	err := run(ctx, func(ch Chunk) bool {
		s.observeStreamChunk(ch)
		switch {
		case ch.Type == "content" && ch.Content != "":
			parts = append(parts, ch.Content)
		case ch.Type == "reasoning" && ch.Content != "":
			reasoning = append(reasoning, ch.Content)
		}
		return false
	})

	if err != nil {
		if errors.Is(err, errWriteFailed) {
			return err
		}
		etype, status := errorType(err)
		writeError(w, err.Error(), etype, nil, status)
		return err
	}

	answer := strings.Join(parts, "")
	finishReason := "stop"
	message := map[string]any{"role": "assistant", "content": answer}

	if thinking := strings.Join(reasoning, ""); thinking != "" {
		message["reasoning_content"] = thinking
	}

	if agentMode {
		if calls := ParseAgentToolCalls(answer); len(calls) > 0 {
			finishReason = "tool_calls"
			message["tool_calls"] = calls
			debugf("agent tool calls parsed: %d", len(calls))
			debugf("agent tool calls: %s", debugJSON(calls))
		}
		answer = StripAgentToolCalls(answer)
		message["content"] = answer
	}

	promptTokens := countWords(prompt)
	completionTokens := countWords(answer)

	writeJSON(w, http.StatusOK, map[string]any{
		"id":      "chatcmpl-" + randomHex(16),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	})
	return nil
}

// ── helpers ──────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, data any) {
	raw, err := json.Marshal(data)
	if err != nil {
		writeError(w, "Internal error", "server_error", nil, 500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if debugMode {
		debugDumpClientResponse(status, w.Header())
		debugf("  -> body: %s", debugIndent(debugTruncate(raw)))
	}
	w.WriteHeader(status)
	w.Write(raw)
}

func writeError(w http.ResponseWriter, msg, etype string, code any, status int) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    etype,
			"param":   nil,
			"code":    code,
		},
	})
}

func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	_, err := decodeBodyRaw(w, r, dst)
	return err
}

// decodeBodyRaw decodes the JSON body into dst and also returns the raw bytes
// so debug mode can dump exactly what the client sent.
func decodeBodyRaw(w http.ResponseWriter, r *http.Request, dst any) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, 16<<20)
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, fmt.Sprintf("Invalid JSON body: %v", err), "invalid_request_error", "invalid_json", 400)
		return nil, err
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		writeError(w, fmt.Sprintf("Invalid JSON body: %v", err), "invalid_request_error", "invalid_json", 400)
		return nil, err
	}
	return raw, nil
}

func toBool(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case string:
		b, _ := strconv.ParseBool(t)
		return b
	}
	return false
}

func countWords(s string) int {
	return len(strings.Fields(s))
}
