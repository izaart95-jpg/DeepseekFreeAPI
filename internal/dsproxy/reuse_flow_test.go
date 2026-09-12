package dsproxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// This file exercises the full stealth reuse flow through handleChat with a
// stubbed DeepSeek upstream: session creation, /chat/completion on the first
// request, /chat/edit_message reuse afterwards, ban_edit rotation, and the
// legacy pool/sync flows staying intact.

// ── fake upstream ─────────────────────────────────────────────────────────────

// fakeUpstream stands in for chat.deepseek.com. It records every request and
// answers the endpoints the reuse flow touches.
type fakeUpstream struct {
	mu sync.Mutex

	sessions    []string // created session ids (in order)
	deleted     []string // deleted session ids
	completions []int    // message_id counter per session
	powCalls    []string
	chatCalls   []struct {
		path, sessionID, prompt string
		msgID                   any
	}

	editBudget int // edits allowed per session before ban_edit=true

	// errResponse, when set, makes the next chat/stream endpoint return this
	// status with the body (then clears itself).
	errResponse *struct {
		status int
		body   string
	}

	// handler hook for custom behavior; nil = default behavior.
	custom func(u *fakeUpstream, w http.ResponseWriter, r *http.Request) bool
}

func newFakeUpstream() *fakeUpstream {
	return &fakeUpstream{editBudget: 6}
}

func (u *fakeUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.custom != nil && u.custom(u, w, r) {
		return
	}

	switch r.URL.Path {
	case "/api/v0/chat/create_pow_challenge":
		u.powCalls = append(u.powCalls, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"code":0,"msg":"","data":{"biz_data":{"challenge":{"algorithm":"DeepSeekHashV1","challenge":"fakechallenge","salt":"fakesalt","difficulty":1,"expire_time":1893456000,"signature":"","target_path":"/web/chat"}}}}`)
		return

	case "/api/v0/chat_session/create":
		id := fmt.Sprintf("fake-sess-%d", len(u.sessions)+1)
		u.sessions = append(u.sessions, id)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"code":0,"msg":"","data":{"biz_data":{"chat_session":{"id":%q}}}}`, id)
		return

	case "/api/v0/chat_session/delete":
		var body struct {
			IDs []string `json:"chat_session_ids"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		u.deleted = append(u.deleted, body.IDs...)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"code":0,"msg":"","data":{"biz_code":0,"biz_msg":"","biz_data":null}}`)
		return

	case "/api/v0/chat/completion", "/api/v0/chat/edit_message":
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		sid, _ := body["chat_session_id"].(string)
		prompt, _ := body["prompt"].(string)
		var msgID any
		if v, ok := body["message_id"]; ok {
			msgID = v
		}
		u.chatCalls = append(u.chatCalls, struct {
			path, sessionID, prompt string
			msgID                   any
		}{r.URL.Path, sid, prompt, msgID})

		if u.errResponse != nil {
			w.WriteHeader(u.errResponse.status)
			fmt.Fprint(w, u.errResponse.body)
			u.errResponse = nil
			return
		}

		// Track per-session message numbering (ids advance +2 per turn).
		idx := 0
		for i, s := range u.sessions {
			if s == sid {
				idx = i
			}
		}
		if len(u.completions) <= idx {
			grow := make([]int, idx+1-len(u.completions))
			u.completions = append(u.completions, grow...)
		}
		isEdit := r.URL.Path == "/api/v0/chat/edit_message"
		reqID := u.completions[idx] + 1
		respID := reqID + 1
		u.completions[idx] += 2

		// Simulate the edit budget: after editBudget edits of one session,
		// ban_edit flips true.
		banEdit := isEdit && u.completions[idx]/2 > u.editBudget

		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "event: ready\ndata: {\"request_message_id\":%d,\"response_message_id\":%d,\"model_type\":\"default\"}\n\n", reqID, respID)
		fmt.Fprintf(w, "data: {\"v\":{\"response\":{\"message_id\":%d,\"parent_id\":%d,\"role\":\"ASSISTANT\",\"ban_edit\":%v,\"status\":\"WIP\",\"fragments\":[{\"id\":%d,\"type\":\"RESPONSE\",\"content\":\"Hi\",\"references\":[],\"stage_id\":1}]}}}\n\n", respID, reqID, banEdit, respID)
		fmt.Fprint(w, "data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\" there\"}\n\n")
		fmt.Fprint(w, "data: {\"p\":\"response/status\",\"o\":\"SET\",\"v\":\"FINISHED\"}\n\n")
		fmt.Fprint(w, "event: close\ndata: {\"click_behavior\":\"none\"}\n\n")
		return
	}

	w.WriteHeader(404)
	fmt.Fprintf(w, "not found: %s", r.URL.Path)
}

func (u *fakeUpstream) chatCallCount() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.chatCalls)
}

// upstream baseURL returns the fake upstream's base URL (no /api/v0).
func (u *fakeUpstream) baseURL(server *httptest.Server) string {
	return strings.TrimSuffix(server.URL, "/")
}

// ── helpers ────────────────────────────────────────────────────────────────────

// newReuseTestProxy spins the proxy with the stealth flow against a fake
// upstream. It returns the test server to close, the proxy and the upstream.
func newReuseTestProxy(t *testing.T) (*httptest.Server, *ProxyServer, *fakeUpstream) {
	t.Helper()
	t.Setenv("DEEPSEEK_TOKEN", "test-token-for-fake-upstream")
	upstream := newFakeUpstream()
	upServer := httptest.NewServer(upstream)
	t.Cleanup(upServer.Close)

	// Redirect the DeepSeek client at the fake upstream.
	old := swapBaseURL(upServer.URL + "/api/v0")
	t.Cleanup(func() { swapBaseURL(old) })

	proxy := NewProxyServer(testLogger(), "")
	proxy.AttachReuseManager(NewReuseManager(testLogger(), &lazyReuseBackend{s: proxy}))
	ts := httptest.NewServer(proxy)
	t.Cleanup(ts.Close)
	return ts, proxy, upstream
}

func reuseChatRequest(t *testing.T, ts *httptest.Server, prompt string, stream bool) map[string]any {
	t.Helper()
	body := map[string]any{
		"model":    "deepseek-v4.1-flash",
		"messages": []map[string]any{{"role": "user", "content": prompt}},
		"stream":   stream,
	}
	raw, _ := json.Marshal(body)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(raw)))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if stream {
		return map[string]any{"stream": string(data), "status": resp.StatusCode}
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal response %s: %v", data, err)
	}
	return out
}

// ── the stealth flow ───────────────────────────────────────────────────────────

// TestStealthFlowFirstCompletionThenEdits: request #1 rides /chat/completion
// (a fresh session's first message); requests #2+ ride /chat/edit_message
// against the advancing message id; all requests share ONE session and no
// session is created or deleted per request.
func TestStealthFlowFirstCompletionThenEdits(t *testing.T) {
	ts, proxy, upstream := newReuseTestProxy(t)

	if got := upstream.chatCallCount(); got != 0 {
		t.Fatalf("startup must not make any upstream chat call, got %d", got)
	}
	if len(upstream.sessions) != 0 {
		t.Fatalf("startup must not create sessions (boot burst!), got %v", upstream.sessions)
	}

	// Request 1: plain completion on a fresh session.
	out := reuseChatRequest(t, ts, "hello world", false)
	choices := out["choices"].([]any)
	if len(choices) != 1 {
		t.Fatalf("choices: %#v", out)
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if !strings.Contains(msg["content"].(string), "Hi there") {
		t.Fatalf("content = %#v", msg["content"])
	}

	if len(upstream.sessions) != 1 {
		t.Fatalf("first request must create exactly 1 session, got %v", upstream.sessions)
	}
	paths := []string{upstream.chatCalls[0].path}
	if !strings.HasSuffix(paths[0], "/chat/completion") {
		t.Fatalf("first request must ride /chat/completion, got %s", paths[0])
	}

	// Requests 2-3: edits of the same session, message id advancing +2.
	out = reuseChatRequest(t, ts, "second question", false)
	_ = out
	if len(upstream.sessions) != 1 {
		t.Fatalf("edits must not create sessions, got %v", upstream.sessions)
	}
	c2 := upstream.chatCalls[1]
	if !strings.HasSuffix(c2.path, "/chat/edit_message") {
		t.Fatalf("second request must ride /chat/edit_message, got %s", c2.path)
	}
	if c2.prompt != "second question" {
		t.Fatalf("edit prompt = %q", c2.prompt)
	}
	if fmt.Sprint(c2.msgID) != "1" {
		t.Fatalf("edit must target message id 1 (the first user message), got %v", c2.msgID)
	}

	out = reuseChatRequest(t, ts, "third question", false)
	_ = out
	c3 := upstream.chatCalls[2]
	if fmt.Sprint(c3.msgID) != "3" {
		t.Fatalf("edit target must advance +2 per turn, got %v", c3.msgID)
	}

	// Streaming variant rides the edit path too, and streams clean SSE.
	sout := reuseChatRequest(t, ts, "stream me", true)
	streamData, _ := sout["stream"].(string)
	if !strings.Contains(streamData, `"Hi"`) || !strings.Contains(streamData, `" there"`) {
		t.Fatalf("streamed response missing content: %q", streamData)
	}
	if !strings.Contains(streamData, "[DONE]") {
		t.Fatalf("stream must end with [DONE]: %q", streamData)
	}
	c4 := upstream.chatCalls[3]
	if !strings.HasSuffix(c4.path, "/chat/edit_message") {
		t.Fatalf("streaming request must ride edit_message, got %s", c4.path)
	}

	// One session, no churn, one PoW per chat call.
	if len(upstream.sessions) != 1 {
		t.Fatalf("the whole conversation must ride 1 session, got %v", upstream.sessions)
	}
	if len(upstream.deleted) != 0 {
		t.Fatalf("nothing should be deleted while the session is healthy, got %v", upstream.deleted)
	}
	_ = proxy
}

// TestStealthFlowRotatesAfterEditBudget: once the edit budget is exhausted
// (upstream reports ban_edit), the session is deleted and the next request
// starts a fresh session via /chat/completion again.
func TestStealthFlowRotatesAfterEditBudget(t *testing.T) {
	ts, _, upstream := newReuseTestProxy(t)
	upstream.editBudget = 2 // rotate quickly in the test

	seen := map[string]bool{}
	for i := 0; i < 12; i++ {
		reuseChatRequest(t, ts, fmt.Sprintf("q%d", i), false)
	}
	if len(upstream.sessions) < 2 {
		t.Fatalf("edit budget exhaustion must rotate sessions, got %v", upstream.sessions)
	}
	for _, s := range upstream.sessions {
		seen[s] = true
	}
	// The used-up sessions are deleted upstream (async retire).
	waitForReuse(t, 5*time.Second, "retired sessions must be deleted upstream", func() bool {
		return len(upstream.deleted) >= len(upstream.sessions)-1
	})
	if len(seen) != len(upstream.sessions) {
		t.Fatalf("duplicate session ids: %v", upstream.sessions)
	}
}

// TestStealthFlowRecoversFromDeadSession: when an edit is rejected with a
// session-not-found style error, the flow rotates and the NEXT request
// succeeds on a fresh session.
func TestStealthFlowRecoversFromDeadSession(t *testing.T) {
	ts, _, upstream := newReuseTestProxy(t)

	reuseChatRequest(t, ts, "first", false)
	reuseChatRequest(t, ts, "second", false)

	// Poison the next edit.
	upstream.errResponse = &struct {
		status int
		body   string
	}{400, "chat session does not exist"}

	bad := reuseChatRequest(t, ts, "third", false)
	if em, ok := bad["error"].(map[string]any); ok {
		if !strings.Contains(em["message"].(string), "does not exist") {
			t.Fatalf("expected the upstream error to surface, got %#v", bad)
		}
	} else {
		t.Fatalf("expected an error response, got %#v", bad)
	}

	// The flow rotated: the next request creates a new session and succeeds.
	good := reuseChatRequest(t, ts, "fourth", false)
	choices := good["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if !strings.Contains(msg["content"].(string), "Hi there") {
		t.Fatalf("recovery request failed: %#v", good)
	}
	if len(upstream.sessions) < 2 {
		t.Fatalf("dead session must trigger rotation, sessions=%v", upstream.sessions)
	}
	last := upstream.chatCalls[len(upstream.chatCalls)-1]
	if !strings.HasSuffix(last.path, "/chat/completion") {
		t.Fatalf("first request on the fresh session must ride completion, got %s", last.path)
	}
}

// ── legacy flows stay intact ──────────────────────────────────────────────────

// TestLegacySyncFlowPerRequestSessions: with the sync flow enabled every
// request creates and deletes its own session (old behavior preserved).
func TestLegacySyncFlowPerRequestSessions(t *testing.T) {
	t.Setenv("DEEPSEEK_TOKEN", "test-token-for-fake-upstream")
	upstream := newFakeUpstream()
	upServer := httptest.NewServer(upstream)
	t.Cleanup(upServer.Close)
	old := swapBaseURL(upServer.URL + "/api/v0")
	t.Cleanup(func() { swapBaseURL(old) })

	proxy := NewProxyServer(testLogger(), "")
	proxy.EnableSyncFlow()
	ts := httptest.NewServer(proxy)
	t.Cleanup(ts.Close)

	for i := 0; i < 3; i++ {
		reuseChatRequest(t, ts, fmt.Sprintf("q%d", i), false)
	}
	if len(upstream.sessions) != 3 {
		t.Fatalf("sync flow must create one session per request, got %v", upstream.sessions)
	}
	if got := upstream.chatCallCount(); got != 3 {
		t.Fatalf("sync flow must run 3 completions, got %d", got)
	}
	if !strings.HasSuffix(upstream.chatCalls[0].path, "/chat/completion") {
		t.Fatalf("sync flow must use /chat/completion, got %s", upstream.chatCalls[0].path)
	}
}

// TestLegacyPoolFlowWarmsBatch: with the pool flow attached the old behavior
// (boot warmup + per-request retire/refill) is preserved.
func TestLegacyPoolFlowWarmsBatch(t *testing.T) {
	t.Setenv("DEEPSEEK_TOKEN", "test-token-for-fake-upstream")
	upstream := newFakeUpstream()
	upServer := httptest.NewServer(upstream)
	t.Cleanup(upServer.Close)
	old := swapBaseURL(upServer.URL + "/api/v0")
	t.Cleanup(func() { swapBaseURL(old) })

	proxy := NewProxyServer(testLogger(), "")
	pool := NewSessionPool(testLogger(), &lazyBackend{s: proxy}, 2)
	proxy.AttachSessionPool(pool)
	ts := httptest.NewServer(proxy)
	t.Cleanup(ts.Close)

	pool.Start()
	waitForReuse(t, 5*time.Second, "pool never warmed", func() bool {
		upstream.mu.Lock()
		defer upstream.mu.Unlock()
		return len(upstream.sessions) == 2
	})

	reuseChatRequest(t, ts, "hello", false)

	if got := upstream.chatCallCount(); got != 1 {
		t.Fatalf("pool flow must run 1 completion, got %d", got)
	}
	// The used session is released (deleted + refilled).
	waitForReuse(t, 5*time.Second, "pool did not retire the used session", func() bool {
		upstream.mu.Lock()
		defer upstream.mu.Unlock()
		return len(upstream.deleted) >= 1
	})
	pool.Shutdown()
}

// ── history mode stays on /chat/completion ───────────────────────────────────

// TestHistoryModeUnaffected: history=true keeps its dedicated session and
// threaded /chat/completion calls; the reuse flow must not touch it.
func TestHistoryModeUnaffected(t *testing.T) {
	ts, _, upstream := newReuseTestProxy(t)

	// Enable history.
	resp, err := http.Post(ts.URL+"/history", "application/json", strings.NewReader(`{"enable":true}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	reuseChatRequest(t, ts, "remember this", false)
	reuseChatRequest(t, ts, "and this", false)

	// History creates exactly one session and uses threaded completions.
	if len(upstream.sessions) != 1 {
		t.Fatalf("history mode must use one dedicated session, got %v", upstream.sessions)
	}
	for i, c := range upstream.chatCalls {
		if !strings.HasSuffix(c.path, "/chat/completion") {
			t.Fatalf("history call %d must ride /chat/completion, got %s", i, c.path)
		}
	}
	if len(upstream.deleted) != 0 {
		t.Fatalf("history sessions must never be deleted, got %v", upstream.deleted)
	}
}

// ── context plumb-through ────────────────────────────────────────────────────

// TestReuseContextCancellation: a client hangup while queued must not leave
// the slot stuck (the next request can still acquire the session).
func TestReuseContextCancellation(t *testing.T) {
	m, _ := newTestReuse()

	first, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin 1: %v", err)
	}

	// A queued Begin whose context dies must give up cleanly.
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		_, _ = m.Begin(ctx)
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	time.Sleep(30 * time.Millisecond)

	first.Complete()

	// The slot must be free again: a fresh Begin succeeds promptly.
	done := make(chan error, 1)
	go func() {
		l, err := m.Begin(context.Background())
		if err == nil {
			l.Complete()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Begin after cancellation: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("slot stuck after cancelled wait + release")
	}
}

// swapBaseURL points the DeepSeek client at a different upstream (used by
// tests to hit a fake upstream). Returns the previous base URL.
func swapBaseURL(newURL string) string {
	old := deepseekBaseURL
	deepseekBaseURL = newURL
	return old
}
