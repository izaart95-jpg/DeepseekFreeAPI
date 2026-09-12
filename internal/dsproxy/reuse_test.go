package dsproxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"testing"
	"time"
)

// ── edit_message wire format ──────────────────────────────────────────────────

// TestBuildEditMessageBody pins the JSON body POSTed to /chat/edit_message:
// it must mirror the web client exactly (no model_type, action null).
func TestBuildEditMessageBody(t *testing.T) {
	body := BuildEditMessageBody(EditChatParams{
		ChatSessionID:   "ba862bc1-5473-47da-8a7c-d5d38341358f",
		MessageID:       3,
		Prompt:          "what is my name",
		ThinkingEnabled: false,
		SearchEnabled:   false,
	})
	if _, ok := body["model_type"]; ok {
		t.Error("edit_message body must NOT carry model_type (web client omits it)")
	}
	if body["action"] != nil {
		t.Errorf("action = %v, want nil", body["action"])
	}
	if got := len(body); got != 7 {
		t.Errorf("edit body has %d keys, want 7: %v", got, body)
	}
	want := map[string]any{
		"chat_session_id":  "ba862bc1-5473-47da-8a7c-d5d38341358f",
		"message_id":       3,
		"ref_file_ids":     []any{},
		"prompt":           "what is my name",
		"search_enabled":   false,
		"thinking_enabled": false,
		"action":           nil,
	}
	for k, v := range want {
		got := body[k]
		if k == "ref_file_ids" {
			if sl, ok := got.([]any); !ok || len(sl) != 0 {
				t.Errorf("%s = %#v, want empty array", k, got)
			}
			continue
		}
		if got != v {
			t.Errorf("%s = %#v, want %#v", k, got, v)
		}
	}
	// Full round-trip sanity against the real web-client payload.
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 7 {
		t.Errorf("marshaled body has %d keys, want 7: %s", len(decoded), raw)
	}
}

// TestBuildEditMessageBodyNormalizesMessageID: numeric-string message ids
// (as decoded JSON floats/strings) must normalize to the u32 the API expects.
func TestBuildEditMessageBodyNormalizesMessageID(t *testing.T) {
	cases := []struct {
		in   any
		want any
	}{
		{3, 3},
		{float64(9), int64(9)},
		{"11", int64(11)},
		{nil, nil},
	}
	for _, tc := range cases {
		body := BuildEditMessageBody(EditChatParams{MessageID: tc.in})
		if body["message_id"] != tc.want {
			t.Errorf("message_id %v -> %#v, want %#v", tc.in, body["message_id"], tc.want)
		}
	}
}

// ── ban_edit stream parsing ──────────────────────────────────────────────────

// TestParseBanEditChunk: the full-response event with "ban_edit": true must
// surface a "ban_edit" chunk (the reuse manager's rotation signal) while the
// RESPONSE fragment still streams normally.
func TestParseBanEditChunk(t *testing.T) {
	chunks := feedStreamEvents(t, []string{
		`{"request_message_id":11,"response_message_id":12,"model_type":"default"}`,
		`{"v":{"response":{"message_id":12,"parent_id":11,"role":"ASSISTANT","ban_edit":true,"status":"WIP","fragments":[{"id":12,"type":"RESPONSE","content":"I","references":[],"stage_id":1}]}}}`,
		`{"v":" forgot"}`,
		`{"p":"response/status","o":"SET","v":"FINISHED"}`,
	})

	var sawBanEdit bool
	var content string
	for _, ch := range chunks {
		if ch.Type == "ban_edit" {
			sawBanEdit = true
		}
		if ch.Type == "content" {
			content += ch.Content
		}
	}
	if !sawBanEdit {
		t.Fatal("ban_edit=true must produce a \"ban_edit\" chunk")
	}
	if content != "I forgot" {
		t.Errorf("content = %q, want %q", content, "I forgot")
	}
}

// TestParseBanEditNegative: ban_edit=false must NOT emit a ban_edit chunk.
func TestParseBanEditNegative(t *testing.T) {
	chunks := feedStreamEvents(t, []string{
		`{"v":{"response":{"message_id":2,"ban_edit":false,"fragments":[{"id":2,"type":"RESPONSE","content":"hi"}]}}}`,
	})
	for _, ch := range chunks {
		if ch.Type == "ban_edit" {
			t.Fatal("ban_edit=false must not produce a \"ban_edit\" chunk")
		}
	}
}

// feedStreamEvents is the local twin of tests' feedEvents: parse SSE payloads
// through parseStreamData and collect all chunks.
func feedStreamEvents(t *testing.T, events []string) []Chunk {
	t.Helper()
	st := &streamState{}
	var out []Chunk
	for _, raw := range events {
		var data map[string]any
		if err := json.Unmarshal([]byte(raw), &data); err != nil {
			t.Fatalf("unmarshal %s: %v", raw, err)
		}
		chunks, complete, err := parseStreamData(data, st)
		if err != nil {
			t.Fatalf("parseStreamData(%s): %v", raw, err)
		}
		out = append(out, chunks...)
		if complete {
			break
		}
	}
	return out
}

// ── ReuseManager lifecycle ────────────────────────────────────────────────────

// stubReuseBackend fakes the session endpoints and records every call.
type stubReuseBackend struct {
	mu      sync.Mutex
	seq     int
	created []string
	deleted []string
	fail    map[string]error // method name -> error to return (until cleared)
}

func (b *stubReuseBackend) CreateChatSession(ctx context.Context) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.fail["create"]; err != nil {
		return "", err
	}
	b.seq++
	id := fmt.Sprintf("sess-%03d", b.seq)
	b.created = append(b.created, id)
	return id, nil
}

func (b *stubReuseBackend) DeleteChatSession(ctx context.Context, ids ...string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.fail["delete"]; err != nil {
		return err
	}
	b.deleted = append(b.deleted, ids...)
	return nil
}

func (b *stubReuseBackend) snapshot() (created, deleted []string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]string(nil), b.created...), append([]string(nil), b.deleted...)
}

func (b *stubReuseBackend) setFail(method string, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil {
		delete(b.fail, method)
		return
	}
	if b.fail == nil {
		b.fail = map[string]error{}
	}
	b.fail[method] = err
}

func newTestReuse() (*ReuseManager, *stubReuseBackend) {
	backend := &stubReuseBackend{}
	return NewReuseManager(testLogger(), backend), backend
}

func testLogger() *log.Logger { return log.New(io.Discard, "", 0) }

// waitForReuse polls cond until it holds or the deadline passes.
func waitForReuse(t *testing.T, timeout time.Duration, msg string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal(msg)
}

// TestReuseLazyCreation: constructing the manager and even Begin-checkouts
// must not create anything until the first request actually needs a session…
// actually they DO create lazily on first Begin — the point is: nothing is
// created at construction time, and the first Begin creates exactly one.
func TestReuseLazyCreation(t *testing.T) {
	m, backend := newTestReuse()

	// No session exists until Begin.
	if created, _ := backend.snapshot(); len(created) != 0 {
		t.Fatalf("construction must not create sessions, got %v", created)
	}

	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	created, _ := backend.snapshot()
	if len(created) != 1 {
		t.Fatalf("first Begin must create exactly 1 session, got %v", created)
	}
	if lease.SessionID() != created[0] {
		t.Fatalf("lease id %q != created %q", lease.SessionID(), created[0])
	}
	if lease.MessageIDForEdit() != nil {
		t.Fatalf("fresh session has no message id yet, got %v", lease.MessageIDForEdit())
	}
	lease.Complete()

	// Second Begin reuses the same session: no new create, no delete.
	lease2, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin 2: %v", err)
	}
	if lease2.SessionID() != created[0] {
		t.Fatalf("second Begin must reuse the same session: %q != %q", lease2.SessionID(), created[0])
	}
	created, deleted := backend.snapshot()
	if len(created) != 1 || len(deleted) != 0 {
		t.Fatalf("reuse must not create/delete: created=%v deleted=%v", created, deleted)
	}
	lease2.Complete()
}

// TestReuseEditsAdvanceAndRotate: each completed lease consumes one edit from
// the budget; when the budget is exhausted the session is deleted upstream
// (async) and the next Begin creates a fresh one — exactly one create per
// rotation, never a burst.
func TestReuseEditsAdvanceAndRotate(t *testing.T) {
	m, backend := newTestReuse()

	var sessions []string
	for i := 0; i < reuseEditBudget; i++ {
		lease, err := m.Begin(context.Background())
		if err != nil {
			t.Fatalf("Begin %d: %v", i, err)
		}
		if lease.EditsRemaining() != reuseEditBudget-i {
			t.Fatalf("lease %d: edits remaining = %d, want %d", i, lease.EditsRemaining(), reuseEditBudget-i)
		}
		sessions = append(sessions, lease.SessionID())
		lease.Complete()
	}

	// All reuseEditBudget turns used the same session, nothing deleted yet.
	if sessions[0] != sessions[len(sessions)-1] {
		t.Fatal("budget of edits must ride the same session")
	}
	created, deleted := backend.snapshot()
	if len(created) != 1 {
		t.Fatalf("budget turns must use 1 session, got %v", created)
	}
	if len(deleted) != 0 {
		t.Fatalf("nothing should be deleted before rotation, got %v", deleted)
	}

	// The budget is exhausted: the NEXT Begin rotates (old deleted, new created).
	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin after budget: %v", err)
	}
	if lease.SessionID() == sessions[0] {
		t.Fatal("rotation must hand out a fresh session")
	}
	waitForReuse(t, 5*time.Second, "rotation must delete the used-up session", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == 1 && deleted[0] == sessions[0]
	})
	created, _ = backend.snapshot()
	if len(created) != 2 {
		t.Fatalf("rotation must create exactly one new session, got %v", created)
	}
	lease.Complete()
}

// TestReuseBanEditRotates: a stream reporting ban_edit must rotate the
// session even when the local budget hasn't hit zero yet.
func TestReuseBanEditRotates(t *testing.T) {
	m, backend := newTestReuse()

	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	first := lease.SessionID()
	lease.ObserveReady(3, 4)
	if lease.MessageIDForEdit() != 3 {
		t.Fatalf("ready must record the next edit target, got %v", lease.MessageIDForEdit())
	}
	lease.ObserveBanEdit()
	lease.Complete()

	waitForReuse(t, 5*time.Second, "ban_edit must delete the session", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == 1 && deleted[0] == first
	})

	lease2, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin after ban: %v", err)
	}
	if lease2.SessionID() == first {
		t.Fatal("ban_edit rotation must create a fresh session")
	}
	lease2.Complete()
}

// TestReuseAbortDoesNotConsumeBudget: a failed request must not burn an edit.
func TestReuseAbortDoesNotConsumeBudget(t *testing.T) {
	m, backend := newTestReuse()

	for i := 0; i < 3; i++ {
		lease, err := m.Begin(context.Background())
		if err != nil {
			t.Fatalf("Begin %d: %v", i, err)
		}
		lease.Abort()
	}
	if _, _, ok := m.Status(); !ok {
		t.Fatal("session must survive aborted requests")
	}
	if _, deleted := backend.snapshot(); len(deleted) != 0 {
		t.Fatalf("aborts must not delete the session, got %v", deleted)
	}

	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin after aborts: %v", err)
	}
	if lease.EditsRemaining() != reuseEditBudget {
		t.Fatalf("aborts must not consume budget: %d left, want %d", lease.EditsRemaining(), reuseEditBudget)
	}
	lease.Complete()
}

// TestReuseDeadSessionRotates: MarkDead (session rejected upstream) rotates.
func TestReuseDeadSessionRotates(t *testing.T) {
	m, backend := newTestReuse()

	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	first := lease.SessionID()
	lease.MarkDead()
	lease.Complete()

	waitForReuse(t, 5*time.Second, "dead session must be deleted", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == 1 && deleted[0] == first
	})
	lease2, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin after dead: %v", err)
	}
	if lease2.SessionID() == first {
		t.Fatal("dead session must not be reused")
	}
	lease2.Complete()
}

// TestReuseSerializesConcurrentRequests: Begin must serialize — a second
// concurrent Begin blocks until the first lease finishes (one generation in
// flight, exactly like a browser user).
func TestReuseSerializesConcurrentRequests(t *testing.T) {
	m, _ := newTestReuse()

	first, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin 1: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		second, err := m.Begin(context.Background())
		if err == nil {
			second.Complete()
		}
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("concurrent Begin must block, not hand out the session twice")
		}
	case <-time.After(150 * time.Millisecond):
		// Still blocked — correct.
	}

	first.Complete()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("queued Begin after release: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("queued Begin never ran after the first lease finished")
	}
}

// TestReuseShutdownClearsSession: graceful stop deletes the live session.
func TestReuseShutdownClearsSession(t *testing.T) {
	m, backend := newTestReuse()

	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	id := lease.SessionID()
	lease.Complete()

	m.Shutdown()
	waitForReuse(t, 5*time.Second, "shutdown must delete the live session", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == 1 && deleted[0] == id
	})

	if _, err := m.Begin(context.Background()); !errors.Is(err, ErrReuseShuttingDown) {
		t.Fatalf("Begin after shutdown = %v, want ErrReuseShuttingDown", err)
	}
}

// TestReuseShutdownIdempotent: a second Shutdown must not double-delete.
func TestReuseShutdownIdempotent(t *testing.T) {
	m, backend := newTestReuse()
	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	lease.Complete()

	m.Shutdown()
	m.Shutdown()
	time.Sleep(50 * time.Millisecond)
	_, deleted := backend.snapshot()
	if len(deleted) != 1 {
		t.Fatalf("shutdown must delete exactly once, got %v", deleted)
	}
}

// TestReuseCreateRetries: a failed create retries once (human-paced) and
// succeeds on the second attempt.
func TestReuseCreateRetries(t *testing.T) {
	m, backend := newTestReuse()
	backend.setFail("create", errors.New("cloudflare"))
	go func() {
		// Clear the failure while the first Begin's retry is still pending.
		time.Sleep(100 * time.Millisecond)
		backend.setFail("create", nil)
	}()
	start := time.Now()
	lease, err := m.Begin(context.Background())
	if err != nil {
		t.Fatalf("Begin with transient create failure: %v", err)
	}
	if elapsed := time.Since(start); elapsed < reuseCreateBackoff {
		t.Fatalf("retry must be human-paced, waited %s", elapsed)
	}
	lease.Complete()
}

// ── session-dead error classification ────────────────────────────────────────

func TestIsSessionDeadError(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{nil, false},
		{errors.New("some random error"), false},
		{APIError{Msg: "session not found", StatusCode: 404}, true},
		{APIError{Msg: "invalid session id", StatusCode: 400}, true},
		{APIError{Msg: "Chat session does not exist", StatusCode: 400}, true},
		{APIError{Msg: "unrelated failure", StatusCode: 500}, false},
		{APIError{Msg: "rate limited", StatusCode: 429}, false},
		{AuthenticationError{Msg: "bad token"}, false},
		{RateLimitError{Msg: "nope"}, false},
	}
	for _, tc := range cases {
		if got := isSessionDeadError(tc.err); got != tc.want {
			t.Errorf("isSessionDeadError(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}
