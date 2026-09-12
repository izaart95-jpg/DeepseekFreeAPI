package dsproxy

import (
	"context"
	"errors"
	"log"
	"strings"
	"sync"
	"time"
)

// Stealth session reuse (history=false, the default flow).
//
// DeepSeek suspends accounts whose traffic looks automated. Two behaviors in
// the old pool were textbook bot fingerprints:
//
//   1. A boot burst: 5 parallel /chat_session/create calls the moment the
//      proxy starts — a human never does that.
//   2. Per-request churn: every stateless request created a brand-new
//      session, sent exactly one message, then deleted the session and
//      created a replacement. Sessions always containing exactly one
//      exchange, and the create/delete cadence tracking request volume, is
//      trivially detectable server-side.
//
// The reuse flow fixes both by mirroring what a real web user does:
//
//   - ONE chat session is created lazily, on the first request, and only one
//     exists at a time (a user opening one chat).
//   - Every following stateless request POSTs /chat/edit_message, re-editing
//     the *last* user message of the session. Each edit replaces the previous
//     prompt, so the conversation stays a single root exchange forever — no
//     context accumulates (the model forgets everything between requests,
//     which is exactly the stateless semantics /history=false promises), and
//     no create/delete churn ever happens at request time.
//   - Requests are serialized: a browser user has one generation in flight,
//     so two parallel edits of one message would look (and behave) wrong.
//     Concurrent requests queue behind the current one, like a user waiting
//     for a response before editing again.
//   - DeepSeek caps edits of one message (observed budget: 6). When the
//     budget runs out (the stream reports ban_edit, or an edit is rejected),
//     the used session is retired (deleted upstream) and the NEXT request
//     lazily creates a fresh session — again a single, human-paced call.
//
// Backward compatibility: SESSION_FLOW=pool / --legacy-pool restores the old
// pre-warmed session-pool flow, and --sync-mode / SESSION_FLOW=sync keeps the
// legacy one-session-per-request behavior.

const (
	// reuseEditBudget is the number of times one user message may be edited
	// on chat.deepseek.com before the edit button disappears (observed: 6).
	reuseEditBudget = 6
	// reuseOpTimeout bounds one upstream create/delete call.
	reuseOpTimeout = 30 * time.Second
	// reuseCreateBackoff is the human-paced pause before the single retry of
	// a failed session create (a user re-clicking "new chat" after an error).
	reuseCreateBackoff = 3 * time.Second
)

// ErrReuseShuttingDown is returned once Shutdown began: no new work starts.
var ErrReuseShuttingDown = errors.New("reuse manager is shutting down")

// ReuseBackend is the slice of DeepSeekAPI the manager needs; *DeepSeekAPI
// satisfies it implicitly and tests substitute a stub.
type ReuseBackend interface {
	CreateChatSession(ctx context.Context) (string, error)
	DeleteChatSession(ctx context.Context, sessionIDs ...string) error
}

// lazyReuseBackend resolves the API through ProxyServer.getAPI so the manager
// can exist before a token is configured (mirrors lazyBackend for the pool).
type lazyReuseBackend struct{ s *ProxyServer }

func (l *lazyReuseBackend) CreateChatSession(ctx context.Context) (string, error) {
	api, err := l.s.getAPI()
	if err != nil {
		return "", err
	}
	return api.CreateChatSession(ctx)
}

func (l *lazyReuseBackend) DeleteChatSession(ctx context.Context, ids ...string) error {
	api, err := l.s.getAPI()
	if err != nil {
		return err
	}
	return api.DeleteChatSession(ctx, ids...)
}

// reuseSession is the current reusable chat session and its edit state.
type reuseSession struct {
	id             string // DeepSeek chat_session_id
	rootMessageID  any    // id of the user message being edited (advances +2 per edit)
	editsRemaining int    // edits left before the message can no longer be edited
}

// ReuseManager owns the single reusable stateless session. The zero extra
// goroutines / zero startup calls are deliberate: from DeepSeek's vantage a
// user opened a chat, sends requests one at a time, edits their message,
// and only occasionally starts a fresh chat.
type ReuseManager struct {
	log     *log.Logger
	backend ReuseBackend

	// slot serializes requests: capacity 1. A request holds it from Begin
	// until Complete/Abort, so at most one edit/generation is in flight —
	// exactly like the web client.
	slot chan struct{}

	mu       sync.Mutex
	current  *reuseSession
	stopped  bool
	stopOnce sync.Once
	stopCh   chan struct{}
}

// NewReuseManager builds a reuse manager. Nothing is created at startup —
// the first request lazily creates the session (see Begin).
func NewReuseManager(logger *log.Logger, backend ReuseBackend) *ReuseManager {
	return &ReuseManager{
		log:     logger,
		backend: backend,
		slot:    make(chan struct{}, 1),
		stopCh:  make(chan struct{}),
	}
}

// Begin checks out the reusable session for one request, creating it lazily
// on first use. Concurrent callers queue (one generation in flight, like a
// browser user). The returned lease's Complete or Abort MUST be called
// exactly once when the request is done.
func (m *ReuseManager) Begin(ctx context.Context) (*Lease, error) {
	// Take the exclusive slot first: everything below runs one-at-a-time.
	select {
	case m.slot <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.stopCh:
		return nil, ErrReuseShuttingDown
	}

	m.mu.Lock()
	if m.stopped {
		m.mu.Unlock()
		<-m.slot
		return nil, ErrReuseShuttingDown
	}
	cur := m.current
	if cur != nil {
		m.mu.Unlock()
		return &Lease{m: m, session: cur}, nil
	}
	m.mu.Unlock()

	// No session yet: create one lazily (single call, human-paced retry).
	id, err := m.createSession(ctx)
	if err != nil {
		<-m.slot
		return nil, err
	}

	m.mu.Lock()
	if m.stopped || m.current != nil {
		// Shutdown (or a surprise) raced in: don't leak the new session.
		dead := m.stopped
		m.mu.Unlock()
		<-m.slot
		if !dead && m.current != nil {
			go m.retire(id) // a second session slipped in; drop ours
		} else {
			go m.retire(id)
		}
		if dead {
			return nil, ErrReuseShuttingDown
		}
		// Extremely unlikely (slot serialization makes this near-impossible):
		// retry from the top on the next tick.
		return m.Begin(ctx)
	}
	m.current = &reuseSession{id: id, editsRemaining: reuseEditBudget}
	m.mu.Unlock()
	return &Lease{m: m, session: m.current}, nil
}

// createSession performs one create call with a single human-paced retry.
func (m *ReuseManager) createSession(ctx context.Context) (string, error) {
	id, err := m.backend.CreateChatSession(ctx)
	if err == nil {
		m.log.Printf("[reuse] session ready: %s", id)
		return id, nil
	}
	warnf("[reuse] session create failed (%v); retrying in %s...", err, reuseCreateBackoff)
	select {
	case <-time.After(reuseCreateBackoff):
	case <-ctx.Done():
		return "", ctx.Err()
	case <-m.stopCh:
		return "", ErrReuseShuttingDown
	}
	cctx, cancel := context.WithTimeout(context.Background(), reuseOpTimeout)
	defer cancel()
	id, err = m.backend.CreateChatSession(cctx)
	if err != nil {
		return "", err
	}
	m.log.Printf("[reuse] session ready (after retry): %s", id)
	return id, nil
}

// Lease is one request's exclusive hold on the reusable session. Exactly one
// of Complete/Abort must be called when the request finishes.
type Lease struct {
	m       *ReuseManager
	session *reuseSession

	banEditSet  bool // stream reported ban_edit=true
	sessionDead bool // the session is unusable (upstream rejected it)
	done        bool // Complete/Abort already called
}

// SessionID exposes the chat_session_id in use.
func (l *Lease) SessionID() string { return l.session.id }

// MessageIDForEdit returns the message id an edit request should target: the
// last request_message_id observed (message ids advance +2 per edit), or nil
// while the session has not produced its first response yet (the first
// request must ride /chat/completion like a user's first message).
func (l *Lease) MessageIDForEdit() any { return l.session.rootMessageID }

// EditsRemaining reports the edit budget left for this session.
func (l *Lease) EditsRemaining() int { return l.session.editsRemaining }

// ObserveReady records the request/response message ids from a stream's
// "ready" event. The request_message_id is the user message the NEXT edit
// must target (it advances +2 per turn, matching the web client's numbering).
func (l *Lease) ObserveReady(requestMessageID, responseMessageID any) {
	if requestMessageID != nil {
		l.session.rootMessageID = requestMessageID
	}
}

// ObserveBanEdit records that upstream marked the message no longer editable
// (edit budget exhausted); the session rotates on release.
func (l *Lease) ObserveBanEdit() { l.banEditSet = true }

// MarkDead records that the session is unusable upstream (e.g. every edit is
// rejected with a session error — the session was deleted server-side). The
// session rotates on release so the next request starts fresh.
func (l *Lease) MarkDead() { l.sessionDead = true }

// Complete finalizes a successful request: consume one edit from the budget
// and rotate the session if the budget ran out or the message is banned.
func (l *Lease) Complete() {
	if l.done {
		return
	}
	l.done = true
	l.m.finishLease(l, true)
}

// Abort releases the lease after a failed request. The edit budget is not
// consumed (a rejected edit does not count), but a banned/dead session still
// rotates.
func (l *Lease) Abort() {
	if l.done {
		return
	}
	l.done = true
	l.m.finishLease(l, false)
}

// finishLease applies lease state and frees the serialization slot. Consuming
// an edit (success) or a dead/banned session triggers rotation: the used
// session is deleted upstream (best-effort, off the request path) and the
// next request creates the replacement lazily.
func (m *ReuseManager) finishLease(l *Lease, success bool) {
	m.mu.Lock()
	rotate := l.sessionDead || (l.banEditSet)
	cur := m.current
	if cur != nil && cur == l.session {
		if success {
			cur.editsRemaining--
		}
		if cur.editsRemaining <= 0 {
			rotate = true
		}
		if rotate {
			m.current = nil
		}
	} else if rotate {
		// The session was already replaced (rotation raced); still retire
		// the stale id if it's ours to retire.
		m.mu.Unlock()
		<-m.slot
		go m.retire(l.session.id)
		return
	}
	m.mu.Unlock()

	if rotate && cur != nil {
		m.log.Printf("[reuse] rotating session %s (budget=%d ban=%v dead=%v)",
			cur.id, cur.editsRemaining, l.banEditSet, l.sessionDead)
		go m.retire(cur.id)
	}
	<-m.slot
}

// retire deletes a used-up session upstream, best-effort, off the request
// path (a user deleting an old chat after moving on — one call, no churn).
func (m *ReuseManager) retire(sessionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), reuseOpTimeout)
	defer cancel()
	if err := m.backend.DeleteChatSession(ctx, sessionID); err != nil {
		warnf("[reuse] warning: failed to retire session %s: %v", sessionID, err)
		return
	}
	m.log.Printf("[reuse] retired chat session: %s", sessionID)
}

// Shutdown stops new checkouts and clears the current session so nothing is
// left behind on the account (the CTRL+C path). In-flight requests finish
// and retire their own session through the lease.
func (m *ReuseManager) Shutdown() {
	m.stopOnce.Do(func() {
		m.mu.Lock()
		m.stopped = true
		cur := m.current
		m.current = nil
		close(m.stopCh)
		m.mu.Unlock()
		if cur != nil {
			m.log.Printf("[reuse] clearing session %s on shutdown", cur.id)
			go m.retire(cur.id)
		}
	})
}

// Status reports the current session id and remaining edit budget (used by
// tests and introspection).
func (m *ReuseManager) Status() (sessionID string, editsRemaining int, ok bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.current == nil {
		return "", 0, false
	}
	return m.current.id, m.current.editsRemaining, true
}

// isSessionDeadError classifies upstream errors that mean "this chat session
// no longer exists server-side" (deleted upstream, expired, or another
// client consumed its edit budget). Such errors must rotate the reuse
// session instead of poisoning every later request.
func isSessionDeadError(err error) bool {
	if err == nil {
		return false
	}
	apiErr, ok := err.(APIError)
	if !ok {
		return false
	}
	if apiErr.StatusCode == 404 || apiErr.StatusCode == 400 {
		return true
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "session") {
		for _, needle := range []string{"not found", "not exist", "invalid", "deleted", "exist"} {
			if strings.Contains(msg, needle) {
				return true
			}
		}
	}
	return false
}
