package tests

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"deepseek/internal/dsproxy"
)

// stubBackend fakes the DeepSeek session endpoints so pool mechanics can be
// tested without network access.
type stubBackend struct {
	mu      sync.Mutex
	seq     int
	created []string
	deleted []string
}

func (b *stubBackend) CreateChatSession(ctx context.Context) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.seq++
	id := fmt.Sprintf("sess-%03d", b.seq)
	b.created = append(b.created, id)
	return id, nil
}

func (b *stubBackend) DeleteChatSession(ctx context.Context, ids ...string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.deleted = append(b.deleted, ids...)
	return nil
}

func (b *stubBackend) snapshot() (created, deleted []string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]string(nil), b.created...), append([]string(nil), b.deleted...)
}

// waitFor polls cond until it holds or the deadline passes; fails the test
// with a message otherwise.
func waitFor(t *testing.T, timeout time.Duration, msg string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal(msg)
}

func newTestPool(size int) (*dsproxy.SessionPool, *stubBackend) {
	backend := &stubBackend{}
	pool := dsproxy.NewSessionPool(discardLogger(), backend, size)
	pool.Start()
	return pool, backend
}

// TestSessionPoolWarmsAndServesInstantly: the standing batch is pre-made at
// startup and Acquire hands out ready sessions without creating anything.
func TestSessionPoolWarmsAndServesInstantly(t *testing.T) {
	pool, backend := newTestPool(3)
	defer pool.Shutdown()

	waitFor(t, 5*time.Second, "pool never warmed up to 3 sessions", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 3
	})

	a, err := pool.Acquire(context.Background(), time.Second)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	b, err := pool.Acquire(context.Background(), time.Second)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if a == b {
		t.Fatalf("Acquire returned the same session twice: %s", a)
	}
	created, deleted := backend.snapshot()
	if len(created) != 3 || len(deleted) != 0 {
		t.Fatalf("acquire must not create/delete upstream: created=%v deleted=%v", created, deleted)
	}
}

// TestSessionPoolReleaseRetiresAndRefills: only after Release (i.e. once the
// response is fully processed) is the used session deleted upstream and a
// replacement created to fill the gap.
func TestSessionPoolReleaseRetiresAndRefills(t *testing.T) {
	pool, backend := newTestPool(2)
	defer pool.Shutdown()

	waitFor(t, 5*time.Second, "pool never warmed up", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 2
	})

	id, err := pool.Acquire(context.Background(), time.Second)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if _, deleted := backend.snapshot(); len(deleted) != 0 {
		t.Fatalf("used session must survive until Release, but %v were deleted", deleted)
	}

	pool.Release(id)

	waitFor(t, 5*time.Second, "released session was not deleted upstream", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == 1 && deleted[0] == id
	})
	waitFor(t, 5*time.Second, "gap was not refilled after release", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 3
	})

	// The refill must be immediately available again.
	next, err := pool.Acquire(context.Background(), 2*time.Second)
	if err != nil {
		t.Fatalf("Acquire after release: %v", err)
	}
	if next == id {
		t.Fatal("refilled session must be fresh, not the retired one")
	}
}

// TestSessionPoolShutdownClearsRemainingSessions: graceful stop deletes every
// still-pooled session and stops refills; later Acquires report shutdown.
func TestSessionPoolShutdownClearsRemainingSessions(t *testing.T) {
	pool, backend := newTestPool(3)

	waitFor(t, 5*time.Second, "pool never warmed up", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 3
	})

	pool.Shutdown()

	created, _ := backend.snapshot()
	waitFor(t, 5*time.Second, "shutdown did not clear all remaining sessions", func() bool {
		_, deleted := backend.snapshot()
		return len(deleted) == len(created)
	})

	// Acquiring after shutdown must fail fast with ErrPoolClosing.
	if _, err := pool.Acquire(context.Background(), time.Second); err != dsproxy.ErrPoolClosing {
		t.Fatalf("Acquire after Shutdown = %v, want ErrPoolClosing", err)
	}

	// No extra sessions may be created after shutdown.
	time.Sleep(100 * time.Millisecond)
	createdAfter, deleted := backend.snapshot()
	if len(createdAfter) != 3 {
		t.Fatalf("created after shutdown = %d, want 3 (%v)", len(createdAfter), createdAfter)
	}
	for _, id := range created {
		found := false
		for _, d := range deleted {
			if d == id {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("pooled session %s was not cleared on shutdown", id)
		}
	}
}

// TestSessionPoolTimeoutReturnsSentinel: when the batch is empty a bounded
// Acquire gives up with ErrPoolTimeout so callers can fall back gracefully.
func TestSessionPoolTimeoutReturnsSentinel(t *testing.T) {
	pool, backend := newTestPool(1)
	defer pool.Shutdown()

	// Wait for the single warm session, then drain it.
	waitFor(t, 5*time.Second, "pool never warmed up", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 1
	})
	if _, err := pool.Acquire(context.Background(), time.Second); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	start := time.Now()
	if _, err := pool.Acquire(context.Background(), 80*time.Millisecond); err != dsproxy.ErrPoolTimeout {
		t.Fatalf("Acquire = %v, want ErrPoolTimeout", err)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("timeout waited too long: %s", elapsed)
	}
}

// TestSessionPoolSizeClamp: a non-positive size falls back to the default
// batch (defaultPoolSize, currently 5).
func TestSessionPoolSizeClamp(t *testing.T) {
	pool := dsproxy.NewSessionPool(discardLogger(), &stubBackend{}, 0)
	if pool.Size() != 5 {
		t.Fatalf("Size() = %d, want default 5", pool.Size())
	}
}

// TestSessionPoolShutdownIdempotent: calling Shutdown twice must not panic or
// double-clear.
func TestSessionPoolShutdownIdempotent(t *testing.T) {
	pool, backend := newTestPool(2)
	waitFor(t, 5*time.Second, "pool never warmed up", func() bool {
		created, _ := backend.snapshot()
		return len(created) == 2
	})
	pool.Shutdown()
	pool.Shutdown() // second call is a no-op

	_, deleted := backend.snapshot()
	if len(deleted) != 2 {
		t.Fatalf("deleted = %d, want exactly the 2 pooled sessions", len(deleted))
	}
}
