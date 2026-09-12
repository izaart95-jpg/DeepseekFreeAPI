package dsproxy

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// Run loads .env (real environment variables always win), applies the
// --debug/--agent-mode/--sync-mode/--legacy-pool flags and DEBUG/AGENT_MODE/
// SYNC_MODE/SESSION_FLOW variables, then starts the OpenAI-compatible proxy.
//
// Stateless (history=false) requests run through the stealth reuse flow by
// default: ONE session, created lazily on the first request and reused via
// /chat/edit_message, so there is no startup burst of session creations and
// no create/delete churn per request — the traffic pattern DeepSeek's bot
// detector was flagging.
//
// Backward compatibility:
//   - --sync-mode / SYNC_MODE=true       -> legacy one-session-per-request
//   - --legacy-pool / SESSION_FLOW=pool  -> legacy pre-warmed session batch
//   - SESSION_FLOW=stealth (default)     -> this new flow (explicit)
func Run() {
	envFile := loadDotEnv()

	debugMode = false
	agentMode = false
	syncMode := false
	legacyPool := false
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--debug":
			debugMode = true
		case "--agent-mode":
			agentMode = true
		case "--sync-mode":
			syncMode = true
		case "--legacy-pool":
			legacyPool = true
		}
	}
	if boolEnv("DEBUG") {
		debugMode = true
	}
	if boolEnv("AGENT_MODE") {
		agentMode = true
	}
	if boolEnv("SYNC_MODE") {
		syncMode = true
	}

	// SESSION_FLOW picks the stateless flow explicitly (env-var form of
	// the mode flags; the flags win when both are given).
	switch strings.ToLower(strings.TrimSpace(envOr("SESSION_FLOW", ""))) {
	case "stealth", "reuse", "edit":
		// nothing: this is the default
	case "pool", "legacy-pool", "async":
		legacyPool = true
	case "sync", "legacy", "per-request":
		syncMode = true
	}

	port := envOr("PORT", "3000")
	proxyKey := envOr("PROXY_API_KEY", "Waguri")
	token := os.Getenv("DEEPSEEK_TOKEN")

	logger := log.New(os.Stdout, "", log.LstdFlags)
	logger.Printf("DeepSeek OpenAI Proxy  v1 (Go port)")
	logger.Printf("  Port      : %s", port)
	logger.Printf("  Proxy key : %s", proxyKey)
	if envFile != "" {
		logger.Printf("  Env file  : %s", envFile)
	}
	if token == "" {
		logger.Printf("  DS token  : NOT SET  <- put DEEPSEEK_TOKEN=... in .env or export it")
	} else {
		logger.Printf("  DS token  : SET")
	}
	logger.Printf("  Debug     : %v", debugMode)
	if agentMode {
		logger.Printf("  Agent mode: ENABLED (OpenAI tools/roles translated, tool calls intercepted)")
	} else {
		logger.Printf("  Agent mode: disabled")
	}

	proxy := NewProxyServer(logger, proxyKey)

	// The reuse manager owns the single lazily-created stateless session.
	// It exists in every mode (cheap), but only the stealth flow uses it.
	reuse := NewReuseManager(logger, &lazyReuseBackend{s: proxy})

	var pool *SessionPool
	switch {
	case syncMode:
		proxy.EnableSyncFlow()
		logger.Printf("  Flow      : sync (--sync-mode: legacy, one session created per request)")
	case legacyPool:
		poolSize := intEnvOr("SESSION_POOL_SIZE", defaultPoolSize)
		waitSecs := intEnvOr("SESSION_ACQUIRE_TIMEOUT", int(defaultPoolWait/time.Second))
		proxy.poolWait = time.Duration(waitSecs) * time.Second
		if waitSecs <= 0 {
			proxy.poolWait = 0 // 0 => wait indefinitely for a pooled session
		}
		pool = NewSessionPool(logger, &lazyBackend{s: proxy}, poolSize)
		proxy.AttachSessionPool(pool)
		logger.Printf("  Flow      : legacy pool (pre-made session batch x%d)", pool.Size())
		logger.Printf("              SESSION_POOL_SIZE=%d SESSION_ACQUIRE_TIMEOUT=%ds", pool.Size(), waitSecs)
	default:
		proxy.AttachReuseManager(reuse)
		logger.Printf("  Flow      : stealth reuse (1 lazy session, /chat/edit_message reuse)")
		logger.Printf("              anti-detection default; --legacy-pool or --sync-mode restores old behavior")
	}

	logger.Printf("  Endpoints : POST /v1/chat/completions")
	logger.Printf("              GET  /v1/models")
	logger.Printf("              GET  /history?enable=true|false")
	logger.Printf("              POST /new")

	server := &http.Server{
		Addr:    "0.0.0.0:" + port,
		Handler: proxy,
	}

	// Start serving before blocking on signals.
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- server.ListenAndServe()
	}()

	// The stealth flow warms nothing at boot (its session is created lazily
	// on the first request — a boot burst is a bot signature). Only the
	// legacy pool pre-warms its standing batch.
	if pool != nil {
		pool.Start()
	}

	logger.Printf("Ready -> http://0.0.0.0:%s/v1", port)

	ctx, stopSignal := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignal()

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatalf("server error: %v", err)
		}
	case <-ctx.Done():
		// Respectful stop (CTRL+C / SIGTERM). Re-arm default signal handling
		// so a second interrupt force-exits immediately.
		stopSignal()
		logger.Printf("Shutdown requested — draining connections and clearing all sessions...")

		drainCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := server.Shutdown(drainCtx); err != nil {
			logger.Printf("drain deadline hit (%v); closing remaining connections", err)
			_ = server.Close()
		}
		cancel()

		// Clear whatever stateless sessions the active flow still holds so
		// nothing is left behind on the DeepSeek account (in-flight requests
		// retire their own sessions through their lease/Release).
		if pool != nil {
			pool.Shutdown()
		}
		reuse.Shutdown()
		logger.Printf("All sessions cleared. Goodbye.")
	}
}
