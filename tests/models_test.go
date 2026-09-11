package tests

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"deepseek/internal/dsproxy"
)

// discardLogger keeps proxy logs out of test output.
func discardLogger() *log.Logger {
	return log.New(io.Discard, "", 0)
}

// ── registry ─────────────────────────────────────────────────────────────────

func mustModel(t *testing.T, id string) dsproxy.Model {
	t.Helper()
	m, err := dsproxy.ResolveModel(id)
	if err != nil {
		t.Fatalf("ResolveModel(%q): %v", id, err)
	}
	return m
}

// TestRegistryPinsTheSingleModel locks down the served model set and its
// capabilities and upstream model_type mapping (deepseek v4.1 serves exactly
// one model; the "expert" class no longer exists).
func TestRegistryPinsTheSingleModel(t *testing.T) {
	// Pin the wire value of the model_type constant.
	if got := string(dsproxy.ModelTypeDefault); got != "default" {
		t.Errorf("ModelTypeDefault = %q, want \"default\"", got)
	}

	ids := dsproxy.SupportedModelIDs()
	if len(ids) != 1 || ids[0] != "deepseek-v4.1-flash" {
		t.Fatalf("expected exactly one supported model (deepseek-v4.1-flash), got %v", ids)
	}

	flash := mustModel(t, "deepseek-v4.1-flash")
	if !flash.IsDefault {
		t.Error("deepseek-v4.1-flash must be the default model")
	}
	if flash.Type != dsproxy.ModelTypeDefault {
		t.Errorf("v4.1-flash model_type = %q, want \"default\"", flash.Type)
	}
	if !flash.SupportsSearch {
		t.Error("deepseek-v4.1-flash must support web search")
	}
	if !flash.SupportsThink {
		t.Error("deepseek-v4.1-flash must support reasoning")
	}
}

// TestResolveModel covers defaulting, case/whitespace tolerance and the
// unknown-model error.
func TestResolveModel(t *testing.T) {
	if m := mustModel(t, ""); m.ID != "deepseek-v4.1-flash" {
		t.Errorf("empty model must resolve to the default, got %q", m.ID)
	}
	if m := mustModel(t, "  DeepSeek-V4.1-Flash  "); m.ID != "deepseek-v4.1-flash" {
		t.Errorf("matching must be case-insensitive/trimmed, got %q", m.ID)
	}

	_, err := dsproxy.ResolveModel("deepseek-chat")
	if err == nil {
		t.Fatal("legacy name deepseek-chat must no longer resolve")
	}
	for _, want := range []string{"deepseek-chat", "deepseek-v4.1-flash"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q must mention %q", err.Error(), want)
		}
	}

	// The retired v4 model ids must be rejected too.
	for _, gone := range []string{"deepseek-v4-flash", "deepseek-v4-pro"} {
		if _, err := dsproxy.ResolveModel(gone); err == nil {
			t.Errorf("retired model %q must no longer resolve", gone)
		}
	}
}

// TestValidateCapabilities: the single model supports search and reasoning.
func TestValidateCapabilities(t *testing.T) {
	flash := mustModel(t, "deepseek-v4.1-flash")

	if err := flash.ValidateCapabilities(true); err != nil {
		t.Errorf("v4.1-flash + search must pass: %v", err)
	}
	if err := flash.ValidateCapabilities(false); err != nil {
		t.Errorf("v4.1-flash without search must pass: %v", err)
	}
}

// TestProxyModelsAdvertisesRegistry verifies /v1/models data is generated
// from the registry (and stays free of removed legacy entries).
func TestProxyModelsAdvertisesRegistry(t *testing.T) {
	want := map[string]bool{"deepseek-v4.1-flash": false}
	for _, m := range dsproxy.ProxyModels {
		id, _ := m["id"].(string)
		if _, ok := want[id]; !ok {
			t.Errorf("unexpected advertised model %q", id)
			continue
		}
		want[id] = true
		if m["object"] != "model" || m["owned_by"] != "deepseek" {
			t.Errorf("model %q has wrong object/owned_by: %v", id, m)
		}
	}
	for id, seen := range want {
		if !seen {
			t.Errorf("model %q missing from /v1/models payload", id)
		}
	}
	if len(dsproxy.ProxyModels) != 1 {
		t.Fatalf("expected exactly 1 advertised model, got %d", len(dsproxy.ProxyModels))
	}
}

// ── upstream wire format ─────────────────────────────────────────────────────

// TestBuildChatCompletionBody pins the /chat/completion payload sent to
// DeepSeek, especially the always-present model_type field.
func TestBuildChatCompletionBody(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		want      string
	}{
		{"v4.1-flash maps to default", "default", "default"},
		{"empty falls back to default", "", "default"},
	}
	for _, tc := range cases {
		body := dsproxy.BuildChatCompletionBody(dsproxy.ChatParams{
			ChatSessionID:   "sess",
			Prompt:          "hi",
			ModelType:       tc.modelType,
			ThinkingEnabled: true,
			SearchEnabled:   false,
		})
		got, _ := body["model_type"].(string)
		if got != tc.want {
			t.Errorf("%s: model_type = %q, want %q", tc.name, got, tc.want)
		}
	}

	body := dsproxy.BuildChatCompletionBody(dsproxy.ChatParams{
		ChatSessionID:   "sess",
		Prompt:          "hi",
		ModelType:       "default",
		ThinkingEnabled: true,
	})
	if body["thinking_enabled"] != true {
		t.Error("thinking_enabled must be forwarded")
	}
	if v, ok := body["parent_message_id"]; !ok || v != nil {
		t.Errorf("absent parent must serialize as null, got %v (ok=%v)", v, ok)
	}
	if _, ok := body["ref_file_ids"]; !ok {
		t.Error("ref_file_ids must stay present")
	}
}

// ── end-to-end rejection paths via ServeHTTP ────────────────────────────────

// postChat fires one chat completion request at the proxy (auth disabled).
func postChat(t *testing.T, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	srv := dsproxy.NewProxyServer(discardLogger(), "")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response not JSON: %v (%q)", err, rec.Body.String())
	}
	return rec, resp
}

func errorCode(t *testing.T, resp map[string]any) string {
	t.Helper()
	errObj, _ := resp["error"].(map[string]any)
	code, _ := errObj["code"].(string)
	return code
}

// TestHandleChatRejectsUnknownModel: an unregistered model id fails fast with
// model_not_found before any upstream session/PoW work happens — including
// the retired v4 ids.
func TestHandleChatRejectsUnknownModel(t *testing.T) {
	for _, model := range []string{"deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o", "gibberish"} {
		rec, resp := postChat(t, `{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("model %q: status = %d, want 400 (body: %s)", model, rec.Code, rec.Body.String())
		}
		if got := errorCode(t, resp); got != "model_not_found" {
			t.Errorf("model %q: error code = %q, want model_not_found", model, got)
		}
	}
}

// TestHandleChatAcceptsV41Flash: the served model (and a missing model field,
// which defaults to it) passes validation and reaches the upstream path
// (failing only on the missing test token, never on model resolution).
func TestHandleChatAcceptsV41Flash(t *testing.T) {
	for _, model := range []string{"deepseek-v4.1-flash", ""} {
		rec, resp := postChat(t, `{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`)
		if rec.Code == http.StatusBadRequest {
			t.Errorf("model %q: rejected with %q (model validation must pass)", model, errorCode(t, resp))
		}
		if got := errorCode(t, resp); got == "model_not_found" {
			t.Errorf("model %q: must not be model_not_found", model)
		}
	}
}

// TestModelsEndpointListsNewRegistry checks GET /v1/models serves the
// v4.1-flash model.
func TestModelsEndpointListsNewRegistry(t *testing.T) {
	srv := dsproxy.NewProxyServer(discardLogger(), "")
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	var resp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	ids := map[string]bool{}
	for _, m := range resp.Data {
		ids[m.ID] = true
	}
	if !ids["deepseek-v4.1-flash"] {
		t.Fatalf("/v1/models must list deepseek-v4.1-flash, got %v", ids)
	}
	if ids["deepseek-v4-flash"] || ids["deepseek-v4-pro"] || ids["deepseek-chat"] {
		t.Error("retired models must be gone from /v1/models")
	}
	if len(ids) != 1 {
		t.Errorf("expected exactly 1 model, got %v", ids)
	}
}
