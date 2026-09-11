package dsproxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// feedEvents runs a sequence of SSE data payloads through parseStreamData and
// returns every chunk with its type and text.
func feedEvents(t *testing.T, events []string) []Chunk {
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

func chunkText(chunks []Chunk, chunkType string) string {
	var b strings.Builder
	for _, ch := range chunks {
		if ch.Type == chunkType {
			b.WriteString(ch.Content)
		}
	}
	return b.String()
}

// TestParseThinkingStream mirrors the live thinking protocol: a THINK
// tail fragment receives incremental appends, then a full RESPONSE fragment
// is appended with the final answer.
func TestParseThinkingStream(t *testing.T) {
	chunks := feedEvents(t, []string{
		`{"request_message_id":1,"response_message_id":2,"model_type":"default"}`,
		`{"v":{"response":{"message_id":2,"role":"ASSISTANT","thinking_enabled":true,"status":"WIP","fragments":[{"id":2,"type":"THINK","content":"We"}]}}}`,
		`{"p":"response/fragments/-1/content","o":"APPEND","v":" need"}`,
		`{"v":" answer"}`,
		`{"v":" 2."}`,
		`{"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE","content":"2"}]}`,
		`{"p":"response/status","o":"SET","v":"FINISHED"}`,
	})

	if got := chunkText(chunks, "content"); got != "2" {
		t.Errorf("content = %q, want %q", got, "2")
	}
	if got := chunkText(chunks, "reasoning"); got != "We need answer 2." {
		t.Errorf("reasoning = %q, want thinking trace", got)
	}
}

// TestParsePlainContentStream: the non-thinking protocol appends straight to
// response/content and everything lands on the content channel.
func TestParsePlainContentStream(t *testing.T) {
	chunks := feedEvents(t, []string{
		`{"request_message_id":1,"response_message_id":2,"model_type":"default"}`,
		`{"p":"response/content","o":"APPEND","v":"OK"}`,
		`{"p":"response/status","v":"FINISHED"}`,
	})
	if got := chunkText(chunks, "content"); got != "OK" {
		t.Errorf("content = %q, want %q", got, "OK")
	}
	if got := chunkText(chunks, "reasoning"); got != "" {
		t.Errorf("unexpected reasoning: %q", got)
	}
}

// TestParseUpstreamErrorEvent: {"type":"error"} must abort with an error,
// never produce empty content (regression for unsupported_client_by_model).
func TestParseUpstreamErrorEvent(t *testing.T) {
	st := &streamState{}
	var data map[string]any
	raw := `{"type":"error","content":"Update to the latest version to use Expert.","finish_reason":"unsupported_client_by_model"}`
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		t.Fatal(err)
	}
	_, _, err := parseStreamData(data, st)
	if err == nil {
		t.Fatal("error event must return an error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "Update to the latest version") || !strings.Contains(msg, "unsupported_client_by_model") {
		t.Errorf("error should carry upstream message and finish_reason, got %q", msg)
	}
}

// TestParseTailFragmentTypeSwitch: an explicit SET on fragments/-1/type
// reroutes subsequent bare-v appends.
func TestParseTailFragmentTypeSwitch(t *testing.T) {
	chunks := feedEvents(t, []string{
		`{"v":{"response":{"message_id":2,"role":"ASSISTANT","status":"WIP","fragments":[{"id":2,"type":"THINK","content":""}]}}}`,
		`{"p":"response/fragments/-1/content","o":"APPEND","v":"hmm"}`,
		`{"p":"response/fragments/-1/type","o":"SET","v":"RESPONSE"}`,
		`{"v":"final"}`,
	})
	if got := chunkText(chunks, "reasoning"); got != "hmm" {
		t.Errorf("reasoning = %q, want %q", got, "hmm")
	}
	if got := chunkText(chunks, "content"); got != "final" {
		t.Errorf("content = %q, want %q", got, "final")
	}
}
