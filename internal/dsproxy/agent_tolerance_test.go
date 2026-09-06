package dsproxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// ── marker finder ────────────────────────────────────────────────────────────

func TestFindAgentMarkerVariants(t *testing.T) {
	cases := []struct {
		in     string
		wantAt int // -1 = no match
		wantLn int
	}{
		{"<<<TOOL_CALL>>>", 0, 15},     // canonical
		{"<<TOOL_CALL>>>", 0, 14},      // live failure: one '<' short
		{"<<<TOOL_CALL>>", 0, 14},      // one '>' short
		{"<<<<TOOL_CALL>>>>", 0, 17},   // worst tolerated spelling
		{"x <<<TOOL_CALL>>> y", 2, 15}, // embedded
		{"<TOOL_CALL>", -1, 0},         // too few brackets
		{"<<<TOOL_CALL>", -1, 0},       // unterminated
		{"TOOL_CALL", -1, 0},           // bare word
		{"<<<<<<<TOOL_CALL>>>", -1, 0}, // bracket run longer than tolerated
	}
	for _, c := range cases {
		at, ln := findAgentMarker(c.in, agentStartWord, true)
		if at != c.wantAt || (c.wantAt >= 0 && ln != c.wantLn) {
			t.Errorf("findAgentMarker(%q) = (%d,%d), want (%d,%d)", c.in, at, ln, c.wantAt, c.wantLn)
		}
	}
	// The TOOL_CALL inside an END marker must never be taken as a start.
	if at, _ := findAgentMarker(agentToolEnd, agentStartWord, true); at != -1 {
		t.Errorf("END marker matched as start at %d", at)
	}
	if n := bracketRunBack("a<<<b", 4, '<'); n != 3 {
		t.Errorf("bracketRunBack = %d, want 3", n)
	}
	if got := agentWorstMarkerLen; got != len("<<<<TOOL_CALL>>>>") {
		t.Errorf("agentWorstMarkerLen = %d, want %d", got, len("<<<<TOOL_CALL>>>>"))
	}

	// Streaming mode: a trailing '>' run touching the end of the data must
	// wait instead of matching short — this keeps the third '>' of a
	// canonical marker from leaking when chunks split the marker.
	for _, tc := range []struct{ in, word string }{
		{"a <<<TOOL_CALL>>", agentStartWord},
		{"x <<<END_TOOL_CALL>>", agentEndWord},
	} {
		if at, _ := findAgentMarker(tc.in, tc.word, false); at != markerIncomplete {
			t.Errorf("findAgentMarker(%q, final=false) = %d, want markerIncomplete", tc.in, at)
		}
	}
	if at, _ := findAgentMarker("<<<TOOL_CALL>> x", agentStartWord, false); at != 0 {
		t.Errorf("terminated short run should match at 0, got %d", at)
	}
}

// ── finished-text parsing ────────────────────────────────────────────────────

// The exact RESPONSE payload reconstructed from the failing debug session
// (deepseek-v4-pro, "Find my public ip").
const malformedPayload = "<<TOOL_CALL>>>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"curl -s https://api.ipify.org\"}}\n<<<END_TOOL_CALL>>>"

func TestParseAgentToolCallsTolerantMarkers(t *testing.T) {
	cases := []struct {
		text     string
		wantArgs string // substring expected inside arguments JSON
	}{
		{malformedPayload, "api.ipify.org"},
		{"Let me check.\n" + malformedPayload + "\n", "api.ipify.org"},
		{"<<<<TOOL_CALL>>>>\n{\"name\":\"bash\",\"arguments\":{}}\n<<<<END_TOOL_CALL>>>>", "{}"},
	}
	for _, c := range cases {
		calls := ParseAgentToolCalls(c.text)
		if len(calls) != 1 {
			t.Fatalf("ParseAgentToolCalls(%q...) => %d calls, want 1", c.text[:12], len(calls))
		}
		fn := calls[0]["function"].(map[string]any)
		if fn["name"] != "bash" {
			t.Errorf("name = %v, want bash", fn["name"])
		}
		args := fn["arguments"].(string)
		if !strings.Contains(args, c.wantArgs) {
			t.Errorf("arguments = %s, want substring %s", args, c.wantArgs)
		}
		if stripped := StripAgentToolCalls(c.text); strings.Contains(stripped, "TOOL_CALL") {
			t.Errorf("StripAgentToolCalls left markers: %q", stripped)
		}
	}
}

func TestNormalizeAgentFencesTolerantMarkers(t *testing.T) {
	in := "```json\n<<TOOL_CALL>>>\n{\"name\":\"bash\",\"arguments\":{}}\n<<<END_TOOL_CALL>>>\n```\ndone"
	got := NormalizeAgentFences(in)
	want := "<<TOOL_CALL>>>\n{\"name\":\"bash\",\"arguments\":{}}\n<<<END_TOOL_CALL>>>\ndone"
	if got != want {
		t.Errorf("normalize:\n got %q\nwant %q", got, want)
	}
}

// ── payload tolerance ────────────────────────────────────────────────────────
//
// Second live failure mode (goal debug session, deepseek-v4-flash, "Find my
// public ip"): the markers were canonical, but the model invented a FLAT
// payload shape — {"tool": "bash", "command": ..., "timeout": 10} instead of
// {"name": ..., "arguments": {...}}. The strict parser accepted the JSON
// with Name == "" and the whole block leaked to the client as plain content
// with finish_reason "stop", so the tool was never executed.

// The exact RESPONSE text reconstructed from the failing debug session.
const flatPayload = `<<<TOOL_CALL>>>{"tool": "bash", "command": "curl -s ifconfig.me", "timeout": 10}<<<END_TOOL_CALL>>>`

func TestParseAgentToolCallsFlatPayload(t *testing.T) {
	calls := ParseAgentToolCalls(flatPayload)
	if len(calls) != 1 {
		t.Fatalf("ParseAgentToolCalls(flat payload) => %d calls, want 1", len(calls))
	}
	fn := calls[0]["function"].(map[string]any)
	if fn["name"] != "bash" {
		t.Errorf("name = %v, want bash", fn["name"])
	}
	args := fn["arguments"].(string)
	if !strings.Contains(args, `"command":"curl -s ifconfig.me"`) || !strings.Contains(args, `"timeout":10`) {
		t.Errorf("arguments = %s, want flat keys folded into an arguments object", args)
	}
	if stripped := StripAgentToolCalls(flatPayload); strings.Contains(stripped, "TOOL_CALL") || strings.TrimSpace(stripped) != "" {
		t.Errorf("StripAgentToolCalls left residue: %q", stripped)
	}
}

func TestParseAgentToolCallsPayloadVariants(t *testing.T) {
	cases := []struct {
		body     string // JSON body between the markers
		wantName string
		wantArgs string // substring the arguments JSON must contain
	}{
		// canonical shape keeps working
		{`{"name":"bash","arguments":{"command":"id"}}`, "bash", `"command":"id"`},
		// alternate explicit-arguments spellings
		{`{"name":"bash","parameters":{"command":"id"}}`, "bash", `"command":"id"`},
		{`{"tool":"bash","arguments":{"command":"id"}}`, "bash", `"command":"id"`},
		// alternate name keys with flat parameters
		{`{"tool_name":"read","path":"/etc/hosts"}`, "read", `"/etc/hosts"`},
		{`{"function":"bash","command":"uname"}`, "bash", `"uname"`},
		// flat payload keyed on "name"
		{`{"name":"bash","command":"pwd","timeout":5}`, "bash", `"command":"pwd"`},
		// a tool parameter literally named "name" must not shadow the tool key
		{`{"tool":"write","name":"a.txt","content":"x"}`, "write", `"content":"x"`},
		// no parameters at all
		{`{"tool":"bash"}`, "bash", `{}`},
	}
	for _, c := range cases {
		text := "<<<TOOL_CALL>>>" + c.body + "<<<END_TOOL_CALL>>>"
		calls := ParseAgentToolCalls(text)
		if len(calls) != 1 {
			t.Errorf("body %s => %d calls, want 1", c.body, len(calls))
			continue
		}
		fn := calls[0]["function"].(map[string]any)
		if fn["name"] != c.wantName {
			t.Errorf("body %s => name %v, want %s", c.body, fn["name"], c.wantName)
		}
		if args := fn["arguments"].(string); !strings.Contains(args, c.wantArgs) {
			t.Errorf("body %s => arguments %s, want substring %s", c.body, args, c.wantArgs)
		}
	}
	// No recognizable tool name at all: stays visible text (existing policy).
	if calls := ParseAgentToolCalls(`<<<TOOL_CALL>>>{"command":"ls"}<<<END_TOOL_CALL>>>`); len(calls) != 0 {
		t.Errorf("nameless payload produced %d calls, want 0", len(calls))
	}
}

// goalFragmentChunks are the upstream SSE fragment appends of the failing
// session, byte for byte (initial RESPONSE fragment content "<<", then every
// APPEND/"v" delta from the debug log).
var goalFragmentChunks = []string{
	"<<", "<", "TO", "OL", "_C", "ALL", ">>>",
	"{\"", "tool", "\":", " \"", "bash", "\",", " \"", "command", "\":",
	" \"", "curl", " -", "s", " if", "config", ".me", "\",", " \"",
	"time", "out", "\":", " ", "10", "}",
	"<<", "<", "END", "_TO", "OL", "_C", "ALL", ">>>",
}

func TestStreamInterceptorReplaysFlatPayloadDebugStream(t *testing.T) {
	for _, step := range []int{1, 3, 7} {
		content, calls, args := feedChunks(t, goalFragmentChunks, step)
		if calls != 1 {
			t.Errorf("step=%d: %d tool calls, want 1 (content=%q)", step, calls, content)
		}
		if !strings.Contains(args, "ifconfig.me") || !strings.Contains(args, `"timeout":10`) {
			t.Errorf("step=%d: arguments %q missing flat payload parameters", step, args)
		}
		if strings.Contains(content, "TOOL_CALL") || strings.TrimSpace(content) != "" {
			t.Errorf("step=%d: markers or junk leaked as content: %q", step, content)
		}
	}
}

// The prompt must pin the payload schema: the bare "{JSON}" placeholder is
// what let the model invent the flat shape in the first place.
func TestAgentPromptPinsPayloadSchema(t *testing.T) {
	prompt := buildAgentPrompt(
		[]chatMessage{{Role: "user", Content: []byte(`"Find my public ip"`)}},
		[]openAITool{{Type: "function", Function: &openAIFnSpec{Name: "bash"}}},
	)
	for _, want := range []string{
		`<<<TOOL_CALL>>>{"name":"<tool_name>","arguments":{<parameter JSON>}}<<<END_TOOL_CALL>>>`,
		`EXACTLY two keys`,
		`"name"`,
		`"arguments"`,
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("agent prompt missing schema fragment %q", want)
		}
	}
	if strings.Contains(prompt, "{JSON}") {
		t.Errorf("agent prompt still carries the underspecified {JSON} placeholder")
	}
}

// ── anti-loop (<already_called>) ─────────────────────────────────────────────

// mkAgentCall builds an assistant message carrying one tool call, the wire
// shape OpenAI clients replay after a previous turn.
func mkAgentCall(id, name, args string) chatMessage {
	return chatMessage{
		Role:    "assistant",
		Content: json.RawMessage(`""`),
		ToolCalls: []assistantToolCall{{
			ID:   id,
			Type: "function",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      name,
				Arguments: json.RawMessage(args),
			},
		}},
	}
}

// TestAlreadyCalledAntiLoop verifies the <already_called> anti-loop section:
// every distinct call already made is listed once — duplicates are collapsed —
// and the section appears late in the prompt (after <recent>, before
// <current_task>) where recency weight is highest, so the model checks it
// against the current task before emitting anything.
func TestAlreadyCalledAntiLoop(t *testing.T) {
	messages := []chatMessage{
		{Role: "user", Content: json.RawMessage(`"find my ip"`)},
		mkAgentCall("call_1", "bash", `{"command":"curl -s https://api.ipify.org"}`),
		{Role: "tool", Content: json.RawMessage(`"203.0.113.7"`), ToolCallID: "call_1"},
		// Same call re-issued by the client (a prior loop) — must dedup.
		mkAgentCall("call_2", "bash", `{"command":"curl -s https://api.ipify.org"}`),
		{Role: "tool", Content: json.RawMessage(`"203.0.113.7"`), ToolCallID: "call_2"},
		// A different call — must be kept.
		mkAgentCall("call_3", "bash", `{"command":"uname -a"}`),
		{Role: "tool", Content: json.RawMessage(`"Linux arm64"`), ToolCallID: "call_3"},
		{Role: "user", Content: json.RawMessage(`"now get the os arch"`)},
	}

	prompt := buildAgentPrompt(messages, []openAITool{{Type: "function", Function: &openAIFnSpec{Name: "bash"}}})

	// Locate the actual section (the tag is also *mentioned* in <system>
	// and <output_rules>, so anchor on the section header line).
	hdr := strings.Index(prompt, "<already_called>\n")
	if hdr < 0 {
		t.Fatal("prompt missing <already_called> section")
	}
	end := strings.Index(prompt[hdr:], "</already_called>")
	if end < 0 {
		t.Fatal("<already_called> section not closed")
	}
	section := prompt[hdr : hdr+end+len("</already_called>")]

	if !strings.Contains(section, "Calls already made in this conversation") {
		t.Error("<already_called> missing its no-repeat header")
	}
	if got := strings.Count(section, "curl -s https://api.ipify.org"); got != 1 {
		t.Errorf("identical repeat call should be listed exactly once in <already_called>, found %d", got)
	}
	if !strings.Contains(section, "- bash {\"command\":\"uname -a\"}") {
		t.Error("distinct call missing from <already_called>")
	}

	// Section order: <recent> … <already_called> … <current_task>.
	iRecent := strings.Index(prompt, "<recent>")
	iTask := strings.Index(prompt, "<current_task>")
	if iRecent < 0 || iTask < 0 || !(iRecent < hdr && hdr < iTask) {
		t.Errorf("<already_called> should sit between <recent> and <current_task> (recent=%d already=%d task=%d)", iRecent, hdr, iTask)
	}
}

// TestAlreadyCalledEmptyWhenNoCalls verifies the <already_called> section is
// omitted entirely when the conversation contains no tool calls (the tag may
// still be *mentioned* by the rules, but the section header must be absent).
func TestAlreadyCalledEmptyWhenNoCalls(t *testing.T) {
	messages := []chatMessage{
		{Role: "user", Content: json.RawMessage(`"hi"`)},
	}
	prompt := buildAgentPrompt(messages, []openAITool{{Type: "function", Function: &openAIFnSpec{Name: "bash"}}})
	if strings.Contains(prompt, "<already_called>\nCalls already made") {
		t.Error("<already_called> section should be omitted when no calls were made")
	}
}

// TestAlreadyCalledAntiLoopRulesPinned mirrors the GLM fix: the <system>
// rules and the <output_rules> final reminder must both carry the no-repeat /
// progress mandates, so the anti-loop contract survives even if the model
// skims past the <already_called> section itself.
func TestAlreadyCalledAntiLoopRulesPinned(t *testing.T) {
	prompt := buildAgentPrompt(
		[]chatMessage{{Role: "user", Content: []byte(`"Find my public ip"`)}},
		[]openAITool{{Type: "function", Function: &openAIFnSpec{Name: "bash"}}},
	)
	for _, want := range []string{
		"NEVER REPEAT: do not re-issue any tool call already listed in <already_called>",
		"PROGRESS: every turn must move the task forward",
		"If the task is fully done, answer with the result in plain text",
		"NO REPEATS: a call already listed in <already_called> must not be re-issued",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("agent prompt missing anti-loop rule %q", want)
		}
	}
}

// ── streaming interceptor ────────────────────────────────────────────────────

// Replays the RESPONSE fragment chunks exactly as they arrived in the failing
// session's upstream SSE log, byte for byte.
var debugFragmentChunks = []string{
	"<<", "<", "TO", "OL", "_C", "ALL", ">>", ">\n",
	"{\"", "name", "\":\"", "bash", "\",\"",
	"arguments", "\":{\"", "command", "\":\"", "curl",
	" -", "s", " https", "://", "api", ".ip", "ify", ".org", "\"", "}}\n",
	"<<", "<", "END", "_TO", "OL", "_C", "ALL", ">>>",
}

func feedChunks(t *testing.T, chunks []string, step int) (content string, toolCalls int, args string) {
	t.Helper()
	in := &AgentStreamInterceptor{}
	var b strings.Builder
	for i := 0; i < len(chunks); i += step {
		piece := ""
		for j := i; j < i+step && j < len(chunks); j++ {
			piece += chunks[j]
		}
		parsed := in.Feed(piece)
		b.WriteString(parsed.Content)
		for _, call := range parsed.ToolCalls {
			fn := call["function"].(map[string]any)
			args, _ = fn["arguments"].(string)
			toolCalls++
		}
	}
	final := in.Finish()
	for _, call := range final.ToolCalls {
		fn := call["function"].(map[string]any)
		args, _ = fn["arguments"].(string)
		toolCalls++
	}
	return b.String() + final.Content, toolCalls, args
}

func TestStreamInterceptorReplaysMalformedDebugStream(t *testing.T) {
	for _, step := range []int{1, 3, 7} { // one chunk / log-like bursts / coarse
		content, calls, args := feedChunks(t, debugFragmentChunks, step)
		if calls != 1 {
			t.Errorf("step=%d: %d tool calls, want 1 (content=%q)", step, calls, content)
		}
		if !strings.Contains(args, "api.ipify.org") {
			t.Errorf("step=%d: arguments %q missing ipify command", step, args)
		}
		if strings.Contains(content, "TOOL_CALL") || strings.TrimSpace(content) != "" {
			t.Errorf("step=%d: markers or junk leaked as content: %q", step, content)
		}
	}
}

func TestStreamInterceptorShortTailVariant(t *testing.T) {
	stream := "checking…\n```json\n<<<TOOL_CALL>>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"uname\"}}\n<<<END_TOOL_CALL>>>\n```\n"
	content, calls, _ := feedChunks(t, splitRunes(stream, 3), 1)
	if calls != 1 {
		t.Errorf("%d tool calls, want 1", calls)
	}
	if strings.Contains(content, "```") || strings.Contains(content, "TOOL_CALL") {
		t.Errorf("leaked: %q", content)
	}
	if !strings.Contains(content, "checking") {
		t.Errorf("prose damaged: %q", content)
	}
}

// An unterminated opening marker must not hang the stream nor be parsed.
func TestStreamInterceptorUnterminatedMarkerStaysContent(t *testing.T) {
	in := &AgentStreamInterceptor{}
	parsed := in.Feed("<TOOL_CALL> just talking about tools")
	final := in.Finish()
	out := parsed.Content + final.Content
	if len(parsed.ToolCalls)+len(final.ToolCalls) != 0 {
		t.Errorf("single-bracket text produced tool calls")
	}
	if out != "<TOOL_CALL> just talking about tools" {
		t.Errorf("single-bracket text altered: %q", out)
	}
}

// Invalid JSON inside a recognized block stays visible text (existing policy).
func TestStreamInvalidBlockLeaksAsContent(t *testing.T) {
	content, calls, _ := feedChunks(t, splitRunes("<<TOOL_CALL>>>\nnot json\n<<<END_TOOL_CALL>>>", 4), 1)
	if calls != 0 {
		t.Errorf("%d calls, want 0 for invalid block", calls)
	}
	if !strings.Contains(content, "not json") {
		t.Errorf("invalid block vanished instead of leaking: %q", content)
	}
}

func TestStreamTwoSequentialBlocks(t *testing.T) {
	stream := "<<TOOL_CALL>>>\n{\"name\":\"a\",\"arguments\":{}}\n<<<END_TOOL_CALL>>>\n<<TOOL_CALL>>>\n{\"name\":\"b\",\"arguments\":{}}\n<<<END_TOOL_CALL>>>"
	in := &AgentStreamInterceptor{}
	names := []string{}
	collect := func(parsed AgentParsedChunk) {
		for _, call := range parsed.ToolCalls {
			names = append(names, call["function"].(map[string]any)["name"].(string))
		}
	}
	for i := 0; i < len(stream); i += 5 {
		end := i + 5
		if end > len(stream) {
			end = len(stream)
		}
		collect(in.Feed(stream[i:end]))
	}
	collect(in.Finish())
	final := in.Finish()
	if strings.Contains(final.Content, "TOOL_CALL") {
		t.Errorf("trailing markers flushed: %q", final.Content)
	}
	if len(names) != 2 || names[0] != "a" || names[1] != "b" {
		t.Errorf("names = %v, want [a b]", names)
	}
}

// splitRunes cuts s into pieces of n bytes (ASCII input assumed).
func splitRunes(s string, n int) []string {
	var out []string
	for i := 0; i < len(s); i += n {
		end := i + n
		if end > len(s) {
			end = len(s)
		}
		out = append(out, s[i:end])
	}
	return out
}
