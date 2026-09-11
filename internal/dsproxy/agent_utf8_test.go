package dsproxy

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// TestAgentStreamInterceptorUTF8Boundary is the regression test for the
// streaming 乱码 bug (issue #2): the interceptor's hold-back window used to
// slice its buffer at a raw byte offset, which could split a multi-byte UTF-8
// character in two. Each emitted piece is passed to json.Marshal, which
// silently replaces invalid UTF-8 with U+FFFD — clients saw garbled CJK text
// in streaming mode while non-streaming (single buffer, never sliced) was
// clean.
func TestAgentStreamInterceptorUTF8Boundary(t *testing.T) {
	// Realistic upstream delta stream mixing CJK and ASCII, long enough that
	// the hold-back window triggers on many different byte offsets.
	deltas := []string{
		"我是", " **", "Deep", "Se", "ek", "模型", "，", "由", "深度", "求", "索",
		"公司", "开发", "的", "。", "量子纠缠", "是", "量子力学中", "最核心",
		"的", "概念", "之一", "。", "详细", "解释", "如下", "：", "测量", "关联",
		"无法", "用", "经典", "物理", "解释", "。",
	}
	var text strings.Builder
	for _, d := range deltas {
		text.WriteString(d)
	}
	want := text.String()

	in := &AgentStreamInterceptor{}
	var got strings.Builder
	for _, d := range deltas {
		parsed := in.Feed(d)
		if parsed.Content == "" && len(parsed.ToolCalls) == 0 {
			continue
		}
		if !utf8.ValidString(parsed.Content) {
			t.Fatalf("Feed emitted content that is not valid UTF-8 (split rune): %q", parsed.Content)
		}
		got.WriteString(parsed.Content)
	}
	final := in.Finish()
	if !utf8.ValidString(final.Content) {
		t.Fatalf("Finish emitted content that is not valid UTF-8 (split rune): %q", final.Content)
	}
	got.WriteString(final.Content)

	if got.String() != want {
		t.Errorf("streamed content drifted from input:\n got: %q\nwant: %q", got.String(), want)
	}
}

// TestAgentStreamInterceptorEveryCutOffset feeds one long CJK text through
// the interceptor so the hold-back cut lands on every byte offset, asserting
// none of them ever splits a rune.
func TestAgentStreamInterceptorEveryCutOffset(t *testing.T) {
	// 60 CJK chars = 180 bytes; the interceptor flushes in pieces of
	// len(rest)-agentStreamKeep bytes, so by varying the prefix length each
	// possible cut offset is exercised.
	base := strings.Repeat("深", 60)
	for shift := 0; shift < 3; shift++ {
		prefix := strings.Repeat("A", shift)
		in := &AgentStreamInterceptor{}
		var out strings.Builder
		parsed := in.Feed(prefix + base)
		if parsed.Content != "" && !utf8.ValidString(parsed.Content) {
			t.Fatalf("shift %d: invalid UTF-8 piece %q", shift, parsed.Content)
		}
		out.WriteString(parsed.Content)
		f := in.Finish()
		if f.Content != "" && !utf8.ValidString(f.Content) {
			t.Fatalf("shift %d: invalid UTF-8 final piece %q", shift, f.Content)
		}
		out.WriteString(f.Content)
		if out.String() != prefix+base {
			t.Errorf("shift %d: round trip mismatch\n got: %q\nwant: %q", shift, out.String(), prefix+base)
		}
	}
}

// TestUTF8TrimToBoundary pins the helper itself.
func TestUTF8TrimToBoundary(t *testing.T) {
	cases := []struct {
		name string
		s    string
		cut  int
		want int
	}{
		{"ascii cut on boundary", "abcdef", 3, 3},
		{"cut lands on continuation byte", "我是Deep", 4, 3}, // 我是 = 6 bytes; cut 4 splits 是
		{"cut on rune start", "我是Deep", 6, 6},
		{"cut inside 4-byte emoji", "a\U0001F600b", 3, 1}, // emoji = 4 bytes at offset 1; safe cut is before it
		{"cut at 0", "深", 0, 0},
		{"cut at len", "深", 3, 3},
	}
	for _, tc := range cases {
		got := utf8TrimToBoundary(tc.s, tc.cut)
		if got != tc.want {
			t.Errorf("%s: utf8TrimToBoundary(%q, %d) = %d, want %d", tc.name, tc.s, tc.cut, got, tc.want)
		}
		if got > 0 && got < len(tc.s) && !utf8.ValidString(tc.s[:got]) {
			t.Errorf("%s: prefix s[:%d] is not valid UTF-8", tc.name, got)
		}
	}
}
