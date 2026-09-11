package dsproxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const deepseekBaseURL = "https://chat.deepseek.com/api/v0"

const deepseekUserAgent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

// Error types mirroring api.py's DeepSeekError hierarchy.
type AuthenticationError struct{ Msg string }

func (e AuthenticationError) Error() string { return e.Msg }

type RateLimitError struct{ Msg string }

func (e RateLimitError) Error() string { return e.Msg }

type NetworkError struct{ Msg string }

func (e NetworkError) Error() string { return e.Msg }

type CloudflareError struct{ Msg string }

func (e CloudflareError) Error() string { return e.Msg }

type APIError struct {
	Msg        string
	StatusCode int
}

func (e APIError) Error() string { return e.Msg }

// Chunk is one yielded event of the chat completion stream
// (mirrors api.py chat_completion() generator output).
type Chunk struct {
	Type              string  `json:"type"`
	RequestMessageID  any     `json:"request_message_id,omitempty"`
	ResponseMessageID any     `json:"response_message_id,omitempty"`
	UpdatedAt         float64 `json:"updated_at,omitempty"`
	Content           string  `json:"content,omitempty"`
	FullContent       string  `json:"full_content,omitempty"`
	FragmentID        string  `json:"fragment_id,omitempty"`
	StageID           string  `json:"stage_id,omitempty"`
	Status            string  `json:"status,omitempty"`
	FinishReason      string  `json:"finish_reason,omitempty"`
}

// DeepSeekAPI is the client for chat.deepseek.com (port of api.py).
type DeepSeekAPI struct {
	authToken string
	pow       *DeepSeekPOW
	cookies   map[string]string
	client    *http.Client
}

// NewDeepSeekAPI builds the client, loads optional cookies.json and
// initializes the WASM PoW solver.
func NewDeepSeekAPI(authToken string) (*DeepSeekAPI, error) {
	if authToken == "" {
		return nil, AuthenticationError{Msg: "Invalid auth token provided"}
	}
	pow, err := NewDeepSeekPOW()
	if err != nil {
		return nil, fmt.Errorf("pow solver init: %w", err)
	}
	c := &DeepSeekAPI{
		authToken: authToken,
		pow:       pow,
		cookies:   map[string]string{},
		client:    &http.Client{}, // no timeout, mirrors Python timeout=None
	}
	c.loadCookies()
	return c, nil
}

func (c *DeepSeekAPI) cookiesPath() string {
	dir := "."
	if exe, err := os.Executable(); err == nil {
		if d := filepath.Dir(exe); d != "" {
			dir = d
		}
	}
	return filepath.Join(dir, "cookies.json")
}

func (c *DeepSeekAPI) loadCookies() {
	path := c.cookiesPath()
	raw, err := os.ReadFile(path)
	if err != nil {
		warnf("Warning: Could not load cookies from %s: %v", path, err)
		return
	}
	var data struct {
		Cookies map[string]string `json:"cookies"`
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		warnf("Warning: Could not load cookies from %s: %v", path, err)
		return
	}
	c.cookies = data.Cookies
}

// refreshCookies reloads cookies.json from disk (cookies must be refreshed
// externally, e.g. exported from a browser after clearing Cloudflare).
func (c *DeepSeekAPI) refreshCookies() {
	c.loadCookies()
}

// baseHeaders mirrors the current chat.deepseek.com web client (Chrome/149,
// x-client-version 2.4.0). The old "x-app-version" header is gone from the
// protocol, and stale version headers used to make the backend reject
// requests with unsupported_client_by_model.
func (c *DeepSeekAPI) baseHeaders(powResponse string) http.Header {
	h := http.Header{}
	h.Set("accept", "*/*")
	h.Set("accept-language", "en-US,en;q=0.9,ur-IN;q=0.8,ur-PK;q=0.7,ur;q=0.6")
	h.Set("authorization", "Bearer "+c.authToken)
	h.Set("content-type", "application/json")
	h.Set("origin", "https://chat.deepseek.com")
	h.Set("priority", "u=1, i")
	h.Set("referer", "https://chat.deepseek.com/")
	h.Set("sec-ch-ua", `"Chromium";v="149", "Not)A;Brand";v="24"`)
	h.Set("sec-ch-ua-mobile", "?0")
	h.Set("sec-ch-ua-platform", `"Linux"`)
	h.Set("sec-fetch-dest", "empty")
	h.Set("sec-fetch-mode", "cors")
	h.Set("sec-fetch-site", "same-origin")
	h.Set("user-agent", deepseekUserAgent)
	h.Set("x-client-bundle-id", "com.deepseek.chat")
	h.Set("x-client-locale", "en_US")
	h.Set("x-client-platform", "web")
	h.Set("x-client-timezone-offset", "19800")
	h.Set("x-client-version", "2.4.0")
	if powResponse != "" {
		h.Set("x-ds-pow-response", powResponse)
	}
	return h
}

// makeRequest performs a JSON POST with Cloudflare detection + retries
// (port of api.py _make_request).
func (c *DeepSeekAPI) makeRequest(ctx context.Context, method, endpoint string, jsonData map[string]any, powRequired bool) (map[string]any, error) {
	url := deepseekBaseURL + endpoint
	maxRetries := 2

	for retry := 0; retry < maxRetries; retry++ {
		var powResponse string
		if powRequired {
			challenge, err := c.GetPowChallenge(ctx)
			if err != nil {
				return nil, err
			}
			powResponse, err = c.pow.SolveChallenge(*challenge)
			if err != nil {
				return nil, APIError{Msg: fmt.Sprintf("PoW solve failed: %v", err)}
			}
			if debugMode {
				debugf("pow challenge: %s", debugJSON(*challenge))
				debugf("pow response: %s", powResponse)
			}
		}

		body, err := json.Marshal(jsonData)
		if err != nil {
			return nil, APIError{Msg: fmt.Sprintf("Request marshal failed: %v", err)}
		}
		req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
		if err != nil {
			return nil, NetworkError{Msg: fmt.Sprintf("Network error occurred: %v", err)}
		}
		req.Header = c.baseHeaders(powResponse)
		for k, v := range c.cookies {
			req.AddCookie(&http.Cookie{Name: k, Value: v})
		}
		if debugMode {
			debugDumpOutgoingRequest(method, url, req.Header, body)
		}

		resp, err := c.client.Do(req)
		if err != nil {
			return nil, NetworkError{Msg: fmt.Sprintf("Network error occurred: %v", err)}
		}
		respBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, NetworkError{Msg: fmt.Sprintf("Network error occurred: %v", readErr)}
		}
		if debugMode {
			debugDumpIncomingResponse(resp.StatusCode, resp.Header, respBody)
		}

		text := string(respBody)
		if strings.Contains(text, "<!DOCTYPE html>") && strings.Contains(text, "Just a moment") {
			warnf("Warning: Cloudflare protection detected. Bypassing...")
			if retry < maxRetries-1 {
				c.refreshCookies()
				continue
			}
		}

		switch {
		case resp.StatusCode == 401:
			return nil, AuthenticationError{Msg: "Invalid or expired authentication token"}
		case resp.StatusCode == 429:
			return nil, RateLimitError{Msg: "API rate limit exceeded"}
		case resp.StatusCode >= 500:
			return nil, APIError{Msg: fmt.Sprintf("Server error occurred: %s", text), StatusCode: resp.StatusCode}
		case resp.StatusCode != 200:
			return nil, APIError{Msg: fmt.Sprintf("API request failed: %s", text), StatusCode: resp.StatusCode}
		}

		var parsed map[string]any
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return nil, APIError{Msg: "Invalid JSON response from server"}
		}
		return parsed, nil
	}

	return nil, APIError{Msg: "Failed to bypass Cloudflare protection after multiple attempts"}
}

// GetPowChallenge fetches a fresh PoW challenge from the API.
func (c *DeepSeekAPI) GetPowChallenge(ctx context.Context) (*Challenge, error) {
	resp, err := c.makeRequest(ctx, "POST", "/chat/create_pow_challenge",
		map[string]any{"target_path": "/api/v0/chat/completion"}, false)
	if err != nil {
		return nil, err
	}
	data, ok := resp["data"].(map[string]any)
	if !ok {
		return nil, APIError{Msg: "Invalid challenge response format from server"}
	}
	bizData, ok := data["biz_data"].(map[string]any)
	if !ok {
		return nil, APIError{Msg: "Invalid challenge response format from server"}
	}
	challengeRaw, err := json.Marshal(bizData["challenge"])
	if err != nil {
		return nil, APIError{Msg: "Invalid challenge response format from server"}
	}
	var challenge Challenge
	if err := json.Unmarshal(challengeRaw, &challenge); err != nil {
		return nil, APIError{Msg: "Invalid challenge response format from server"}
	}
	return &challenge, nil
}

// CreateChatSession creates a new chat session and returns its ID.
func (c *DeepSeekAPI) CreateChatSession(ctx context.Context) (string, error) {
	resp, err := c.makeRequest(ctx, "POST", "/chat_session/create",
		map[string]any{"character_id": nil}, false)
	if err != nil {
		return "", err
	}
	data, ok := resp["data"].(map[string]any)
	if !ok {
		return "", APIError{Msg: "Invalid session creation response format from server"}
	}
	bizData, ok := data["biz_data"].(map[string]any)
	if !ok {
		return "", APIError{Msg: "Invalid session creation response format from server"}
	}
	// Current protocol nests the session object (client >= 2.x):
	//   biz_data.chat_session.id
	// Older builds returned the id directly at biz_data.id.
	if session, ok := bizData["chat_session"].(map[string]any); ok {
		if id, _ := session["id"].(string); id != "" {
			return id, nil
		}
	}
	id, _ := bizData["id"].(string)
	if id == "" {
		return "", APIError{Msg: "Invalid session creation response format from server"}
	}
	return id, nil
}

// DeleteChatSession removes chat sessions server-side so their IDs don't
// accumulate on the account. The endpoint accepts an array of IDs, mirroring:
//
//	POST /chat_session/delete
//	{"chat_session_ids":["d1b09b01-3501-4083-b826-d631b1297f1b"]}
//	-> {"code":0,"msg":"","data":{"biz_code":0,"biz_msg":"","biz_data":null}}
//
// No PoW challenge is required for this endpoint.
func (c *DeepSeekAPI) DeleteChatSession(ctx context.Context, sessionIDs ...string) error {
	if len(sessionIDs) == 0 {
		return nil
	}
	ids := make([]any, len(sessionIDs))
	for i, id := range sessionIDs {
		if id == "" {
			return APIError{Msg: "DeleteChatSession: empty session ID"}
		}
		ids[i] = id
	}
	_, err := c.makeRequest(ctx, "POST", "/chat_session/delete",
		map[string]any{"chat_session_ids": ids}, false)
	return err
}

// ChatParams mirror the JSON body of /chat/completion.
type ChatParams struct {
	ChatSessionID   string
	Prompt          string
	ParentMessageID any
	ThinkingEnabled bool
	SearchEnabled   bool
	// ModelType selects the backend model class on chat.deepseek.com.
	// v4.1 serves a single class: "default" (deepseek-v4.1-flash). Empty
	// resolves to "default".
	ModelType string
}

// BuildChatCompletionBody renders the JSON payload POSTed to DeepSeek's
// /chat/completion. model_type is always present so every request explicitly
// names the backend model class it targets.
func BuildChatCompletionBody(params ChatParams) map[string]any {
	modelType := params.ModelType
	if modelType == "" {
		modelType = string(ModelTypeDefault)
	}
	body := map[string]any{
		"chat_session_id":   params.ChatSessionID,
		"parent_message_id": nil,
		"prompt":            params.Prompt,
		"ref_file_ids":      []any{},
		"thinking_enabled":  params.ThinkingEnabled,
		"search_enabled":    params.SearchEnabled,
		"model_type":        modelType,
	}
	if params.ParentMessageID != nil {
		body["parent_message_id"] = asU32(params.ParentMessageID)
	}
	return body
}

// ChatCompletion streams chunks to onChunk (port of api.py chat_completion).
// onChunk returns true to stop early (e.g. on FINISHED).
func (c *DeepSeekAPI) ChatCompletion(ctx context.Context, params ChatParams, onChunk func(Chunk) bool) error {
	if params.Prompt == "" {
		return fmt.Errorf("Prompt must be a non-empty string")
	}
	if params.ChatSessionID == "" {
		return fmt.Errorf("Chat session ID must be a non-empty string")
	}

	challenge, err := c.GetPowChallenge(ctx)
	if err != nil {
		return err
	}
	powResponse, err := c.pow.SolveChallenge(*challenge)
	if err != nil {
		return APIError{Msg: fmt.Sprintf("PoW solve failed: %v", err)}
	}
	if debugMode {
		debugf("pow challenge: %s", debugJSON(*challenge))
		debugf("pow response: %s", powResponse)
	}

	jsonData := BuildChatCompletionBody(params)

	body, err := json.Marshal(jsonData)
	if err != nil {
		return APIError{Msg: fmt.Sprintf("Request marshal failed: %v", err)}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", deepseekBaseURL+"/chat/completion", bytes.NewReader(body))
	if err != nil {
		return NetworkError{Msg: fmt.Sprintf("Network error occurred during streaming: %v", err)}
	}
	req.Header = c.baseHeaders(powResponse)
	for k, v := range c.cookies {
		req.AddCookie(&http.Cookie{Name: k, Value: v})
	}
	if debugMode {
		debugDumpOutgoingRequest("POST", deepseekBaseURL+"/chat/completion", req.Header, body)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return NetworkError{Msg: fmt.Sprintf("Network error occurred during streaming: %v", err)}
	}
	defer resp.Body.Close()
	if debugMode {
		debugDumpIncomingResponse(resp.StatusCode, resp.Header, nil) // body is streamed below
	}

	if resp.StatusCode != 200 {
		reader := bufio.NewReader(resp.Body)
		firstLine, _ := reader.ReadString('\n')
		errorText := strings.TrimSpace(firstLine)
		switch resp.StatusCode {
		case 401:
			return AuthenticationError{Msg: "Invalid or expired authentication token"}
		case 429:
			return RateLimitError{Msg: "API rate limit exceeded"}
		default:
			return APIError{Msg: fmt.Sprintf("API request failed: %s", errorText), StatusCode: resp.StatusCode}
		}
	}

	reader := bufio.NewReader(resp.Body)
	stream := &streamState{}

	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(line, "event: ") {
				continue
			}
			if strings.HasPrefix(line, "data: ") {
				dataStr := strings.TrimPrefix(line, "data: ")
				if dataStr == "" {
					continue
				}
				debugf("sse <- %s", dataStr)
				var data map[string]any
				if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
					return APIError{Msg: fmt.Sprintf("Invalid JSON in response chunk: %v", err)}
				}
				chunks, complete, err := parseStreamData(data, stream)
				if err != nil {
					return err
				}
				for _, ch := range chunks {
					if onChunk(ch) {
						return nil
					}
				}
				if complete {
					if onChunk(Chunk{Type: "complete", FinishReason: "stop"}) {
						return nil
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return APIError{Msg: fmt.Sprintf("Error parsing chunk: %v", err)}
		}
	}
	return nil
}

// streamState carries per-stream context needed to interpret incremental
// JSON-patch events: deltas append to the LAST fragment
// (response/fragments/-1), so knowing its type tells thinking text apart
// from answer text.
type streamState struct {
	tailFragType string // "THINK", "RESPONSE", ... of the tail fragment
}

// chunkTypeForFrag maps a fragment type to the chunk channel it streams on.
// The backend spells the thinking kind "THINK"; accept any THINK* spelling
// defensively. Everything unknown counts as answer content.
func chunkTypeForFrag(fragType string) string {
	if strings.HasPrefix(strings.ToUpper(fragType), "THINK") {
		return "reasoning"
	}
	return "content"
}

// parseStreamData converts one SSE data payload into chunks (port of the
// event handling in api.py chat_completion, adapted to the live protocol:
// status FINISHED arrives without an "o" field, content appends use
// "response/content" or "response/fragments/-1/content" paths).
//
// Reasoning (thinking) text is yielded as separate "reasoning" chunks so it
// never leaks into the answer content.
func parseStreamData(data map[string]any, st *streamState) ([]Chunk, bool, error) {
	var chunks []Chunk
	complete := false

	// Protocol errors arrive as {"type":"error","content":...,
	// "finish_reason":...} (e.g. unsupported_client_by_model). They must
	// abort the stream with a real error, never degrade to an empty answer.
	if strVal(data, "type") == "error" {
		msg := strings.TrimSpace(strVal(data, "content"))
		if msg == "" {
			msg = "upstream error event"
		}
		if finish := strVal(data, "finish_reason"); finish != "" {
			msg = fmt.Sprintf("%s (finish_reason=%s)", msg, finish)
		}
		return nil, false, APIError{Msg: msg}
	}

	if _, ok := data["request_message_id"]; ok {
		chunks = append(chunks, Chunk{
			Type:              "ready",
			RequestMessageID:  valOrNil(data, "request_message_id"),
			ResponseMessageID: valOrNil(data, "response_message_id"),
		})
		return chunks, false, nil
	}

	if _, ok := data["updated_at"]; ok {
		if v, ok := data["updated_at"].(float64); ok {
			chunks = append(chunks, Chunk{Type: "session_update", UpdatedAt: v})
		}
		return chunks, false, nil
	}

	if _, hasTitle := data["title"]; hasTitle {
		if _, hasContent := data["content"]; hasContent {
			chunks = append(chunks, Chunk{Type: "title", Content: strVal(data, "content")})
			return chunks, false, nil
		}
	}

	if v, ok := data["v"].(map[string]any); ok {
		if response, ok := v["response"].(map[string]any); ok {
			if fragments, ok := response["fragments"].([]any); ok {
				for _, frag := range fragments {
					f, ok := frag.(map[string]any)
					if !ok {
						continue
					}
					content := strVal(f, "content")
					if content == "" {
						continue
					}
					if ct := chunkTypeForFrag(strVal(f, "type")); ct == "reasoning" {
						chunks = append(chunks, Chunk{Type: ct, Content: content})
					} else {
						chunks = append(chunks, Chunk{
							Type:       ct,
							Content:    content,
							FragmentID: strVal(f, "id"),
							StageID:    strVal(f, "stage_id"),
						})
					}
				}
				if n := len(fragments); n > 0 {
					if f, ok := fragments[n-1].(map[string]any); ok {
						st.tailFragType = strVal(f, "type")
					}
				}
			}
			return chunks, false, nil
		}
	}

	path := strVal(data, "p")

	// JSON patch events (carry "p"); status FINISHED may arrive without "o"
	if path != "" {
		if path == "response/status" && strVal(data, "v") == "FINISHED" {
			chunks = append(chunks, Chunk{Type: "status", Status: "FINISHED"})
			return chunks, true, nil
		}
		op := strVal(data, "o")
		// A whole new fragment object arrives with its type: it becomes the
		// new tail fragment, and RESPONSE fragments carry their full content.
		if op == "APPEND" && path == "response/fragments" {
			if frags, ok := data["v"].([]any); ok {
				for _, frag := range frags {
					f, ok := frag.(map[string]any)
					if !ok {
						continue
					}
					st.tailFragType = strVal(f, "type")
					content := strVal(f, "content")
					if content == "" {
						continue
					}
					chunks = append(chunks, Chunk{Type: chunkTypeForFrag(st.tailFragType), Content: content})
				}
			}
			return chunks, false, nil
		}
		// The tail fragment switched kind (e.g. THINKING -> RESPONSE).
		if op == "SET" && path == "response/fragments/-1/type" {
			st.tailFragType = strVal(data, "v")
			return chunks, false, nil
		}
		if op == "APPEND" && path == "response/fragments/-1/content" {
			if v, ok := data["v"].(string); ok {
				chunks = append(chunks, Chunk{Type: chunkTypeForFrag(st.tailFragType), Content: v})
			}
		} else if op == "APPEND" && path == "response/content" {
			if v, ok := data["v"].(string); ok {
				chunks = append(chunks, Chunk{Type: "content", Content: v})
			}
		} else if op == "SET" && path == "response/status" {
			if v, ok := data["v"].(string); ok {
				chunks = append(chunks, Chunk{Type: "status", Status: v})
				if v == "FINISHED" {
					complete = true
				}
			}
		} else if op == "BATCH" {
			if ops, ok := data["v"].([]any); ok {
				for _, item := range ops {
					operation, ok := item.(map[string]any)
					if !ok {
						continue
					}
					if operation["p"] == "response/status" && operation["v"] == "FINISHED" {
						chunks = append(chunks, Chunk{Type: "status", Status: "FINISHED"})
						complete = true
					}
				}
			}
		}
		return chunks, complete, nil
	}

	if v, ok := data["v"].(string); ok {
		// Bare {"v": ...} events continue the previous patch context, i.e.
		// they append to the current tail fragment.
		chunks = append(chunks, Chunk{Type: chunkTypeForFrag(st.tailFragType), Content: v})
		return chunks, false, nil
	}

	return chunks, false, nil
}

func strVal(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

// valOrNil keeps the raw JSON type (string or number) of a field.
func valOrNil(m map[string]any, key string) any {
	v, ok := m[key]
	if !ok {
		return nil
	}
	return v
}

// asU32 normalizes message IDs to the u32 the API expects: numeric strings
// like "2" are converted to int, floats/ints pass through, others are kept.
func asU32(v any) any {
	switch t := v.(type) {
	case string:
		if n, err := strconv.ParseInt(t, 10, 64); err == nil {
			return n
		}
		return t
	case float64:
		if t == float64(int64(t)) {
			return int64(t)
		}
	}
	return v
}

// AsErrorType maps errors to a response type/status like proxy.py.
func errorType(err error) (etype string, status int) {
	var auth AuthenticationError
	var rate RateLimitError
	var cf CloudflareError
	var apiErr APIError
	var netErr NetworkError
	switch {
	case errors.As(err, &auth):
		return "authentication_error", 401
	case errors.As(err, &rate):
		return "rate_limit_error", 429
	case errors.As(err, &cf):
		return "cloudflare_error", 503
	case errors.As(err, &apiErr):
		return "api_error", 502
	case errors.As(err, &netErr):
		return "api_error", 502
	default:
		return "server_error", 500
	}
}
