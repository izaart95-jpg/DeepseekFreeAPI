package dsproxy

import (
	"fmt"
	"strings"
)

// ============== MODEL REGISTRY ==============
// The served model is real configuration, not a display label: it maps to the
// "model_type" value POSTed to DeepSeek's /chat/completion and carries the
// capability set the proxy enforces before any upstream call.
//
// As of DeepSeek v4.1 the upstream backend serves a single model,
// deepseek-v4.1-flash, which supports both thinking and web search and is
// selected with model_type "default" — no other model_type exists anymore.

// ModelType is the value sent upstream as "model_type" in the
// /chat/completion JSON body. It selects the backend model class on
// chat.deepseek.com.
type ModelType string

const (
	// ModelTypeDefault is the only model_type the v4.1 backend accepts; it
	// serves deepseek-v4.1-flash (thinking + search capable).
	ModelTypeDefault ModelType = "default"
)

// Model describes one OpenAI-facing model and how it maps onto the DeepSeek
// chat backend.
type Model struct {
	ID             string    // id clients put in the request's "model" field
	Type           ModelType // value sent upstream as "model_type"
	SupportsSearch bool      // web search may be enabled on this model
	SupportsThink  bool      // reasoning (thinking) may be enabled on this model
	IsDefault      bool      // served when the request omits "model"
}

// modelRegistry is the single source of truth for every model this proxy
// serves. Order defines the /v1/models listing order.
var modelRegistry = []Model{
	{
		ID:             "deepseek-v4.1-flash",
		Type:           ModelTypeDefault,
		SupportsSearch: true,
		SupportsThink:  true,
		IsDefault:      true,
	},
}

// DefaultModel returns the model served when a request omits "model".
func DefaultModel() Model {
	for _, m := range modelRegistry {
		if m.IsDefault {
			return m
		}
	}
	return modelRegistry[0]
}

// LookupModel resolves an exact (already normalized) model id against the
// registry.
func LookupModel(id string) (Model, bool) {
	for _, m := range modelRegistry {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// ResolveModel turns the client-supplied "model" field into a registry entry:
// empty/whitespace resolves to the default model, matching is
// case-insensitive, and anything unknown is an error listing the valid ids.
func ResolveModel(id string) (Model, error) {
	normalized := strings.ToLower(strings.TrimSpace(id))
	if normalized == "" {
		return DefaultModel(), nil
	}
	if m, ok := LookupModel(normalized); ok {
		return m, nil
	}
	return Model{}, fmt.Errorf("unknown model %q (supported models: %s)",
		id, strings.Join(SupportedModelIDs(), ", "))
}

// SupportedModelIDs lists the advertised model ids in registry order.
func SupportedModelIDs() []string {
	ids := make([]string, len(modelRegistry))
	for i, m := range modelRegistry {
		ids[i] = m.ID
	}
	return ids
}

// ValidateCapabilities reports whether the requested feature flags fit the
// model's supported feature set. Called before any upstream session or PoW
// work so invalid combinations fail fast with a 400.
func (m Model) ValidateCapabilities(search bool) error {
	if search && !m.SupportsSearch {
		return fmt.Errorf("model %q does not support web search; use %q for search-enabled requests",
			m.ID, DefaultModel().ID)
	}
	return nil
}

// ── /v1/models payload ───────────────────────────────────────────────────────

const modelListCreated = 1700000000

// ProxyModels is the data array returned by GET /v1/models, generated from
// the registry so the advertised list can never drift from actual behavior.
var ProxyModels = buildProxyModels()

func buildProxyModels() []map[string]any {
	models := make([]map[string]any, len(modelRegistry))
	for i, m := range modelRegistry {
		models[i] = map[string]any{
			"id":       m.ID,
			"object":   "model",
			"created":  modelListCreated + int64(i),
			"owned_by": "deepseek",
		}
	}
	return models
}
