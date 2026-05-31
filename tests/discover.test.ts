import { afterEach, describe, expect, it, vi } from "vitest";
import {
  anthropicBaseUrl,
  buildCompat,
  discoverModels,
  isAdaptiveThinkingModel,
  isAnthropicModel,
  normalizeBaseUrl,
  shouldSuppressReasoningContent,
} from "../src/discover.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("normalizeBaseUrl", () => {
  it("strips trailing slashes", () => {
    expect(normalizeBaseUrl("https://x.example.com/")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com///")).toBe("https://x.example.com");
  });

  it("strips a single trailing /v1 suffix", () => {
    expect(normalizeBaseUrl("https://x.example.com/v1")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com/v1/")).toBe("https://x.example.com");
  });

  it("is case-insensitive on /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/V1")).toBe("https://x.example.com");
  });

  it("does not strip /v2 or /v1xxx", () => {
    expect(normalizeBaseUrl("https://x.example.com/v2")).toBe("https://x.example.com/v2");
    expect(normalizeBaseUrl("https://x.example.com/v1beta")).toBe("https://x.example.com/v1beta");
  });

  it("preserves a base path that is not /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/proxy")).toBe("https://x.example.com/proxy");
  });
});

describe("buildCompat", () => {
  it("returns supportsStore: false for non-anthropic models", () => {
    expect(buildCompat("openai/gpt-4o")).toEqual({ supportsStore: false });
    expect(buildCompat("gemini/gemini-2.0-flash")).toEqual({ supportsStore: false });
    expect(buildCompat("gpt-5.5")).toEqual({ supportsStore: false });
  });

  it("adds Moonshot-compatible tool calling flags for Kimi models", () => {
    expect(buildCompat("kimi-k2.6")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
    expect(buildCompat("moonshotai/kimi-k2")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
  });

  it("adds cacheControlFormat for anthropic-prefixed models", () => {
    expect(buildCompat("anthropic/claude-3-5-sonnet")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("adds cacheControlFormat for bare Claude aliases", () => {
    // Legacy / non-adaptive models keep budget-based thinking (no forceAdaptiveThinking).
    for (const id of ["claude-3-5-sonnet", "haiku-4.5"]) {
      expect(buildCompat(id)).toEqual({
        supportsStore: false,
        cacheControlFormat: "anthropic",
      });
    }
  });

  it("forces adaptive thinking for Opus 4.6/4.7/4.8 and Sonnet 4.6", () => {
    for (const id of ["opus-4.6", "opus-4.7", "opus-4.8", "sonnet-4.6", "claude-opus-4-8"]) {
      expect(buildCompat(id)).toEqual({
        supportsStore: false,
        cacheControlFormat: "anthropic",
        forceAdaptiveThinking: true,
      });
    }
  });

  it("does NOT force adaptive thinking for Haiku 4.5 (it rejects adaptive)", () => {
    expect(buildCompat("claude-haiku-4-5")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("adds cacheControlFormat + adaptive for routed Anthropic aliases", () => {
    expect(buildCompat("google/claude-sonnet-4-6")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
      forceAdaptiveThinking: true,
    });
  });

  it("does not match non-Anthropic tokens that start with Anthropic family names", () => {
    expect(buildCompat("openai/sonnetic-gpt")).toEqual({ supportsStore: false });
    expect(buildCompat("vendor/opusflow")).toEqual({ supportsStore: false });
  });

  it("matches case-insensitively", () => {
    expect(buildCompat("Opus-4.7")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
      forceAdaptiveThinking: true,
    });
    expect(buildCompat("CLAUDE-3-5-SONNET")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });
});

describe("isAdaptiveThinkingModel", () => {
  it("matches Opus 4.6/4.7/4.8 and Sonnet 4.6 across separators and prefixes", () => {
    for (const id of [
      "claude-opus-4-8",
      "claude-opus-4-7",
      "claude-opus-4-6",
      "claude-sonnet-4-6",
      "anthropic/claude-opus-4.8",
      "opus-4-8",
      "Opus-4.7",
      "google/claude-sonnet-4-6",
    ]) {
      expect(isAdaptiveThinkingModel(id)).toBe(true);
    }
  });

  it("does NOT match Haiku 4.5 (rejects adaptive) or older/unrelated models", () => {
    for (const id of [
      "claude-haiku-4-5",
      "haiku-4.5",
      "claude-opus-4-5",
      "claude-sonnet-4-5",
      "claude-opus-4-1",
      "claude-3-5-sonnet",
      "openai/gpt-4o",
      "vendor/opusflow",
    ]) {
      expect(isAdaptiveThinkingModel(id)).toBe(false);
    }
  });
});

describe("shouldSuppressReasoningContent", () => {
  it("suppresses separate reasoning streams for Kimi/Moonshot aliases", () => {
    expect(shouldSuppressReasoningContent("kimi-k2.6")).toBe(true);
    expect(shouldSuppressReasoningContent("moonshotai/kimi-k2")).toBe(true);
  });

  it("does not suppress explicit forced-thinking models", () => {
    expect(shouldSuppressReasoningContent("kimi-k2-thinking")).toBe(false);
  });

  it("does not suppress unrelated models", () => {
    expect(shouldSuppressReasoningContent("openai/gpt-4o")).toBe(false);
  });
});

describe("discoverModels via /model/info", () => {
  it("parses a /model/info success response with cost mapping", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                max_output_tokens: 8192,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
            {
              model_name: "openai/gpt-4o",
              model_info: {
                mode: "chat",
                max_input_tokens: 128000,
                max_output_tokens: 16384,
              },
            },
            {
              model_name: "openai/text-embedding-3-large",
              model_info: { mode: "embedding" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    // embedding model filtered out by mode !== "chat"
    expect(result.models).toHaveLength(2);

    const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet");
    expect(anthropic).toMatchObject({
      id: "anthropic/claude-3-5-sonnet",
      name: "anthropic/claude-3-5-sonnet",
      contextWindow: 200000,
      maxTokens: 8192,
      input: ["text", "image"],
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
      // Claude models route to LiteLLM's native Anthropic /v1/messages passthrough
      // so image input is not lossily re-translated from openai-completions.
      api: "anthropic-messages",
      baseUrl: "https://litellm.example.com/anthropic",
    });
    // cost is per-token in LiteLLM, per-million-tokens in pi-ai
    expect(anthropic?.cost).toEqual({ input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 });

    const openai = result.models.find((m) => m.id === "openai/gpt-4o");
    expect(openai).toMatchObject({
      id: "openai/gpt-4o",
      input: ["text"],
      compat: { supportsStore: false },
    });
    // Non-Anthropic models keep the provider-default openai-completions routing.
    expect(openai?.api).toBeUndefined();
    expect(openai?.baseUrl).toBeUndefined();
  });
});

describe("anthropic native routing", () => {
  const ENV_KEY = "LITELLM_ANTHROPIC_NATIVE";
  const original = process.env[ENV_KEY];
  afterEach(() => {
    if (original === undefined) delete process.env[ENV_KEY];
    else process.env[ENV_KEY] = original;
  });

  it("identifies Anthropic-backed model ids", () => {
    expect(isAnthropicModel("claude-opus-4-8")).toBe(true);
    expect(isAnthropicModel("anthropic/claude-3-5-sonnet")).toBe(true);
    expect(isAnthropicModel("google/claude-sonnet-4-6")).toBe(true);
    expect(isAnthropicModel("opus-4.7")).toBe(true);
    expect(isAnthropicModel("openai/gpt-4o")).toBe(false);
    expect(isAnthropicModel("gemini-2.5-pro")).toBe(false);
  });

  it("derives the native Anthropic passthrough base", () => {
    expect(anthropicBaseUrl("https://litellm.example.com")).toBe("https://litellm.example.com/anthropic");
  });

  it("does not override routing when LITELLM_ANTHROPIC_NATIVE=0", async () => {
    process.env[ENV_KEY] = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat", supports_vision: true } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    const claude = result.models.find((m) => m.id === "claude-opus-4-8");
    expect(claude?.api).toBeUndefined();
    expect(claude?.baseUrl).toBeUndefined();
    // cache_control + adaptive-thinking compat are still applied regardless of the
    // native-routing flag (they describe the model, not the transport).
    expect(claude?.compat).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
      forceAdaptiveThinking: true,
    });
  });
});

describe("discoverModels fallback to /v1/models", () => {
  for (const status of [401, 403, 404]) {
    it(`falls back when /model/info returns ${status}`, async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof URL ? input.toString() : String(input);
        if (url.endsWith("/model/info")) return new Response(null, { status });
        if (url.endsWith("/v1/models")) {
          return jsonResponse(200, {
            data: [{ id: "openai/gpt-4o" }, { id: "anthropic/claude-3-5-sonnet" }],
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.source).toBe("models_list");
      expect(result.models.map((m) => m.id).sort()).toEqual(["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]);
      const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet")!;
      expect(anthropic.name).toBe("anthropic/claude-3-5-sonnet (no metadata)");
      expect(anthropic.compat).toEqual({ supportsStore: false, cacheControlFormat: "anthropic" });
      // Native Anthropic routing is also applied on the /v1/models fallback path.
      expect(anthropic.api).toBe("anthropic-messages");
      expect(anthropic.baseUrl).toBe("https://litellm.example.com/anthropic");
    });
  }

  it("uses models.dev metadata when LiteLLM returns provider ownership", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
        });
      }
      if (url === "https://models.dev/api.json") {
        return jsonResponse(200, {
          openai: {
            models: {
              "gpt-5.5": {
                id: "gpt-5.5",
                name: "GPT-5.5",
                reasoning: true,
                modalities: { input: ["text", "image", "pdf"], output: ["text"] },
                limit: { context: 1_050_000, input: 922_000, output: 128_000 },
                cost: { input: 5, output: 30, cache_read: 0.5 },
              },
            },
          },
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toContain("https://models.dev/api.json");
    expect(result.source).toBe("models_list");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      reasoning: true,
      thinkingLevelMap: { off: "none", xhigh: "xhigh" },
      input: ["text", "image"],
      contextWindow: 1050000,
      maxTokens: 128000,
      compat: { supportsStore: false },
    });
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 30, cacheRead: 0.5, cacheWrite: 0 });
  });

  it("throws when /model/info returns a non-401/403/404 error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(/500/);
  });
});

describe("discoverModels timeout", () => {
  it("aborts the fetch after timeoutMs", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as { signal?: AbortSignal } | undefined)?.signal;
        signal?.addEventListener("abort", () => reject(signal.reason ?? new Error("aborted")));
      });
    });
    const start = Date.now();
    await expect(discoverModels("https://litellm.example.com", "sk-test", { timeoutMs: 30 })).rejects.toBeDefined();
    expect(Date.now() - start).toBeLessThan(500);
  });
});
