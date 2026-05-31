import type { Api, KnownProvider, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type {
  DiscoveryOptions,
  DiscoveryResult,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;
const KNOWN_PROVIDER_SET = new Set<string>(getProviders());
const MODELS_DEV_URL = "https://models.dev/api.json";
let modelsDevCatalog: ModelsDevResponse | undefined;

interface ModelsDevModel {
  name?: string;
  reasoning?: boolean;
  modalities?: {
    input?: string[];
  };
  limit?: {
    context?: number;
    input?: number;
    output?: number;
  };
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
  };
}

type ModelsDevResponse = Record<string, { models?: Record<string, ModelsDevModel> }>;

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

// Matches both the conventional `anthropic/...` prefix and aliases that
// LiteLLM deployments commonly assign to Anthropic-backed routes (e.g.
// `google/claude-sonnet-4-6`, `opus-4.7`, `sonnet-4.6`, `haiku-4.5`). Without
// the `cacheControlFormat: "anthropic"` flag, pi never relays cache_control
// markers through the proxy, so prompt caching silently no-ops on Claude models.
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|opus|sonnet|haiku)(?=$|[-_/.:]))/i;
const MOONSHOT_MODEL_PATTERN = /^(moonshotai\/|moonshot\/|kimi[-/])/i;
const FORCED_THINKING_MODEL_PATTERN = /(?:^|[-/])thinking(?:[-/]|$)/i;

export function isMoonshotModel(modelId: string): boolean {
  return MOONSHOT_MODEL_PATTERN.test(modelId);
}

export function shouldSuppressReasoningContent(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function isAnthropicModel(modelId: string): boolean {
  return ANTHROPIC_MODEL_PATTERN.test(modelId);
}

// LiteLLM's /model/info frequently reports `supports_vision: null` and
// `supports_reasoning: null` for proxied Claude models (the proxy simply
// doesn't populate them). Taken literally that marks the model text-only,
// and pi-ai's transform-messages then strips every image block, replacing it
// with "(image omitted: model does not support images)" -- so the UI renders
// the pasted image but the model never sees it. All modern Claude models are
// vision- and reasoning-capable, so when the proxy is silent we infer support
// from the model id rather than defaulting to text-only.
export function inferImageInput(
  modelId: string,
  reported: boolean | null | undefined,
): ("text" | "image")[] {
  if (reported === true || (reported == null && isAnthropicModel(modelId))) {
    return ["text", "image"];
  }
  return ["text"];
}

export function inferReasoning(
  modelId: string,
  reported: boolean | null | undefined,
): boolean {
  if (reported === true) return true;
  if (reported == null && isAnthropicModel(modelId)) return true;
  return false;
}

// Newer Claude models (Opus 4.6/4.7/4.8, Sonnet 4.6) only accept the
// `thinking: { type: "adaptive" }` + `output_config.effort` shape; Opus 4.7/4.8
// outright REJECT the legacy `thinking: { type: "enabled", budget_tokens }`
// format with a 400. pi-ai (>=0.76) keys this decision off
// `compat.forceAdaptiveThinking`, so we set it for the adaptive-capable family.
// Haiku 4.5 is deliberately excluded: it only supports the legacy `enabled`
// format and REJECTS adaptive, so it must keep the default budget-based path.
const ADAPTIVE_THINKING_MODEL_PATTERN =
  /(?:^|[-_/.:])(?:opus[-_.]?4[-_.](?:6|7|8)|sonnet[-_.]?4[-_.]6)(?=$|[-_/.:])/i;

export function isAdaptiveThinkingModel(modelId: string): boolean {
  return ADAPTIVE_THINKING_MODEL_PATTERN.test(modelId);
}

// LiteLLM exposes an Anthropic-native passthrough at `<base>/anthropic` that
// implements `/v1/messages`. Routing Claude models there lets pi speak the
// native Anthropic API (image source blocks with explicit media_type) instead
// of OpenAI chat-completions `image_url` data URIs, which LiteLLM must then
// re-translate to Anthropic format -- a lossy hop that degrades image input.
// Set LITELLM_ANTHROPIC_NATIVE=0 to disable and fall back to openai-completions.
const ENV_ANTHROPIC_NATIVE = "LITELLM_ANTHROPIC_NATIVE";

export function anthropicNativeEnabled(): boolean {
  return process.env[ENV_ANTHROPIC_NATIVE] !== "0";
}

export function anthropicBaseUrl(normalizedBase: string): string {
  return `${normalizedBase}/anthropic`;
}

// Apply native-Anthropic routing overrides in place for Claude-backed models.
function applyAnthropicRouting(models: ProviderModelConfig[], normalizedBase: string): void {
  if (!anthropicNativeEnabled()) return;
  const anthropicBase = anthropicBaseUrl(normalizedBase);
  for (const model of models) {
    if (isAnthropicModel(model.id)) {
      model.api = "anthropic-messages";
      model.baseUrl = anthropicBase;
    }
  }
}

export function buildCompat(modelId: string): ProviderModelConfig["compat"] {
  if (isMoonshotModel(modelId)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return {
      supportsStore: false,
      cacheControlFormat: "anthropic",
      ...(isAdaptiveThinkingModel(modelId) ? { forceAdaptiveThinking: true } : {}),
    };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): KnownProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as KnownProvider) : undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  const prefixProvider = toKnownProvider(id.split("/")[0]);
  const candidates = [toKnownProvider(ownedBy), prefixProvider].filter(
    (provider): provider is KnownProvider => provider !== undefined,
  );

  for (const provider of candidates) {
    const exact = getModels(provider).find((model) => model.id === id);
    if (exact) return exact;
    const providerQualified = getModels(provider).find((model) => model.id === `${provider}/${id}`);
    if (providerQualified) return providerQualified;
  }

  for (const provider of getProviders()) {
    const exact = getModels(provider).find((model) => model.id === id);
    if (exact) return exact;
  }

  return undefined;
}

function getFallbackProviderAndModel(id: string, ownedBy?: string): { provider?: string; modelId: string } {
  const [prefix, ...rest] = id.split("/");
  const prefixProvider = toKnownProvider(prefix);
  if (prefixProvider && rest.length > 0) {
    return { provider: prefixProvider, modelId: rest.join("/") };
  }
  return { provider: toKnownProvider(ownedBy), modelId: id };
}

function findModelsDevModel(
  catalog: ModelsDevResponse | undefined,
  id: string,
  ownedBy?: string,
): ModelsDevModel | undefined {
  const { provider, modelId } = getFallbackProviderAndModel(id, ownedBy);
  if (!provider) return undefined;
  return catalog?.[provider]?.models?.[modelId];
}

function withTimeout(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

async function fetchJson<T>(
  url: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ ok: true; data: T } | { ok: false; status: number }> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, cancel } = withTimeout(timeoutMs, options.signal);
  try {
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal,
    });
    if (!response.ok) return { ok: false, status: response.status };
    const data = (await response.json()) as T;
    return { ok: true, data };
  } finally {
    cancel();
  }
}

async function fetchPublicJson<T>(url: string, options: DiscoveryOptions): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, cancel } = withTimeout(timeoutMs, options.signal);
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      signal,
    });
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return (await response.json()) as T;
  } finally {
    cancel();
  }
}

async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  if (modelsDevCatalog) return modelsDevCatalog;
  try {
    modelsDevCatalog = await fetchPublicJson<ModelsDevResponse>(MODELS_DEV_URL, options);
    return modelsDevCatalog;
  } catch {
    return undefined;
  }
}

function mapModelsDevMetadata(model: ModelsDevModel | undefined): Partial<ProviderModelConfig> {
  if (!model) return {};
  const metadata: Partial<ProviderModelConfig> = {};
  if (model.name) metadata.name = model.name;
  if (model.reasoning !== undefined) metadata.reasoning = model.reasoning;
  if (model.modalities?.input) {
    metadata.input = model.modalities.input.includes("image") ? ["text", "image"] : ["text"];
  }
  const contextWindow = model.limit?.context ?? model.limit?.input;
  if (contextWindow !== undefined) metadata.contextWindow = contextWindow;
  if (model.limit?.output !== undefined) metadata.maxTokens = model.limit.output;
  if (model.cost) {
    metadata.cost = {
      input: model.cost.input ?? 0,
      output: model.cost.output ?? 0,
      cacheRead: model.cost.cache_read ?? 0,
      cacheWrite: model.cost.cache_write ?? 0,
    };
  }
  return metadata;
}

function mapFromModelInfo(entry: ModelInfoEntry): ProviderModelConfig | undefined {
  const id = entry.model_name;
  if (!id) return undefined;
  const info = entry.model_info ?? {};
  if (info.mode && info.mode !== "chat") return undefined;
  return {
    id,
    name: id,
    reasoning: inferReasoning(id, info.supports_reasoning),
    input: inferImageInput(id, info.supports_vision),
    cost: {
      input: (info.input_cost_per_token ?? 0) * 1_000_000,
      output: (info.output_cost_per_token ?? 0) * 1_000_000,
      cacheRead: (info.cache_read_input_token_cost ?? 0) * 1_000_000,
      cacheWrite: (info.cache_creation_input_token_cost ?? 0) * 1_000_000,
    },
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

function mapFromModelsList(
  entry: ModelsListEntry,
  modelsDev: ModelsDevResponse | undefined,
): ProviderModelConfig | undefined {
  const id = entry.id;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id, entry.owned_by);
  const modelsDevMetadata = mapModelsDevMetadata(findModelsDevModel(modelsDev, id, entry.owned_by));
  return {
    id,
    name: modelsDevMetadata.name ?? catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: modelsDevMetadata.reasoning ?? catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: modelsDevMetadata.input ?? catalogModel?.input ?? ["text"],
    cost: modelsDevMetadata.cost ?? catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: modelsDevMetadata.contextWindow ?? catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: modelsDevMetadata.maxTokens ?? catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const models = (infoResult.data.data ?? [])
      .map(mapFromModelInfo)
      .filter((m): m is ProviderModelConfig => m !== undefined);
    applyAnthropicRouting(models, base);
    return { source: "model_info", models };
  }
  if (![401, 403, 404].includes(infoResult.status)) {
    throw new Error(`/model/info returned ${infoResult.status}`);
  }
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  const modelsDev = await getModelsDevCatalog(options);
  const models = (listResult.data.data ?? [])
    .map((entry) => mapFromModelsList(entry, modelsDev))
    .filter((m): m is ProviderModelConfig => m !== undefined);
  applyAnthropicRouting(models, base);
  return { source: "models_list", models };
}
