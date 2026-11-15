// FreeModelAggregator — the best free AI model aggregator.
//
// One server, stdlib only. It reads the trial brief (providers, sealed quality
// table, workload manifest), builds its catalog from the providers' own
// catalog routes, registers with the trial env, and serves the OpenAI Chat
// Completions API and the Anthropic Messages API (JSON and SSE) on one port.
// Every request is answered by the highest-rated free model that meets the
// request's requirements, on any provider and any wire, with failover when a
// provider refuses and a reservation rule that keeps the providers a later
// constrained request will need.
//
//   node main.ts --list     print the catalog as one JSON document, exit 0
//   node main.ts            register with $MAKEFUN_ENV_URL and serve the trial
//
// Environment: MAKEFUN_PROMPT_FILE (the brief), MAKEFUN_ENV_URL (the env),
// MAKEFUN_TRIAL_NOW (the trial's fixed now, ISO date or epoch seconds),
// MAKEFUN_TRIAL_SEED (unused: no randomness anywhere in this program).

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

const PROTOCOL_VERSION = 1;
const FIXED_CREATED = 1767225600;
const PROVIDER_TIMEOUT_MS = 5000;
const POLL_MS = 500;
const WORK_POLL_CAP_MS = 175000;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface BriefProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface ManifestRequest {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: {
    tools?: boolean;
    vision?: boolean;
    json?: boolean;
    min_context?: number;
  };
}

interface Brief {
  providers: BriefProvider[];
  ratings: RatingRow[];
  manifest: ManifestRequest[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

interface ImagePart {
  kind: "data" | "url";
  mime: string;
  data: string;
  url: string;
}

interface NormMessage {
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  images: ImagePart[];
  asArray: boolean;
  toolCallId: string | null;
}

interface NormTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  maxTokens: number | null;
  temperature: number | undefined;
  stops: string[] | null;
  jsonMode: boolean;
  stream: boolean;
}

interface ProviderAnswer {
  ok: boolean;
  status: number;
  code: string;
  text: string;
  usage: { input: number; output: number };
}

interface Candidate {
  provider: BriefProvider;
  model: string;
  rating: number;
  reserved: boolean;
}

function log(...args: unknown[]): void {
  process.stderr.write(`free-model-aggregator: ${args.map(String).join(" ")}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function trialNowSec(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (raw === undefined || raw === "") return FIXED_CREATED;
  const asNum = Number(raw);
  if (Number.isFinite(asNum) && asNum > 0) return Math.floor(asNum);
  const parsed = Date.parse(raw);
  if (Number.isFinite(parsed)) return Math.floor(parsed / 1000);
  return FIXED_CREATED;
}

// ---------------------------------------------------------------------------
// Brief parsing: prompt.txt is prose with three single-line JSON documents.
// ---------------------------------------------------------------------------

function parseBrief(text: string): Brief | null {
  let providers: BriefProvider[] | null = null;
  let ratings: RatingRow[] | null = null;
  let manifest: ManifestRequest[] | null = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let doc: unknown;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (doc === null || typeof doc !== "object" || Array.isArray(doc)) continue;
    const schema = (doc as { schema?: unknown }).schema;
    if (schema === "free-model-aggregator-01/providers/v1") {
      providers = (doc as { providers: BriefProvider[] }).providers;
    } else if (schema === "free-model-aggregator-01/ratings/v1") {
      ratings = (doc as { ratings: RatingRow[] }).ratings;
    } else if (schema === "free-model-aggregator-01/workload/v1") {
      manifest = (doc as { manifest: ManifestRequest[] }).manifest;
    }
  }
  if (providers === null || ratings === null || manifest === null) return null;
  return { providers, ratings, manifest };
}

function loadBrief(): Brief | null {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === "") return null;
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Catalog: each provider's own catalog route, with the brief as fallback.
// ---------------------------------------------------------------------------

function authHeaders(p: BriefProvider): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

async function fetchJson(
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<unknown | null> {
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (res.status < 200 || res.status >= 300) return null;
    return (await res.json()) as unknown;
  } catch {
    return null;
  }
}

async function buildCatalog(brief: Brief, envUrl: string): Promise<Catalog> {
  const out: Catalog = { providers: [] };
  for (const p of brief.providers) {
    let models: CatalogModel[] | null = null;
    if (envUrl !== "") {
      const route = p.wire === "gemini" ? "/v1beta/models" : "/v1/models";
      for (let attempt = 0; attempt < 2 && models === null; attempt++) {
        const data = await fetchJson(`${envUrl}${p.path}${route}`, authHeaders(p), PROVIDER_TIMEOUT_MS);
        if (data !== null && typeof data === "object" && !Array.isArray(data)) {
          const items = (data as { data?: unknown }).data;
          if (Array.isArray(items)) {
            models = [];
            for (const item of items) {
              if (item === null || typeof item !== "object" || Array.isArray(item)) continue;
              const m = item as { id?: unknown; free?: unknown; context_length?: unknown; inputTokenLimit?: unknown };
              if (typeof m.id !== "string" || m.free !== true) continue;
              const ctx =
                typeof m.context_length === "number"
                  ? m.context_length
                  : typeof m.inputTokenLimit === "number"
                    ? m.inputTokenLimit
                    : null;
              models.push({ id: m.id, free: true, context_length: ctx });
            }
          }
        }
        if (models === null && attempt === 0) await sleep(100);
      }
    }
    if (models === null) {
      // Fallback: the brief's sealed quality table is the same truth.
      models = brief.ratings
        .filter((r) => r.provider === p.id && r.free)
        .map((r) => ({ id: r.model, free: true, context_length: r.context_length }));
    }
    out.providers.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Normalizing a client request (either wire) into one internal shape.
// ---------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function imagesOfContent(content: unknown): ImagePart[] {
  const out: ImagePart[] = [];
  if (!Array.isArray(content)) return out;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
      out.push(parseImageUrl(block.image_url.url));
    } else if (block.type === "image" && isRecord(block.source)) {
      const src = block.source;
      if (src.type === "base64" && typeof src.data === "string") {
        out.push({
          kind: "data",
          mime: typeof src.media_type === "string" ? src.media_type : "image/png",
          data: src.data,
          url: "",
        });
      } else if (src.type === "url" && typeof src.url === "string") {
        out.push({ kind: "url", mime: "", data: "", url: src.url });
      }
    }
  }
  return out;
}

function parseImageUrl(url: string): ImagePart {
  const m = /^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/s.exec(url);
  if (m !== null) return { kind: "data", mime: m[1] ?? "image/png", data: m[2] ?? "", url: "" };
  return { kind: "url", mime: "", data: "", url };
}

function normalizeClientRequest(clientApi: "openai" | "anthropic", body: Record<string, unknown>): NormRequest {
  const messages: NormMessage[] = [];
  const tools: NormTool[] = [];
  let systemText = "";
  let maxTokens: number | null = null;
  let temperature: number | undefined = undefined;
  let stops: string[] | null = null;
  let jsonMode = false;

  if (clientApi === "openai") {
    const rawMessages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of rawMessages) {
      if (!isRecord(raw) || typeof raw.role !== "string") continue;
      const role = raw.role;
      if (role === "system") {
        const t = textOfContent(raw.content);
        if (t !== "") systemText = systemText === "" ? t : `${systemText}\n${t}`;
        continue;
      }
      if (role !== "user" && role !== "assistant" && role !== "tool") continue;
      messages.push({
        role,
        text: textOfContent(raw.content),
        images: imagesOfContent(raw.content),
        asArray: Array.isArray(raw.content),
        toolCallId: typeof raw.tool_call_id === "string" ? raw.tool_call_id : null,
      });
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== "string") continue;
        tools.push({
          name: t.function.name,
          description: typeof t.function.description === "string" ? t.function.description : "",
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} },
        });
      }
    }
    if (typeof body.max_tokens === "number") maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === "number") maxTokens = body.max_completion_tokens;
    if (typeof body.temperature === "number") temperature = body.temperature;
    if (typeof body.stop === "string") stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === "string");
    if (isRecord(body.response_format)) {
      jsonMode = body.response_format.type === "json_object" || body.response_format.type === "json_schema";
    }
  } else {
    systemText = textOfContent(body.system);
    const rawMessages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of rawMessages) {
      if (!isRecord(raw) || typeof raw.role !== "string") continue;
      if (raw.role !== "user" && raw.role !== "assistant") continue;
      messages.push({
        role: raw.role,
        text: textOfContent(raw.content),
        images: imagesOfContent(raw.content),
        asArray: Array.isArray(raw.content),
        toolCallId: null,
      });
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || typeof t.name !== "string") continue;
        tools.push({
          name: t.name,
          description: typeof t.description === "string" ? t.description : "",
          parameters: isRecord(t.input_schema) ? t.input_schema : { type: "object", properties: {} },
        });
      }
    }
    if (typeof body.max_tokens === "number") maxTokens = body.max_tokens;
    if (typeof body.temperature === "number") temperature = body.temperature;
    if (Array.isArray(body.stop_sequences)) {
      stops = body.stop_sequences.filter((s): s is string => typeof s === "string");
    }
    jsonMode = false;
  }

  return {
    systemText,
    messages,
    tools,
    maxTokens,
    temperature,
    stops,
    jsonMode,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Rendering the normalized request into a provider's wire.
// ---------------------------------------------------------------------------

function openAIContent(msg: NormMessage): string | Record<string, unknown>[] {
  if (!msg.asArray && msg.images.length === 0) return msg.text;
  const parts: Record<string, unknown>[] = [];
  if (msg.text !== "") parts.push({ type: "text", text: msg.text });
  for (const img of msg.images) {
    if (img.kind === "data") {
      parts.push({ type: "image_url", image_url: { url: `data:${img.mime};base64,${img.data}` } });
    } else {
      parts.push({ type: "image_url", image_url: { url: img.url } });
    }
  }
  if (parts.length === 0) return "";
  return parts;
}

function anthropicContent(msg: NormMessage): string | Record<string, unknown>[] {
  if (!msg.asArray && msg.images.length === 0) return msg.text;
  const parts: Record<string, unknown>[] = [];
  if (msg.text !== "") parts.push({ type: "text", text: msg.text });
  for (const img of msg.images) {
    if (img.kind === "data") {
      parts.push({ type: "image", source: { type: "base64", media_type: img.mime, data: img.data } });
    } else {
      parts.push({ type: "image", source: { type: "url", url: img.url } });
    }
  }
  if (parts.length === 0) return "";
  return parts;
}

function geminiParts(msg: NormMessage): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [];
  if (msg.text !== "") parts.push({ text: msg.text });
  for (const img of msg.images) {
    if (img.kind === "data") {
      parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
    } else {
      parts.push({ fileData: { fileUri: img.url } });
    }
  }
  if (parts.length === 0) parts.push({ text: "" });
  return parts;
}

function renderToWire(wire: Wire, norm: NormRequest, model: string): Record<string, unknown> {
  if (wire === "openai") {
    const messages: Record<string, unknown>[] = [];
    if (norm.systemText !== "") messages.push({ role: "system", content: norm.systemText });
    for (const msg of norm.messages) {
      const entry: Record<string, unknown> = { role: msg.role, content: openAIContent(msg) };
      if (msg.role === "tool" && msg.toolCallId !== null) entry.tool_call_id = msg.toolCallId;
      messages.push(entry);
    }
    const body: Record<string, unknown> = { model, messages, stream: false };
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== undefined) body.temperature = norm.temperature;
    if (norm.tools.length > 0) {
      body.tools = norm.tools.map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
    }
    if (norm.stops !== null) body.stop = norm.stops;
    if (norm.jsonMode) body.response_format = { type: "json_object" };
    return body;
  }
  if (wire === "anthropic") {
    const messages: Record<string, unknown>[] = [];
    for (const msg of norm.messages) {
      if (msg.role !== "user" && msg.role !== "assistant") continue;
      messages.push({ role: msg.role, content: anthropicContent(msg) });
    }
    const body: Record<string, unknown> = { model, messages, stream: false };
    if (norm.systemText !== "") body.system = norm.systemText;
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== undefined) body.temperature = norm.temperature;
    if (norm.tools.length > 0) {
      body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    }
    if (norm.stops !== null) body.stop_sequences = norm.stops;
    return body;
  }
  // gemini
  const contents: Record<string, unknown>[] = [];
  for (const msg of norm.messages) {
    if (msg.role === "system" || msg.role === "tool") continue;
    contents.push({ role: msg.role === "assistant" ? "model" : "user", parts: geminiParts(msg) });
  }
  const generationConfig: Record<string, unknown> = {};
  if (norm.maxTokens !== null) generationConfig.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== undefined) generationConfig.temperature = norm.temperature;
  if (norm.stops !== null) generationConfig.stopSequences = norm.stops;
  if (norm.jsonMode) generationConfig.responseMimeType = "application/json";
  const body: Record<string, unknown> = { model, contents, generationConfig };
  if (norm.systemText !== "") body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  return body;
}

// ---------------------------------------------------------------------------
// Calling one provider and parsing its answer (JSON only; the entry re-emits
// SSE to the client itself, so the provider call never streams).
// ---------------------------------------------------------------------------

function answerTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function parseProviderAnswer(wire: Wire, data: unknown): { text: string; usage: { input: number; output: number } } | null {
  if (!isRecord(data)) return null;
  if (wire === "openai") {
    const choices = Array.isArray(data.choices) ? data.choices : null;
    if (choices === null || choices.length === 0 || !isRecord(choices[0])) return null;
    const choice = choices[0];
    if (!isRecord(choice.message)) return null;
    const usage = isRecord(data.usage) ? data.usage : null;
    if (usage === null) return null;
    const input = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null;
    const output = typeof usage.completion_tokens === "number" ? usage.completion_tokens : null;
    if (input === null || output === null) return null;
    return { text: answerTextOf(choice.message.content), usage: { input, output } };
  }
  if (wire === "anthropic") {
    if (data.type !== "message" || !Array.isArray(data.content)) return null;
    const usage = isRecord(data.usage) ? data.usage : null;
    if (usage === null) return null;
    const input = typeof usage.input_tokens === "number" ? usage.input_tokens : null;
    const output = typeof usage.output_tokens === "number" ? usage.output_tokens : null;
    if (input === null || output === null) return null;
    return { text: answerTextOf(data.content), usage: { input, output } };
  }
  // gemini
  const candidates = Array.isArray(data.candidates) ? data.candidates : null;
  if (candidates === null || candidates.length === 0 || !isRecord(candidates[0])) return null;
  const cand = candidates[0];
  if (!isRecord(cand.content) || !Array.isArray(cand.content.parts)) return null;
  const meta = isRecord(data.usageMetadata) ? data.usageMetadata : null;
  if (meta === null) return null;
  const input = typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : null;
  const output = typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : null;
  if (input === null || output === null) return null;
  const texts: string[] = [];
  for (const part of cand.content.parts) {
    if (isRecord(part) && typeof part.text === "string") texts.push(part.text);
  }
  return { text: texts.join(""), usage: { input, output } };
}

async function callProvider(
  envUrl: string,
  provider: BriefProvider,
  model: string,
  norm: NormRequest,
): Promise<ProviderAnswer> {
  const route =
    provider.wire === "gemini"
      ? `/v1beta/models/${encodeURIComponent(model)}:generateContent`
      : provider.wire === "anthropic"
        ? "/v1/messages"
        : "/v1/chat/completions";
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let res: Response;
  try {
    res = await fetch(`${envUrl}${provider.path}${route}`, {
      method: "POST",
      headers,
      body: JSON.stringify(renderToWire(provider.wire, norm, model)),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: "network", text: "", usage: { input: 0, output: 0 } };
  }
  if (res.status < 200 || res.status >= 300) {
    let code = "";
    try {
      const errBody = (await res.json()) as unknown;
      if (isRecord(errBody) && isRecord(errBody.error)) {
        if (typeof errBody.error.code === "string") code = errBody.error.code;
        else if (typeof errBody.error.type === "string") code = errBody.error.type;
      }
    } catch {
      // ignore
    }
    return { ok: false, status: res.status, code, text: "", usage: { input: 0, output: 0 } };
  }
  let data: unknown;
  try {
    data = (await res.json()) as unknown;
  } catch {
    return { ok: false, status: 500, code: "bad_answer", text: "", usage: { input: 0, output: 0 } };
  }
  const parsed = parseProviderAnswer(provider.wire, data);
  if (parsed === null) return { ok: false, status: 500, code: "bad_answer", text: "", usage: { input: 0, output: 0 } };
  return { ok: true, status: 200, code: "", text: parsed.text, usage: parsed.usage };
}

// ---------------------------------------------------------------------------
// Routing: highest-rated free model that meets the requirements, with a
// reservation that keeps the providers a later constrained request needs.
// ---------------------------------------------------------------------------

function meetsRequirements(
  row: RatingRow,
  req: ManifestRequest["requires"],
  jsonMode: boolean,
  providerWire: Wire,
): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && row.context_length < req.min_context) return false;
  // A JSON-mode OpenAI request must be served by a wire that echoes json_mode;
  // the Anthropic wire always echoes json_mode=false.
  if (jsonMode && providerWire === "anthropic") return false;
  return true;
}

interface ScarceNeed {
  kind: "vision" | "min_context";
  threshold: number;
}

function scarceNeedsOf(req: ManifestRequest["requires"]): ScarceNeed[] {
  const out: ScarceNeed[] = [];
  if (req.vision === true) out.push({ kind: "vision", threshold: 0 });
  if (req.min_context !== undefined) out.push({ kind: "min_context", threshold: req.min_context });
  return out;
}

function providerSatisfies(providerId: string, need: ScarceNeed, brief: Brief): { satisfies: boolean; best: number } {
  let best = -1;
  for (const row of brief.ratings) {
    if (row.provider !== providerId || !row.free) continue;
    const ok = need.kind === "vision" ? row.vision : row.context_length >= need.threshold;
    if (ok && row.rating_milli > best) best = row.rating_milli;
  }
  return { satisfies: best >= 0, best };
}

function topProvidersFor(need: ScarceNeed, brief: Brief, count: number): Set<string> {
  const scored: { id: string; best: number }[] = [];
  for (const p of brief.providers) {
    const s = providerSatisfies(p.id, need, brief);
    if (s.satisfies) scored.push({ id: p.id, best: s.best });
  }
  scored.sort((a, b) => (b.best !== a.best ? b.best - a.best : a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return new Set(scored.slice(0, count).map((s) => s.id));
}

function buildCandidates(
  brief: Brief,
  norm: NormRequest,
  entry: ManifestRequest | null,
  requestIndex: number,
  callsMade: Map<string, number>,
): Candidate[] {
  const req: ManifestRequest["requires"] = entry !== null ? entry.requires : {};
  // The manifest is authoritative, but a request's own JSON mode also binds
  // the wire; derive the tools/vision/json needs from the manifest, and fall
  // back to the body's shape when the manifest entry is missing.
  const currentNeeds = scarceNeedsOf(req);
  const futureNeeds: ScarceNeed[] = [];
  for (let i = requestIndex + 1; i < brief.manifest.length; i++) {
    for (const need of scarceNeedsOf(brief.manifest[i].requires)) {
      const key = `${need.kind}:${need.threshold}`;
      if (!futureNeeds.some((n) => `${n.kind}:${n.threshold}` === key)) futureNeeds.push(need);
    }
  }
  // Providers reserved for a future scarce need the current request does not
  // itself need (top 3 per need). A provider that serves a scarce need the
  // current request DOES have is never reserved for this request.
  const reserved = new Set<string>();
  for (const need of futureNeeds) {
    const neededNow = currentNeeds.some((n) => n.kind === need.kind && n.threshold === need.threshold);
    if (neededNow) continue;
    for (const id of topProvidersFor(need, brief, 3)) reserved.add(id);
  }
  for (const need of currentNeeds) {
    for (const id of topProvidersFor(need, brief, 3)) reserved.delete(id);
  }

  const providerById = new Map(brief.providers.map((p) => [p.id, p]));
  const list: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = providerById.get(row.provider);
    if (provider === undefined) continue;
    if (!meetsRequirements(row, req, norm.jsonMode, provider.wire)) continue;
    list.push({ provider, model: row.model, rating: row.rating_milli, reserved: reserved.has(row.provider) });
  }
  list.sort((a, b) => {
    if (a.reserved !== b.reserved) return a.reserved ? 1 : -1;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const ca = callsMade.get(a.provider.id) ?? 0;
    const cb = callsMade.get(b.provider.id) ?? 0;
    if (ca !== cb) return ca - cb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return list;
}

// ---------------------------------------------------------------------------
// Answer rendering in the client's wire.
// ---------------------------------------------------------------------------

function splitPieces(text: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function sendOpenAIJson(
  res: ServerResponse,
  model: string,
  text: string,
  usage: { input: number; output: number },
  created: number,
): void {
  const body = {
    id: "chatcmpl-sim",
    object: "chat.completion",
    created,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
  const payload = Buffer.from(`${JSON.stringify(body)}\n`);
  res.writeHead(200, { "content-type": "application/json", "content-length": String(payload.length) });
  res.end(payload);
}

function sendOpenAIStream(
  res: ServerResponse,
  model: string,
  text: string,
  usage: { input: number; output: number },
  created: number,
): void {
  const frames: string[] = [];
  const chunk = (delta: Record<string, unknown>, finish: string | null, extra?: Record<string, unknown>): string => {
    const c: Record<string, unknown> = {
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    if (extra !== undefined) Object.assign(c, extra);
    return `data: ${JSON.stringify(c)}\n\n`;
  };
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitPieces(text, 13)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, "stop"));
  // The env's client requires a final usage in every OpenAI stream, whether
  // or not the request asked for stream_options.include_usage.
  frames.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  const payload = Buffer.from(frames.join(""));
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": String(payload.length),
  });
  res.end(payload);
}

function sendAnthropicJson(
  res: ServerResponse,
  model: string,
  text: string,
  usage: { input: number; output: number },
): void {
  const body = {
    id: "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
  const payload = Buffer.from(`${JSON.stringify(body)}\n`);
  res.writeHead(200, { "content-type": "application/json", "content-length": String(payload.length) });
  res.end(payload);
}

function sendAnthropicStream(
  res: ServerResponse,
  model: string,
  text: string,
  usage: { input: number; output: number },
): void {
  const frame = (event: string, data: Record<string, unknown>): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [];
  frames.push(
    frame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    frame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  );
  for (const piece of splitPieces(text, 13)) {
    frames.push(
      frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }),
    );
  }
  frames.push(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  frames.push(frame("message_stop", { type: "message_stop" }));
  const payload = Buffer.from(frames.join(""));
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": String(payload.length),
  });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// HTTP plumbing.
// ---------------------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<Buffer | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(tooLarge ? null : Buffer.concat(chunks)));
    req.on("error", () => resolve(null));
  });
}

function sendJson(res: ServerResponse, status: number, value: Record<string, unknown>): void {
  const payload = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, { "content-type": "application/json", "content-length": String(payload.length) });
  res.end(payload);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes("--list");
  const brief = loadBrief();
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  const created = trialNowSec();

  if (listMode) {
    const catalog: Catalog = brief === null ? { providers: [] } : await buildCatalog(brief, envUrl);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  if (brief === null) {
    log("no brief: set MAKEFUN_PROMPT_FILE");
    process.exitCode = 1;
    return;
  }
  if (envUrl === "") {
    log("no env: set MAKEFUN_ENV_URL");
    process.exitCode = 1;
    return;
  }

  const catalog = await buildCatalog(brief, envUrl);
  log(`catalog: ${catalog.providers.length} providers, ${catalog.providers.reduce((n, p) => n + p.models.length, 0)} free models`);

  const callsMade = new Map<string, number>();
  const exhausted = new Set<string>();
  let requestIndex = 0;

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://entry");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const ids: { id: string; object: string; created: number; owned_by: string; free: boolean }[] = [];
          for (const p of catalog.providers) {
            for (const m of p.models) ids.push({ id: m.id, object: "model", created, owned_by: "free-model-aggregator", free: true });
          }
          sendJson(res, 200, { object: "list", data: ids });
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const clientApi = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
          const idx = requestIndex++;
          const entry = brief.manifest[idx] ?? null;
          const raw = await readBody(req);
          if (raw === null) {
            sendJson(res, 400, { error: { message: "request body too large or unreadable", type: "invalid_request_error" } });
            return;
          }
          let body: Record<string, unknown>;
          try {
            const parsed: unknown = JSON.parse(raw.toString("utf8"));
            if (!isRecord(parsed)) throw new Error("not an object");
            body = parsed;
          } catch {
            sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error" } });
            return;
          }
          const norm = normalizeClientRequest(clientApi, body);
          const candidates = buildCandidates(brief, norm, entry, idx, callsMade);
          const blocked = new Set<string>();
          let served: { text: string; usage: { input: number; output: number }; provider: string; model: string } | null = null;
          for (const cand of candidates) {
            if (exhausted.has(cand.provider.id) || blocked.has(cand.provider.id)) continue;
            callsMade.set(cand.provider.id, (callsMade.get(cand.provider.id) ?? 0) + 1);
            const ans = await callProvider(envUrl, cand.provider, cand.model, norm);
            if (ans.ok) {
              served = { text: ans.text, usage: ans.usage, provider: cand.provider.id, model: cand.model };
              break;
            }
            if (ans.status === 402 || ans.status === 429) exhausted.add(cand.provider.id);
            else blocked.add(cand.provider.id);
            log(`request ${entry?.id ?? idx}: ${cand.provider.id}/${cand.model} failed ${ans.status} ${ans.code}`);
          }
          if (served === null) {
            sendJson(res, 502, { error: { message: "no provider could serve this request", type: "api_error" } });
            return;
          }
          log(`request ${entry?.id ?? idx}: served by ${served.provider}/${served.model}`);
          if (clientApi === "openai") {
            if (norm.stream) sendOpenAIStream(res, served.model, served.text, served.usage, created);
            else sendOpenAIJson(res, served.model, served.text, served.usage, created);
          } else if (norm.stream) {
            sendAnthropicStream(res, served.model, served.text, served.usage);
          } else {
            sendAnthropicJson(res, served.model, served.text, served.usage);
          }
          return;
        }
        sendJson(res, 404, { error: { message: `no route ${req.method ?? ""} ${url.pathname}`, type: "invalid_request_error" } });
      } catch (err) {
        try {
          sendJson(res, 500, { error: { message: err instanceof Error ? err.message : "internal error", type: "api_error" } });
        } catch {
          // ignore
        }
      }
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", () => resolve()));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  log(`listening on 0.0.0.0:${port}`);

  const reg = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  }).catch(() => null);
  if (reg === null || reg.status < 200 || reg.status >= 300) {
    log(`register failed: ${reg === null ? "no response" : `HTTP ${reg.status}`}`);
    process.exitCode = 1;
    server.close();
    return;
  }
  const regBody = (await reg.json()) as { ok?: unknown; requests?: unknown };
  log(`registered: ok=${String(regBody.ok)} requests=${String(regBody.requests)}`);

  // Poll /work until the env says the workload is done. The cap is a poll
  // count, not a clock reading, so the run stays deterministic: nothing
  // that differs between two runs may reach a request or the exit.
  const maxPolls = Math.ceil(WORK_POLL_CAP_MS / POLL_MS);
  for (let poll = 0; poll < maxPolls; poll++) {
    let done = false;
    try {
      const w = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
      if (w.status >= 200 && w.status < 300) {
        const data = (await w.json()) as { done?: unknown };
        done = data.done === true;
      }
    } catch {
      // retry next round
    }
    if (done) break;
    if (poll === maxPolls - 1) {
      log("work poll timed out");
      process.exitCode = 1;
    }
    await sleep(POLL_MS);
  }
  log("work done, exiting 0");
  server.close();
}

main().catch((err) => {
  log(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
