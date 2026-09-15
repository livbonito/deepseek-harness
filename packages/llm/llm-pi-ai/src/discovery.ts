/**
 * Answering "which models can this provider serve?" for the configuration
 * surface's "fetch available models" action.
 *
 * A route the installed pi-ai catalog ships is answered **from that catalog**,
 * with no network call at all: pi-ai's registry is the authoritative list for
 * its own providers, and it carries the capacities a listing endpoint would
 * not disclose. Only a route the catalog does not describe — a gateway, a
 * self-hosted server — is interrogated over the wire.
 *
 * Neither path is a catalog refresh. Nothing here is stored: the request
 * carries a draft the user is still editing, and the reply is candidate
 * metadata the surface offers for adoption. `settings.yaml` remains the only
 * thing that decides what a route serves.
 *
 * Only OpenAI-compatible protocols are interrogated. Their listing is the one
 * shape a gateway, a self-hosted server, and the official endpoints all agree
 * on, which is the case this action exists for; every other protocol reports
 * that it cannot be interrogated so the surface falls back to hand-entry
 * rather than guessing a response shape.
 *
 * One gap that listing shape cannot close: a self-hosted server names its
 * models but discloses no capacity for them, so a surface adopting the rows
 * guesses a context window — and a local server serves the context it loaded,
 * which is usually a fraction of what the model supports. When the OpenAI
 * listing adopted no capacity at all, a best-effort second pass asks the same
 * endpoint's own native listing (LM Studio serves `/api/v0/models` beside its
 * `/v1` one) for the truth. That pass never fails a reply that already
 * succeeded: an endpoint without the native listing, or one whose reply is
 * not the expected shape, keeps the OpenAI listing's rows unchanged.
 *
 * @module dsh-llm-pi-ai/discovery
 */

import { INVALID_CREDENTIAL_CODE, LlmError, normalizeApiKey } from '@deepseek-ai/dsh-llm'
import type { LlmDiscoveredModel, LlmModelDiscoveryRequest } from '@deepseek-ai/dsh-llm'
import { attributionHeaders } from '@deepseek-ai/dsh-llm'
import { catalogModels } from './catalog.ts'

/**
 * Protocols whose model listing this module can read: the two that speak
 * OpenAI's `GET /models` shape with bearer auth. Azure is absent despite its
 * OpenAI lineage — it authenticates with an `api-key` header and requires an
 * `api-version` query — and Codex authenticates through OAuth; guessing at
 * either would report an authentication failure as a provider with no models.
 * pi-ai's remaining protocols are absent for the same reason.
 */
const LISTABLE_PROTOCOLS: ReadonlySet<string> = new Set([
  'openai-completions',
  'openai-responses',
])

/**
 * Endpoint replies larger than this are refused. The endpoint is whatever URL
 * the user typed, so the ceiling holds on the bytes actually read rather than
 * on the length the server claims — the same two-stage shape `dsh-web-fetch`
 * uses for its own caller-supplied URLs, except that a truncated model listing
 * is not parseable, so overflow rejects instead of truncating.
 */
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024

/** One entry of an OpenAI-compatible `GET /models` reply. */
interface ListingEntry {
  id?: unknown
  /** Common gateway extensions; absent from the official listings. */
  name?: unknown
  display_name?: unknown
  context_window?: unknown
  context_length?: unknown
  max_tokens?: unknown
  max_output_tokens?: unknown
}

/**
 * One entry of a native `GET /api/v0/models` reply, the listing LM Studio
 * serves beside its OpenAI-compatible one. Only the fields this module reads
 * are named; the shape is validated by content, so another server that
 * happens to answer the same path is read only through fields it genuinely
 * carries.
 */
interface NativeEntry {
  id?: unknown
  /** LM Studio's kind for the row: `llm`, `vlm`, or `embeddings`. */
  type?: unknown
  /** Context the model supports at most. */
  max_context_length?: unknown
  /** Context the server actually loaded, when the model is loaded. */
  loaded_context_length?: unknown
}

/** A positive integer field of a listing entry, or `undefined` when absent or unusable. */
function capacity(...candidates: readonly unknown[]): number | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate > 0) return candidate
  }
  return undefined
}

/** A non-empty string field of a listing entry, or `undefined`. */
function label(...candidates: readonly unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) return candidate
  }
  return undefined
}

/**
 * Join the endpoint base with the listing path. The base is treated as a
 * prefix rather than a URL to resolve against, so a deployment path such as
 * `https://gateway.example/openai/v1` keeps its segments instead of losing
 * them to `URL` resolution.
 */
function listingUrl(baseURL: string): string {
  return `${baseURL.replace(/\/+$/, '')}/models`
}

/**
 * Read a reply body, refusing one that outgrows the ceiling. A declared length
 * is checked first so an honest server is turned away without transferring
 * anything; the accumulated total is what actually enforces the bound, because
 * a server that under-declares (or streams) tells us nothing up front.
 */
async function readBounded(response: Response, url: string): Promise<string> {
  const oversized = (): LlmError =>
    new LlmError(`${url} answered with more than ${MAX_RESPONSE_BYTES} bytes`, 'DISCOVERY_FAILED')
  const declared = Number(response.headers.get('content-length') ?? Number.NaN)
  if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw oversized()
  }
  /* v8 ignore next -- fetch always exposes a body stream on a 2xx Response; the null guard is defensive. */
  if (response.body === null) return ''
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_RESPONSE_BYTES) throw oversized()
      chunks.push(value)
    }
  } finally {
    /* v8 ignore next 4 -- cancel() after a completed or abandoned read settles without rejecting; unobserved best-effort cleanup. */
    await reader.cancel().catch(() => {
      // Cancel after a drained read, or after this function walked away from
      // an oversized one, is cleanup; the reply is already decided either way.
    })
  }
  const body = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder().decode(body)
}

/**
 * Read one OpenAI-compatible listing reply. Entries without a usable id are
 * skipped rather than failing the whole interrogation: a single malformed row
 * should not deny the user the rest of a working endpoint's catalog.
 * @param body - the parsed reply body.
 * @returns the adopted rows, and whether any of them disclosed a context
 * window — the fact that decides whether the native-listing pass is worth a
 * second request.
 */
function readListing(body: unknown): { models: LlmDiscoveredModel[]; disclosedContext: boolean } {
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) {
    throw new LlmError(
      'the endpoint\'s model listing has no "data" array; enter this provider\'s models by hand',
      'DISCOVERY_FAILED',
    )
  }
  const models: LlmDiscoveredModel[] = []
  let disclosedContext = false
  for (const raw of data) {
    const entry = raw as ListingEntry | null
    const id = label(entry?.id)
    if (id === undefined) continue
    const name = label(entry?.name, entry?.display_name)
    const contextWindow = capacity(entry?.context_window, entry?.context_length)
    const maxTokens = capacity(entry?.max_output_tokens, entry?.max_tokens)
    disclosedContext ||= contextWindow !== undefined
    models.push({
      id,
      ...name === undefined ? {} : { name },
      ...contextWindow === undefined ? {} : { contextWindow },
      ...maxTokens === undefined ? {} : { maxTokens },
    })
  }
  return { models, disclosedContext }
}

/**
 * Accept one probe key, or refuse it before the header is built. Without this
 * the `fetch` below would throw a ByteString `TypeError` that this function's
 * catch reports as `could not reach <url>` — blaming the network for a local,
 * deterministic fault.
 * @param raw - the key typed into the form or read from storage.
 * @returns the trimmed, usable key.
 */
function usableProbeKey(raw: string): string {
  const checked = normalizeApiKey(raw)
  if (checked.ok) return checked.value
  throw new LlmError(
    checked.reason === 'empty'
      ? 'this provider\'s API key is blank; enter it on the Models page, or clear it to probe unauthenticated'
      : 'this provider\'s API key contains characters no HTTP header can carry; paste the raw key only',
    INVALID_CREDENTIAL_CODE,
  )
}

/**
 * Interrogate one draft provider endpoint for the models it advertises.
 * @param request - the endpoint, protocol, and one-shot credential to use.
 * @param storedApiKey - the credential the named route already stored, asked
 *   for only when the draft carries none and only on the path that reaches the
 *   network. A configuration surface never holds a stored secret — it edits a
 *   redacted descriptor — so without this an already-configured route would be
 *   interrogated unauthenticated and answer 401.
 * @returns the advertised models in endpoint order.
 * @throws LlmError when the protocol has no readable listing, the endpoint
 *   refuses or fails the request, or the reply is not a model listing.
 */
export async function discoverModels(
  request: LlmModelDiscoveryRequest,
  storedApiKey?: () => Promise<string | undefined>,
): Promise<readonly LlmDiscoveredModel[]> {
  // A catalog route already has its answer, and a better one: the installed
  // entries carry context windows and output caps no listing endpoint reports.
  if (request.provider !== undefined) {
    const installed = catalogModels(request.provider)
    if (installed.size > 0) {
      return [...installed.values()].map(model => ({
        id: model.id,
        name: model.name,
        contextWindow: model.contextWindow,
        maxTokens: model.maxTokens,
      }))
    }
  }
  if (request.baseURL === undefined || request.baseURL.length === 0) {
    throw new LlmError(
      `pi-ai ships no catalog for provider "${request.provider ?? ''}", so its models can only come from its`
      + " endpoint; set a baseURL, or enter this provider's models by hand",
      'DISCOVERY_FAILED',
    )
  }
  // A draft that has not chosen a protocol yet is asked as OpenAI Chat
  // Completions: it is the shape a gateway is overwhelmingly likely to speak,
  // and the alternative — refusing until the field is filled — would withhold
  // the action from the case it exists for. The cost is a misdirected message
  // when the endpoint speaks something else (an Anthropic gateway answers 401,
  // which reads as a credential problem), and hand-entry remains the way out.
  const api = request.api ?? 'openai-completions'
  if (!LISTABLE_PROTOCOLS.has(api)) {
    throw new LlmError(
      `pi-ai protocol "${api}" has no model listing this build can read; enter this provider's models by hand`,
      'DISCOVERY_UNSUPPORTED',
    )
  }
  const url = listingUrl(request.baseURL)
  // A key typed into the form wins: it is the one the user is testing, and it
  // may be the replacement for exactly the stored key that is failing. The
  // stored one is only asked for here, past the catalog short-circuit and the
  // protocol check, so a route answered from the registry costs no credential
  // lookup — and no diagnostic about a credential it never needed.
  // A probe carrying no key stays unauthenticated, which is how a route that
  // relies on the provider's own ambient discovery is meant to be asked.
  const supplied = request.apiKey ?? await storedApiKey?.()
  const apiKey = supplied === undefined ? undefined : usableProbeKey(supplied)
  let response: Response
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        ...apiKey === undefined ? {} : { authorization: `Bearer ${apiKey}` },
        ...attributionHeaders(),
      },
      ...request.signal === undefined ? {} : { signal: request.signal },
    })
  } catch (error: unknown) {
    if (request.signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError(`could not reach ${url}`, 'DISCOVERY_FAILED', { cause: error })
  }
  if (!response.ok) {
    throw new LlmError(
      `${url} answered ${response.status}${response.status === 401 || response.status === 403 ? '; check the API key' : ''}`,
      'DISCOVERY_FAILED',
    )
  }
  let text: string
  try {
    text = await readBounded(response, url)
  } catch (error: unknown) {
    // Cancellation during the body read rejects with the abort reason, which
    // may be any value; the caller gets the same coded failure it would have
    // for a cancellation before the request went out.
    if (request.signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw error
  }
  let body: unknown
  try {
    body = JSON.parse(text)
  } catch (error: unknown) {
    throw new LlmError(`${url} did not answer with JSON`, 'DISCOVERY_FAILED', { cause: error })
  }
  const { models, disclosedContext } = readListing(body)
  if (!disclosedContext) return enrichFromNativeListing(request, models, apiKey)
  return models
}

/**
 * Where the same server's native listing lives, when the OpenAI base points
 * at one. LM Studio serves `/api/v0/models` at its root while its
 * OpenAI-compatible base is `/v1`, so the one trailing `/v1` segment is
 * dropped from the configured base; every other base keeps its segments, which
 * is also what a proxied deployment wants — its root is wherever its own
 * `/v1` hangs.
 * @param baseURL - the configured OpenAI-compatible base.
 * @returns the native listing URL beside it.
 */
function nativeListingUrl(baseURL: string): string {
  const trimmed = baseURL.replace(/\/+$/, '')
  return `${/\/v1$/.test(trimmed) ? trimmed.slice(0, -3) : trimmed}/api/v0/models`
}

/**
 * The context one adopted row can actually use, from the native listing's row
 * for it. What the server loaded wins over what the model supports at most,
 * because the loaded figure is the ceiling a request can reach today — a
 * model that supports 262k served at 119k must be configured as 119k or its
 * sessions overflow the load, not the model.
 * @param entry - the native row, when the reply carried one for the id.
 * @returns the usable context, or `undefined` when the row states none.
 */
function nativeContext(entry: NativeEntry | undefined): number | undefined {
  if (entry === undefined) return undefined
  // An embeddings row is not a chat model: adopting its tiny context would
  // hand a conversation model the embedding model's limit.
  if (entry.type !== 'llm' && entry.type !== 'vlm') return undefined
  return capacity(entry.loaded_context_length, entry.max_context_length)
}

/**
 * Fill in the context windows a bare OpenAI listing left out, from the same
 * endpoint's native listing. Best-effort by construction: every way this
 * second request can fail — the path is absent, the reply is not the expected
 * shape, the caller aborted mid-read — keeps the already-successful listing's
 * rows exactly as they were, because enrichment may only improve a reply that
 * already stands on its own.
 * @param request - the interrogation this pass serves, for its signal.
 * @param models - the rows the OpenAI listing adopted.
 * @param apiKey - the credential the first request used, sent again so a
 * server that demanded it for `/v1/models` is not asked unauthenticated now.
 * @returns the rows, with `contextWindow` filled where the native listing
 * disclosed one for that id.
 */
async function enrichFromNativeListing(
  request: LlmModelDiscoveryRequest,
  models: readonly LlmDiscoveredModel[],
  apiKey: string | undefined,
): Promise<readonly LlmDiscoveredModel[]> {
  // An empty listing leaves nothing to enrich, and a second request that
  // cannot change the reply is a second request the endpoint never owed.
  if (models.length === 0) return models
  const nativeUrl = nativeListingUrl(request.baseURL ?? '')
  try {
    const response = await fetch(nativeUrl, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        ...apiKey === undefined ? {} : { authorization: `Bearer ${apiKey}` },
        ...attributionHeaders(),
      },
      ...request.signal === undefined ? {} : { signal: request.signal },
    })
    if (!response.ok) return models
    const body: unknown = JSON.parse(await readBounded(response, nativeUrl))
    const data = (body as { data?: unknown } | null)?.data
    if (!Array.isArray(data)) return models
    const contexts = new Map<string, number>()
    for (const raw of data) {
      const entry = raw as NativeEntry | null
      const id = label(entry?.id)
      const context = nativeContext(entry ?? undefined)
      if (id === undefined || context === undefined) continue
      contexts.set(id, context)
    }
    if (contexts.size === 0) return models
    return models.map((model) => {
      const context = contexts.get(model.id)
      return context === undefined || model.contextWindow !== undefined ? model : { ...model, contextWindow: context }
    })
  } catch {
    // Anything the enrichment path throws — unreachable path, foreign reply,
    // a caller that aborted after the listing succeeded — is not a fault in
    // the reply being enriched.
    return models
  }
}
