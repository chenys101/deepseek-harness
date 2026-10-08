/**
 * [pieqi-fork] Smoke test that the BUILT artifact is a drop-in replacement.
 *
 * The fork's shipping path is `lib/index.js` copied over the installed
 * `@deepseek-ai/dsh-acp`, so the bundle — not `src/` — is what must work. This
 * spec loads the built entry directly and re-checks the two most load-bearing
 * fork behaviors end to end over a real ACP connection: the catalog is hidden
 * from `configOptions` and relayed through response `_meta`, and a caller-named
 * route is applied.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import {
  client as createAcpClientApp,
  methods,
  ndJsonStream,
  PROTOCOL_VERSION,
  type PromptRequest,
  type PromptResponse,
  type SendRequestOptions,
  type Stream,
} from '@agentclientprotocol/sdk'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { type GenerateOptions, LlmAdapter, type StreamChunk } from '@deepseek-ai/dsh-llm'
import JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import type { AcpConfig } from '../src/index.ts'

/** The built artifact whose sha256 `install.mjs` compares. */
const BUILT_ENTRY = new URL('../lib/index.js', import.meta.url)

/** The built artifact's exports, typed by the source it was compiled from. */
interface BuiltPlugin {
  name: string
  inject: string[]
  Config: unknown
  apply: (ctx: Context, config: AcpConfig) => void
}

/**
 * Load the built artifact.
 *
 * The published entry is JavaScript with no sibling `.d.ts` inside `lib/`
 * (`lib/types/` holds them), so the import is typed by an explicit assertion
 * onto the same surface `src/index.ts` exports. This spec exists to prove the
 * bundle loads and runs, which only the built file can show.
 * @returns the built plugin's exports.
 */
async function loadBuiltPlugin(): Promise<BuiltPlugin> {
  return await import(BUILT_ENTRY.href) as BuiltPlugin
}

/** Minimal scripted adapter; the artifact spec needs routing, not catalog breadth. */
class StubAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  override providerInfo(provider: string) {
    if (provider !== 'mock') throw new Error(`StubAdapter: unknown provider ${provider}`)
    return { id: 'mock', name: 'Mock' }
  }

  override listModels(provider: string) {
    return Promise.resolve(provider === 'mock'
      ? [
        { provider: 'mock', id: 'mock', name: 'Mock', inputModalities: ['text'] as const },
        { provider: 'mock', id: 'plain', name: 'Plain', inputModalities: ['text'] as const },
      ]
      : [])
  }

  override resolveModel(provider: string, model: string) {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ['text'] as const,
      context: { contextWindow: 1_024 },
    })
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'ok' }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: 'ok' } }
    yield { type: 'usage', usage: { inputTokens: 1, outputTokens: 2 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

describe('[pieqi-fork] built lib/index.js artifact', () => {
  let ctx: Context | undefined

  afterEach(async () => {
    await ctx?.fiber.dispose()
    ctx = undefined
  })

  it('exports the function-plugin surface the Loader consumes', async () => {
    const built = await loadBuiltPlugin()
    expect(typeof built.apply).toBe('function')
    expect(built.name).toBe('acp')
    expect(built.inject).toEqual(['agents', 'llm', 'sessionPersistence', 'sessions'])
    expect(built.Config).toBeDefined()
  })

  it('omits any runtime import of the optional default-model package', async () => {
    // B1 reads agentDefaultModel through ctx.get, so the built bundle must not
    // import it: dsh-acp does not declare that dependency, and importing it
    // would make the artifact fail to load in a deployment without it.
    const source = await readFile(BUILT_ENTRY, 'utf8')
    expect(source).not.toContain('agent-default-model')
    expect(source).toContain('agentDefaultModel')
  })

  it('serves a session with hidden configOptions and a _meta catalog from the bundle', async () => {
    const built = new Context()
    ctx = built
    const adapter = new StubAdapter()
    await mountAgentLoopTestDependencies(built, { systemPrompt: { personaPrefix: '' } })
    await built.plugin(JsonlSessionPersistence, {
      root: `${(await import('node:os')).tmpdir()}/dsh-acp-artifact-${createHash('sha1').update(String(Date.now())).digest('hex').slice(0, 8)}`,
      compression: 'none',
    })
    await built.plugin(AgentLoop, { agents: [] })
    built.llm.registerAdapter(['mock'], adapter)

    const agentToClient = new TransformStream<Uint8Array, Uint8Array>()
    const clientToAgent = new TransformStream<Uint8Array, Uint8Array>()
    const writer = clientToAgent.writable.getWriter()
    const agentStream: Stream = ndJsonStream(agentToClient.writable, clientToAgent.readable)
    const clientStream: Stream = ndJsonStream(
      new WritableStream<Uint8Array>({ write: chunk => writer.write(chunk) }),
      agentToClient.readable,
    )

    const config = { stream: agentStream, provider: 'mock', model: 'mock' }
    const plugin = await loadBuiltPlugin()
    await built.plugin({
      name: 'acp-built',
      inject: [...plugin.inject],
      apply: (inner: Context) => { plugin.apply(inner, config) },
    })
    const client = createAcpClientApp({ name: 'artifact-smoke' }).connect(clientStream).agent
    await client.request(methods.agent.initialize, { protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const created = await client.request(methods.agent.session.new, { cwd: process.cwd(), mcpServers: [] })

    // A1: the protocol field stays empty. A3: the real catalog travels in _meta.
    expect(created.configOptions).toEqual([])
    const catalog = created._meta?.['pieqi/configOptions'] as { id: string; currentValue?: string }[] | undefined
    expect(catalog?.find(option => option.id === 'model')).toMatchObject({ currentValue: '["mock","mock"]' })

    // B3: a per-turn route named in _meta is applied to that turn.
    const response: PromptResponse = await client.request(methods.agent.session.prompt, {
      sessionId: created.sessionId,
      prompt: [{ type: 'text', text: 'route me' }],
      _meta: { model: '["mock","plain"]' },
    } as PromptRequest & SendRequestOptions)
    expect(response.stopReason).toBe('end_turn')
    expect(adapter.requests[0]).toMatchObject({ provider: 'mock', model: 'plain' })

    await client.request(methods.agent.session.close, { sessionId: created.sessionId })
  }, 30_000)
})
