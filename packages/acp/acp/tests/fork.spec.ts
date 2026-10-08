/**
 * [pieqi-fork] Behavior of the local ACP changes recorded in `FORK.md`.
 *
 * The six changes split in two groups: A1–A3 hide the model catalog from the
 * protocol's own `configOptions` field and relay it through response `_meta`,
 * and B1–B3 let a caller or the deployment default supply the route.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { PROTOCOL_VERSION, type SessionConfigOption } from '@agentclientprotocol/sdk'
import AgentDefaultModel from '@deepseek-ai/dsh-agent-default-model'
import { textResponse, type BridgeHarness } from './harness.ts'
import { makeBridgeHarness } from './harness.ts'

/** Read the forked fork's catalog relay without depending on its literal key twice. */
const CONFIG_OPTIONS_META_KEY = 'pieqi/configOptions'

/** Extract the relayed catalog from a session/new or session/resume response. */
function relayedCatalog(response: { _meta?: Record<string, unknown> | null }): SessionConfigOption[] | undefined {
  return response._meta?.[CONFIG_OPTIONS_META_KEY] as SessionConfigOption[] | undefined
}

/** Locate one option in a catalog by its config id. */
function optionById(options: readonly SessionConfigOption[] | undefined, id: string): SessionConfigOption | undefined {
  return options?.find(option => option.id === id)
}

/** Read the opaque selector value of an advertised model choice by its display name. */
function modelValueNamed(options: readonly SessionConfigOption[] | undefined, name: string): string {
  const model = optionById(options, 'model')
  if (model?.type !== 'select') throw new Error('expected a model select option')
  const choice = model.options.flatMap(option => 'group' in option ? option.options : [option])
    .find(option => option.name === name)
  if (choice === undefined) throw new Error(`expected a model choice named ${name}`)
  return choice.value
}

describe('[pieqi-fork] hidden ACP config', () => {
  let harness: BridgeHarness | undefined

  afterEach(async () => {
    await harness?.dispose()
    harness = undefined
  })

  it('keeps configOptions empty and relays the real catalog through response _meta', async () => {
    harness = await makeBridgeHarness()
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    expect(created.configOptions).toEqual([])
    const catalog = relayedCatalog(created)
    const model = optionById(catalog, 'model')
    expect(model).toMatchObject({ currentValue: '["mock","mock"]' })
    if (model?.type !== 'select') throw new Error('expected a model select option')
    const groups = model.options.filter(option => 'group' in option)
    expect(groups.map(group => group.group)).toEqual(['mock'])
  })

  it('relays the catalog on session/resume without advertising it on the protocol field', async () => {
    harness = await makeBridgeHarness()
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.closeSession({ sessionId: created.sessionId })

    const resumed = await harness.client.resumeSession({ sessionId: created.sessionId, cwd: process.cwd() })

    expect(resumed.configOptions).toEqual([])
    expect(optionById(relayedCatalog(resumed), 'model')).toMatchObject({ currentValue: '["mock","mock"]' })
  })

  it('omits the relay key when the catalog is empty', async () => {
    harness = await makeBridgeHarness({ config: { provider: undefined, model: undefined } })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    expect(created.configOptions).toEqual([])
    expect(created._meta?.[CONFIG_OPTIONS_META_KEY]).toBeUndefined()
  })

  it('still rejects a stale pinned route at session creation', async () => {
    harness = await makeBridgeHarness()
    vi.spyOn(harness.ctx.llm, 'resolveCallConfig').mockRejectedValue(new Error('route is gone'))
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    // A1 keeps routing validation inside the hidden call, so a stale pinned
    // model still fails session/new instead of surfacing at the first prompt.
    await expect(harness.client.newSession({ cwd: process.cwd(), mcpServers: [] }))
      .rejects.toThrow(/Internal error/)
  })

  it('does not push config_option_update when adapter topology changes', async () => {
    harness = await makeBridgeHarness()
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    harness.registerCatalogProvider('other')
    // The catalog read happens either way; only its publication is suppressed.
    await vi.waitFor(() => { expect(harness!.ctx.llm.listProviders().map(item => item.id)).toContain('other') })
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(harness.updates.filter(update => update.sessionUpdate === 'config_option_update')).toEqual([])
  })

  it('keeps a session usable through standard config options while hiding the catalog', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('plain answer')] })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    const plain = modelValueNamed(relayedCatalog(created), 'Mock Plain')

    // The fork hides configOptions on session/new and session/resume only;
    // set_config_option keeps returning the real state as upstream does.
    const selected = await harness.client.setSessionConfigOption({
      sessionId: created.sessionId,
      configId: 'model',
      value: plain,
    })
    expect(optionById(selected.configOptions, 'model')).toMatchObject({ currentValue: '["mock","plain"]' })

    await harness.client.prompt({ sessionId: created.sessionId, prompt: [{ type: 'text', text: 'use plain' }] })
    expect(harness.adapter.requests[0]).toMatchObject({ provider: 'mock', model: 'plain' })
  })
})

describe('[pieqi-fork] externally supplied model route', () => {
  let harness: BridgeHarness | undefined

  afterEach(async () => {
    await harness?.dispose()
    harness = undefined
  })

  it('falls back to the deployment default model when the static route is empty', async () => {
    harness = await makeBridgeHarness({
      config: { provider: undefined, model: undefined },
      script: [textResponse('default-routed')],
    })
    await harness.ctx.plugin(AgentDefaultModel, { provider: 'mock', model: 'plain' })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    expect(optionById(relayedCatalog(created), 'model')).toMatchObject({ currentValue: '["mock","plain"]' })
    await harness.client.prompt({ sessionId: created.sessionId, prompt: [{ type: 'text', text: 'go' }] })
    expect(harness.adapter.requests[0]).toMatchObject({ provider: 'mock', model: 'plain' })
  })

  it('still leaves the route unset when no deployment default is mounted', async () => {
    harness = await makeBridgeHarness({ config: { provider: undefined, model: undefined } })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    expect(harness.ctx.agents.get(sessionId as never)?.options).toEqual({})
  })

  it('accepts both _meta encodings when creating a session', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('one'), textResponse('two')] })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const opaque = await harness.client.newSession({
      cwd: process.cwd(),
      mcpServers: [],
      _meta: { model: '["mock","plain"]' },
    })
    expect(optionById(relayedCatalog(opaque), 'model')).toMatchObject({ currentValue: '["mock","plain"]' })

    const split = await harness.client.newSession({
      cwd: process.cwd(),
      mcpServers: [],
      _meta: { provider: 'mock', model: 'plain' },
    })
    expect(optionById(relayedCatalog(split), 'model')).toMatchObject({ currentValue: '["mock","plain"]' })
  })

  it('ignores unrecognized _meta instead of failing the request', async () => {
    harness = await makeBridgeHarness()
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })

    const created = await harness.client.newSession({
      cwd: process.cwd(),
      mcpServers: [],
      _meta: { model: 'not-a-selector', unrelated: { nested: true } },
    })

    expect(optionById(relayedCatalog(created), 'model')).toMatchObject({ currentValue: '["mock","mock"]' })
  })

  it('applies a per-turn _meta model to exactly that turn', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('first'), textResponse('second')] })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    await harness.client.prompt({
      sessionId: created.sessionId,
      prompt: [{ type: 'text', text: 'first' }],
      _meta: { model: '["mock","plain"]' },
    })
    await harness.client.prompt({ sessionId: created.sessionId, prompt: [{ type: 'text', text: 'second' }] })

    expect(harness.adapter.requests.map(request => request.model)).toEqual(['plain', 'plain'])
  })

  it('rejects a per-turn _meta model that is not in the advertised catalog', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('never')] })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })

    await expect(harness.client.prompt({
      sessionId: created.sessionId,
      prompt: [{ type: 'text', text: 'go' }],
      _meta: { model: '["mock","missing"]' },
    })).rejects.toThrow(/unknown model option/)
  })

  it('routes a resumed session from _meta when the log recorded no route', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('resumed')] })
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const created = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.closeSession({ sessionId: created.sessionId })

    const resumed = await harness.client.resumeSession({
      sessionId: created.sessionId,
      cwd: process.cwd(),
      _meta: { model: '["mock","plain"]' },
    })

    expect(optionById(relayedCatalog(resumed), 'model')).toMatchObject({ currentValue: '["mock","plain"]' })
  })
})
