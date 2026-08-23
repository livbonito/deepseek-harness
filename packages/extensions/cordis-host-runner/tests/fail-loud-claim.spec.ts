import { describe, expect, it } from 'vitest'
import { AGENT_A, mount, setup } from './helpers.ts'

/**
 * The runner's fail-loud claim, exercised through the real global registry it
 * registers into (the same array app-boot's installFailLoud consults): a
 * rejection whose stack carries `cordis-dyn-<pluginId>.js` frames from a live
 * run is claimed and steered into the owning session; every other rejection —
 * no sandbox frames, unknown plugin, or no live run — stays fatal.
 */

/** The `Symbol.for` key shared with app-boot's `registerFailLoudClaim`. */
const FAIL_LOUD_CLAIMS = Symbol.for('dsh.failLoudClaims')

type Claim = (reason: unknown) => boolean

function failLoudClaims(): Claim[] {
  const holder = globalThis as { [key: symbol]: Claim[] | undefined }
  return (holder[FAIL_LOUD_CLAIMS] ??= [])
}

/**
 * The stack shape a floating sandbox rejection carries — guard host frames
 * above the nearest sandbox frame, exactly as captured from a real
 * `dsh web` fatal load failure.
 */
function sandboxRejection(pluginId: string): Error {
  const error = new Error('harness.defineTool parameters.id.required must be true when present')
  error.stack = [
    'Error: harness.defineTool parameters.id.required must be true when present',
    '    at normalizePropertyMap (…/cordis-host-runner/src/guard.ts:378:13)',
    '    at Object.sandboxDefineTool (…/cordis-host-runner/src/guard.ts:553:22)',
    `    at report (cordis-dyn-${pluginId}.js:11:41)`,
    `    at cordis-dyn-${pluginId}.js:25:18`,
  ].join('\n')
  return error
}

describe('fail-loud claim for sandbox-origin rejections', () => {
  it('claims a live plugin rejection and steers its session, leaving every other rejection fatal', async () => {
    // Isolate this test's registrations: the global registry must contain only
    // the claim the setup below installs, and earlier ones are restored after.
    const previous = failLoudClaims().splice(0, failLoudClaims().length)
    const steered: string[] = []
    const harness = await setup()
    try {
      harness.ctx.provide('agents', {
        get: (id: string) => id === AGENT_A.id
          ? {
            id,
            steer: (message: { content: Array<{ type: string; text?: string }> }) => {
              steered.push(message.content
                .filter(block => block.type === 'text')
                .map(block => block.text ?? '')
                .join(''))
            },
          }
          : undefined,
      })
      const pluginId = await mount(harness, 'return { name: \'idle\', apply() {} }')
      const neverRun = harness.runner.define({
        sessionId: AGENT_A.id,
        plugin: { kind: 'new', idPrefix: 'idle' },
        name: 'never-run',
        purpose: 'spec fixture',
        code: { host: 'return { name: \'never\', apply() {} }' },
      })

      expect(failLoudClaims()).toHaveLength(1)
      const claim = failLoudClaims()[0]!

      // A live plugin's sandbox rejection is claimed and steered into its session.
      expect(claim(sandboxRejection(String(pluginId)))).toBe(true)
      expect(steered).toHaveLength(1)
      expect(steered[0]).toContain('guard rejected runtime code')
      expect(steered[0]).toContain(String(pluginId))

      // No sandbox frames, an unknown plugin, or a plugin with no live run: fatal.
      expect(claim(new Error('ordinary host failure'))).toBe(false)
      expect(claim(sandboxRejection('ghost-9'))).toBe(false)
      expect(claim(sandboxRejection(String(neverRun.pluginId)))).toBe(false)
      expect(steered).toHaveLength(1)
    } finally {
      failLoudClaims().push(...previous)
      await harness.ctx.fiber.dispose()
    }
  })

  it('registers one claim per tree and removes it when the tree disposes', async () => {
    const before = failLoudClaims().length
    const harness = await setup()
    expect(failLoudClaims().length).toBe(before + 1)
    await harness.ctx.fiber.dispose()
    expect(failLoudClaims().length).toBe(before)
  })
})
