const assert = require('node:assert/strict')
const test = require('node:test')
const launcher = require('../pinokio.js')

const info = (installed, running = [], url) => ({
  exists: path => installed && path === 'app/env/.installed',
  running: path => running.includes(path),
  local: () => ({ url }),
})

test('partial installation offers Install; successful installation offers Start', async () => {
  assert.equal((await launcher.menu(null, info(false)))[0].href, 'install.js')
  assert.equal((await launcher.menu(null, info(true)))[0].href, 'start.js')
})

test('maintenance remains visible after reset removes the environment', async () => {
  for (const script of ['update.js', 'reset.js', 'link.js', 'install.js']) {
    assert.equal((await launcher.menu(null, info(false, [script])))[0].href, script)
  }
})

test('server capture opens the actual local URL', async () => {
  const start = require('../start.js')
  const shell = start.run.find(step => step.method === 'shell.run')
  const pattern = shell.params.on[0].event
  const match = 'Running on local URL:  http://127.0.0.1:7891'.match(new RegExp(pattern.slice(1, -1)))
  assert.equal(match[1], 'http://127.0.0.1:7891')
  assert.equal(start.run.find(step => step.method === 'local.set').params.url, '{{input.event[1]}}')
  assert.equal((await launcher.menu(null, info(true, ['start.js'], match[1])))[0].href, match[1])
})

test('each platform selects exactly one torch installation', () => {
  const script = require('../torch.js')
  for (const [platform, arch, gpu, expected] of [
    ['win32', 'x64', 'nvidia', 'cu128'],
    ['linux', 'x64', 'nvidia', 'cu128'],
    ['linux', 'x64', 'amd', 'rocm6.3'],
    ['win32', 'x64', 'amd', '/cpu'],
    ['linux', 'x64', null, '/cpu'],
    ['darwin', 'arm64', 'apple', 'torch==2.7.0'],
    ['darwin', 'x64', null, 'Intel macOS is unsupported'],
  ]) {
    const commands = []
    for (const step of script.run) {
      if (step.when && !Function('platform', 'arch', 'gpu', `return ${step.when.slice(2, -2)}`)(platform, arch, gpu)) continue
      commands.push(step.params.message)
      if (step.next === null) break
    }
    assert.equal(commands.length, 1)
    assert.ok(commands[0].includes(expected))
    assert.ok(!commands[0].includes('--no-deps'))
  }
})

test('installation validates before marking ready and update reinstalls dependencies', () => {
  const steps = require('../install.js').run
  const validation = steps.findIndex(step => step.params.message?.includes('uv pip check'))
  const marker = steps.findIndex(step => step.method === 'fs.write')
  assert.ok(validation >= 0 && marker > validation)
  const update = require('../update.js').run
  assert.equal(update[0].params.message, 'git pull --ff-only')
  assert.equal(update[1].params.uri, 'install.js')
})
