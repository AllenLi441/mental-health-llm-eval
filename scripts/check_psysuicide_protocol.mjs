#!/usr/bin/env node
// Offline deterministic guard for PsySUICIDE split/profile/test protocol.
import { fileURLToPath } from 'node:url';

process.env.EVAL_IGNORE_DOTENV = '1';

const lib = await import('../lib.mjs');
const task = await import('../tasks/psysuicide.mjs');

function args(argv) {
  return lib.parseArgs(argv);
}

function expectThrow(name, fn, pattern) {
  try {
    fn();
  } catch (error) {
    if (!pattern.test(String(error.message || error))) {
      throw new Error(`${name}: wrong error: ${error.message || error}`);
    }
    return;
  }
  throw new Error(`${name}: expected rejection`);
}

lib.selftestTask(task, args([]));

expectThrow(
  'missing split',
  () => lib.taskRunArgs(task, args(['--model', 'deepseek-v4-pro'])),
  /require explicit --split/,
);
expectThrow(
  'train cannot be scored',
  () => lib.taskRunArgs(task, args(['--split', 'train', '--model', 'deepseek-v4-pro'])),
  /not a scored optimization split/,
);
expectThrow(
  'unknown prompt profile',
  () => lib.taskRunArgs(task, args(['--split', 'valid', '--prompt-profile', 'invented'])),
  /unsupported --prompt-profile/,
);
expectThrow(
  'test default run id',
  () => lib.taskRunArgs(task, args(['--split', 'test', '--confirm-test', '--prereg', 'x.json'])),
  /non-default --run-id/,
);
expectThrow(
  'test confirmation',
  () => lib.taskRunArgs(task, args(['--split', 'test', '--run-id', 'frozen-v1', '--prereg', 'x.json'])),
  /--confirm-test/,
);
expectThrow(
  'test preregistration',
  () => lib.taskRunArgs(task, args(['--split', 'test', '--run-id', 'frozen-v1', '--confirm-test'])),
  /--prereg PATH/,
);
expectThrow(
  'legacy model alias',
  () => lib.resolveConfig(args(['--model', 'deepseek-chat'])),
  /retired\/ambiguous/,
);

for (const promptProfile of task.promptProfiles) {
  const effective = lib.taskRunArgs(
    task,
    args(['--split', 'valid', '--prompt-profile', promptProfile, '--model', 'deepseek-v4-pro']),
  );
  if (effective.split !== 'valid' || effective.promptProfile !== promptProfile) {
    throw new Error(`effective protocol mismatch for ${promptProfile}`);
  }
}

const usage = lib.aggregateUsage([
  { usage: { prompt_tokens: 10, completion_tokens: 2, completion_tokens_details: { reasoning_tokens: 1 } } },
  { usage: { prompt_tokens: 7, completion_tokens: 3, completion_tokens_details: { reasoning_tokens: 2 } } },
  { usage: null },
]);
if (
  usage.requests_with_usage !== 2 ||
  usage.totals.prompt_tokens !== 17 ||
  usage.totals.completion_tokens !== 5 ||
  usage.totals.completion_tokens_details.reasoning_tokens !== 3
) {
  throw new Error('usage aggregation selftest failed');
}

lib.assertArtifactIdentity({ task: 'psysuicide', split: 'test' }, { task: 'psysuicide', split: 'test' });
expectThrow(
  'preregistration identity mismatch',
  () => lib.assertArtifactIdentity({ task: 'psysuicide', split: 'valid' }, { task: 'psysuicide', split: 'test' }),
  /preregistration mismatch for split/,
);
const committedRevision = lib.committedArtifactRevision(
  fileURLToPath(new URL('../.env.example', import.meta.url)),
);
if (!/^[0-9a-f]{40}$/.test(committedRevision)) {
  throw new Error('committed artifact revision selftest failed');
}
expectThrow(
  'preregistration outside repository',
  () => lib.committedArtifactRevision('/tmp/not-a-preregistration.json'),
  /committed file inside this repository/,
);

console.log('PsySUICIDE protocol check PASS: split isolation, 3 prompt profiles, legacy-model block, exact committed preregistration, usage aggregation');
