#!/usr/bin/env node
// Offline deterministic guard for PsySUICIDE split/profile/test protocol.
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

console.log('PsySUICIDE protocol check PASS: split isolation, 3 prompt profiles, legacy-model block, frozen-test gate');
