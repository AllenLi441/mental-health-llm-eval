#!/usr/bin/env node
// eval-suite runner.
//   node run.mjs list
//   node run.mjs <task-key> [--selftest] [--sample N] [--concurrency C] [--model M] ...
//   node run.mjs all [--selftest | flags]        (runs every task sequentially)
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { SUITE, parseArgs, runTask } from './lib.mjs';

const MODULES = [
  './tasks/emobench.mjs', './tasks/mdd5k.mjs', './tasks/psysuicide.mjs',
  './tasks/cbtbench.mjs', './tasks/mentalmanip.mjs', './tasks/imhi.mjs',
  './tasks/cpsyexam.mjs', './tasks/eatd.mjs',
];

async function registry() {
  const tasks = [];
  for (const m of MODULES) {
    const mod = await import(m);
    if (mod.variants) tasks.push(...mod.variants);
    else tasks.push(mod);
  }
  return tasks;
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const tasks = await registry();
  if (!cmd || cmd === 'list') {
    for (const t of tasks) {
      console.log(`${t.key.padEnd(18)} sample=${t.defaultSample === 0 ? 'full' : t.defaultSample}  ${t.description}`);
    }
    return;
  }
  const args = parseArgs(rest);
  const selected = cmd === 'all' ? tasks : tasks.filter((t) => t.key === cmd);
  if (!selected.length) {
    console.error(`unknown task "${cmd}". Use: node run.mjs list`);
    process.exit(2);
  }
  const summaries = [];
  for (const t of selected) {
    const s = await runTask(t, args); // sequential: one dataset at a time
    if (s) summaries.push(s);
  }
  if (cmd === 'all' && summaries.length) {
    mkdirSync(join(SUITE, 'results'), { recursive: true });
    const outPath = join(SUITE, 'results', `all-${args.runId}.summary.json`);
    writeFileSync(outPath, JSON.stringify(summaries, null, 2));
    console.log('\n===== SUITE OVERVIEW =====');
    for (const s of summaries) {
      console.log(`${s.task.padEnd(18)} n=${String(s.n).padEnd(5)} acc=${(s.accuracy * 100).toFixed(2)}%` +
        (s.weightedF1 !== undefined ? `  wF1=${s.weightedF1.toFixed(3)}` : '') +
        (s.positiveF1 !== undefined ? `  posF1=${s.positiveF1.toFixed(3)}` : ''));
    }
    console.log(`overview -> ${outPath}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
