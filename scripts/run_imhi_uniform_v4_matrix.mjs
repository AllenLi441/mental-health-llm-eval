#!/usr/bin/env node
// Dry-run by default. Compare one control and one candidate protocol across all
// nine IMHI tasks; task-wise cherry-picking is forbidden.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { seededShuffle } from '../lib.mjs';
import { variants as tasks } from '../tasks/imhi.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const RESULTS = join(ROOT, 'results');
const PRICING = JSON.parse(
  readFileSync(join(ROOT, 'lib', 'deepseek_v4_pricing_2026-07-27.json'), 'utf8')
);
const MODEL = 'deepseek-v4-pro';
const MAX_OUTPUT_TOKENS = 2048;
const ARMS = [
  { key: 'uniform-control', profile: 'uniform-v3u' },
  { key: 'contrastive-candidate', profile: 'contrastive-v4' },
];

function parseArgs(argv) {
  const args = {
    phase: 'smoke',
    batchId: '',
    smokeBatchId: '',
    concurrency: 8,
    approvedBudgetUsd: 0,
    execute: false,
    selftest: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const next = () => argv[++index];
    if (flag === '--phase') args.phase = next();
    else if (flag === '--batch-id') args.batchId = next();
    else if (flag === '--smoke-batch-id') args.smokeBatchId = next();
    else if (flag === '--concurrency') args.concurrency = Number(next());
    else if (flag === '--approved-budget-usd') args.approvedBudgetUsd = Number(next());
    else if (flag === '--execute') args.execute = true;
    else if (flag === '--selftest') args.selftest = true;
    else throw new Error(`unknown flag: ${flag}`);
  }
  if (!['smoke', 'full'].includes(args.phase)) throw new Error('--phase must be smoke or full');
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 16) {
    throw new Error('--concurrency must be an integer from 1 to 16');
  }
  for (const [name, value] of [['batch-id', args.batchId], ['smoke-batch-id', args.smokeBatchId]]) {
    if (value && !/^[A-Za-z0-9._-]+$/.test(value)) throw new Error(`unsafe --${name}`);
  }
  if (args.execute && (!args.batchId || !(args.approvedBudgetUsd > 0))) {
    throw new Error('--execute requires --batch-id and positive --approved-budget-usd');
  }
  if (args.phase === 'full' && !args.smokeBatchId) {
    throw new Error('--phase full requires --smoke-batch-id');
  }
  return args;
}

function runId(batchId, phase, arm, task) {
  return `${batchId}-${phase}-${arm.key}-${task.key}`;
}

function summaryPath(batchId, phase, arm, task) {
  const id = runId(batchId, phase, arm, task);
  return join(RESULTS, `${task.key}-${arm.profile}-${MODEL}-${id}.summary.json`);
}

function usageCost(summary) {
  const rates = PRICING.models[MODEL];
  const totals = summary.usage?.totals || {};
  const prompt = Number(totals.prompt_tokens) || 0;
  let hit = Number(totals.prompt_cache_hit_tokens) || 0;
  let miss = Number(totals.prompt_cache_miss_tokens) || 0;
  if (hit + miss === 0) miss = prompt;
  const completion = Number(totals.completion_tokens) || 0;
  return (hit * rates.input_cache_hit + miss * rates.input_cache_miss + completion * rates.output) / 1e6;
}

function validateSummary(path, batchId, phase, arm, task, expectedN) {
  if (!existsSync(path)) throw new Error(`missing summary: ${path}`);
  const summary = JSON.parse(readFileSync(path, 'utf8'));
  const responseModels = [...new Set(summary.response_models || summary.api_models || [])];
  const checks = [
    [summary.task === task.key, 'task'],
    [summary.prompt_profile === arm.profile, 'prompt_profile'],
    [summary.requested_model === MODEL, 'requested_model'],
    [responseModels.length === 1 && responseModels[0] === MODEL, 'response_models'],
    [summary.run_id === runId(batchId, phase, arm, task), 'run_id'],
    [summary.n === expectedN, 'n'],
    [summary.errors === 0, 'errors'],
    [summary.usage?.requests_with_usage === expectedN, 'usage coverage'],
  ];
  const failed = checks.filter(([ok]) => !ok).map(([, name]) => name);
  if (failed.length) throw new Error(`${path}: protocol checks failed: ${failed.join(', ')}`);
  return summary;
}

function loadTaskPools() {
  return tasks.map((task) => {
    const items = task.load();
    if (!items.length) throw new Error(`${task.key}: empty dataset`);
    return { task, items };
  });
}

function buildPlan(args, pools) {
  const rates = PRICING.models[MODEL];
  const entries = [];
  for (const { task, items } of pools) {
    const selected = args.phase === 'smoke'
      ? seededShuffle(items, 42).slice(0, 10)
      : seededShuffle(items, 42).slice(0, Math.min(500, items.length));
    for (const arm of ARMS) {
      const inputBytes = selected.reduce((total, item) => (
        total + task.messages(item, { promptProfile: arm.profile })
          .reduce((subtotal, message) => subtotal + Buffer.byteLength(message.content, 'utf8') + 16, 0)
      ), 0);
      entries.push({
        task: task.key,
        arm: arm.key,
        profile: arm.profile,
        n: selected.length,
        run_id: runId(args.batchId || 'DRY_RUN', args.phase, arm, task),
        summary_path: summaryPath(args.batchId || 'DRY_RUN', args.phase, arm, task),
        static_upper_usd: (
          inputBytes * rates.input_cache_miss
          + selected.length * MAX_OUTPUT_TOKENS * rates.output
        ) / 1e6,
      });
    }
  }
  let required = entries.reduce((total, entry) => total + entry.static_upper_usd, 0) * 1.25;
  let budgetBasis = 'static conservative byte/token bound with 1.25x reserve';
  if (args.phase === 'full') {
    required = 0;
    for (const { task } of pools) {
      for (const arm of ARMS) {
        const smoke = validateSummary(
          summaryPath(args.smokeBatchId, 'smoke', arm, task),
          args.smokeBatchId, 'smoke', arm, task, 10,
        );
        const target = entries.find((entry) => entry.task === task.key && entry.arm === arm.key);
        required += usageCost(smoke) * (target.n / 10) * 1.5;
      }
    }
    budgetBasis = 'observed same-10 smoke usage extrapolated per task/arm with 1.5x reserve';
  }
  return {
    schema_version: 1,
    protocol: 'IMHI 9-task uniform v4 matrix; no task-wise cherry-picking',
    phase: args.phase,
    batch_id: args.batchId || null,
    model: MODEL,
    seed: 42,
    primary_aggregate: 'unweighted mean of nine task weighted-F1 values',
    advancement_rule: 'candidate mean weighted-F1 must exceed control and candidate may not regress by more than 2pp on more than three tasks',
    entries,
    budget_basis: budgetBasis,
    required_approved_budget_usd: required,
  };
}

function execute(args, plan, pools) {
  if (args.approvedBudgetUsd + 1e-12 < plan.required_approved_budget_usd) {
    throw new Error(
      `approved budget $${args.approvedBudgetUsd.toFixed(4)} below required $${plan.required_approved_budget_usd.toFixed(4)}`
    );
  }
  mkdirSync(RESULTS, { recursive: true });
  const planPath = join(RESULTS, `imhi-v4-${args.batchId}-${args.phase}.plan.json`);
  writeFileSync(planPath, `${JSON.stringify(plan, null, 2)}\n`);
  const summaries = [];
  for (const { task, items } of pools) {
    const expectedN = args.phase === 'smoke' ? 10 : Math.min(500, items.length);
    for (const arm of ARMS) {
      const path = summaryPath(args.batchId, args.phase, arm, task);
      if (existsSync(path)) throw new Error(`refusing to overwrite completed arm: ${path}`);
      const id = runId(args.batchId, args.phase, arm, task);
      const command = [
        process.execPath, 'run.mjs', task.key,
        '--prompt-profile', arm.profile,
        '--model', MODEL,
        '--thinking', 'enabled',
        '--reasoning-effort', 'high',
        '--seed', '42',
        '--sample', String(expectedN),
        '--concurrency', String(args.concurrency),
        '--run-id', id,
      ];
      const child = spawnSync(command[0], command.slice(1), {
        cwd: ROOT,
        env: process.env,
        encoding: 'utf8',
        stdio: 'inherit',
      });
      if (child.status !== 0) throw new Error(`${task.key}/${arm.key} failed`);
      const summary = validateSummary(path, args.batchId, args.phase, arm, task, expectedN);
      summaries.push({
        task: task.key,
        arm: arm.key,
        profile: arm.profile,
        n: summary.n,
        accuracy: summary.accuracy,
        macro_f1: summary.macroF1,
        weighted_f1: summary.weightedF1,
        invalid: summary.invalid,
        errors: summary.errors,
        estimated_usd: usageCost(summary),
        prompt_template_sha256: summary.prompt_template_sha256,
        dataset_manifest_sha256: summary.dataset_manifest_sha256,
      });
    }
  }
  const byArm = Object.fromEntries(ARMS.map((arm) => {
    const rows = summaries.filter((row) => row.arm === arm.key);
    return [arm.key, {
      mean_weighted_f1: rows.reduce((total, row) => total + row.weighted_f1, 0) / rows.length,
      tasks: Object.fromEntries(rows.map((row) => [row.task, row.weighted_f1])),
    }];
  }));
  const control = byArm['uniform-control'];
  const candidate = byArm['contrastive-candidate'];
  const severeRegressions = Object.keys(control.tasks).filter(
    (task) => candidate.tasks[task] - control.tasks[task] < -0.02
  );
  const advances = (
    candidate.mean_weighted_f1 > control.mean_weighted_f1
    && severeRegressions.length <= 3
  );
  const completion = {
    schema_version: 1,
    protocol: plan.protocol,
    phase: args.phase,
    batch_id: args.batchId,
    model: MODEL,
    summaries,
    aggregate: byArm,
    selection: {
      candidate_advances: advances,
      decision: advances ? 'ADVANCE_CONTRASTIVE_V4' : 'REJECT_CONTRASTIVE_V4',
      severe_regressions_over_2pp: severeRegressions,
      full_run_allowed_only_after_complete_smoke: args.phase === 'smoke',
    },
    total_estimated_usd: summaries.reduce((total, row) => total + row.estimated_usd, 0),
    publishing_boundary: 'aggregate metrics only; social-media rows, ids, predictions, and outputs remain ignored',
  };
  const path = join(RESULTS, `imhi-v4-${args.batchId}-${args.phase}.completion.json`);
  writeFileSync(path, `${JSON.stringify(completion, null, 2)}\n`);
  console.log(JSON.stringify(completion, null, 2));
}

function selftest() {
  if (tasks.length !== 9) throw new Error(`expected 9 IMHI tasks, got ${tasks.length}`);
  for (const task of tasks) {
    if (task.promptProfiles.join(',') !== 'uniform-v3u,contrastive-v4') {
      throw new Error(`${task.key}: prompt profiles missing`);
    }
    const item = {
      id: 'fixture', gold: task.labels[0], post: 'Synthetic benchmark fixture.',
      question: 'Does this match the requested category?',
    };
    for (const profile of task.promptProfiles) {
      const messages = task.messages(item, { promptProfile: profile });
      if (messages.length !== 2 || messages.some((message) => !message.content)) {
        throw new Error(`${task.key}/${profile}: invalid prompt`);
      }
    }
  }
  console.log('IMHI uniform v4 matrix selftest PASS: 9 tasks, 2 universal arms, same-10 smoke, aggregate advancement rule');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selftest) {
    selftest();
    return;
  }
  const pools = loadTaskPools();
  const plan = buildPlan(args, pools);
  console.log(JSON.stringify(plan, null, 2));
  if (!args.execute) {
    console.log('Dry-run only. Add --execute --batch-id ID --approved-budget-usd N for paid calls.');
    return;
  }
  execute(args, plan, pools);
}

main();
