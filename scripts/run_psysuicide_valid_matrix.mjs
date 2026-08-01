#!/usr/bin/env node
// Dry-run by default. Paid calls require --execute and an explicit approved budget.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { seededShuffle } from '../lib.mjs';
import * as task from '../tasks/psysuicide.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const RESULTS = join(ROOT, 'results');
const PRICING_PATH = join(ROOT, 'lib', 'deepseek_v4_pricing_2026-07-27.json');
const RETRY_RESERVE_FACTOR = 1.25;
const MAX_OUTPUT_TOKENS = 2048;
const ARMS = [
  { key: 'flash-baseline', model: 'deepseek-v4-flash', profile: 'baseline' },
  { key: 'pro-baseline', model: 'deepseek-v4-pro', profile: 'baseline' },
  { key: 'pro-taxonomy', model: 'deepseek-v4-pro', profile: 'taxonomy' },
  { key: 'pro-hierarchical', model: 'deepseek-v4-pro', profile: 'hierarchical' },
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
  for (let index = 0; index < argv.length; index++) {
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
    if (value && !/^[A-Za-z0-9._-]+$/.test(value)) throw new Error(`--${name} contains unsafe characters`);
  }
  if (args.execute && !args.batchId) throw new Error('--execute requires --batch-id');
  if (args.execute && !(args.approvedBudgetUsd > 0)) {
    throw new Error('--execute requires a positive --approved-budget-usd');
  }
  if (args.phase === 'full' && !args.smokeBatchId) {
    throw new Error('--phase full requires --smoke-batch-id');
  }
  return args;
}

function runId(batchId, phase, arm) {
  return `${batchId}-${phase}-${arm.key}`;
}

function resultStem(arm, id) {
  return `psysuicide-valid-${arm.profile}-${arm.model}-${id}`;
}

function summaryPath(arm, id) {
  return join(RESULTS, `${resultStem(arm, id)}.summary.json`);
}

function resultPath(arm, id) {
  return join(RESULTS, `${resultStem(arm, id)}.jsonl`);
}

function validateSummary(path, arm, id, expectedN) {
  if (!existsSync(path)) throw new Error(`required summary not found: ${path}`);
  const summary = JSON.parse(readFileSync(path, 'utf8'));
  const responseModels = [...new Set(summary.response_models || summary.api_models || [])];
  const checks = [
    [summary.task === 'psysuicide', 'task'],
    [summary.split === 'valid', 'split'],
    [summary.prompt_profile === arm.profile, 'prompt_profile'],
    [summary.requested_model === arm.model, 'requested_model'],
    [responseModels.length === 1 && responseModels[0] === arm.model, 'response_models'],
    [summary.run_id === id, 'run_id'],
    [summary.thinking === 'enabled', 'thinking'],
    [summary.reasoning_effort === 'high', 'reasoning_effort'],
    [summary.n === expectedN, 'n'],
    [summary.errors === 0, 'errors'],
    [summary.usage?.requests_with_usage === expectedN, 'usage coverage'],
  ];
  const failed = checks.filter(([ok]) => !ok).map(([, name]) => name);
  if (failed.length) throw new Error(`${path}: protocol checks failed: ${failed.join(', ')}`);
  return summary;
}

function actualCost(summary, pricing, model) {
  const rates = pricing.models[model];
  const totals = summary.usage?.totals || {};
  const promptTokens = Number(totals.prompt_tokens) || 0;
  let cacheHit = Number(totals.prompt_cache_hit_tokens) || 0;
  let cacheMiss = Number(totals.prompt_cache_miss_tokens) || 0;
  if (cacheHit + cacheMiss === 0) cacheMiss = promptTokens;
  const completionTokens = Number(totals.completion_tokens) || 0;
  return {
    prompt_tokens: promptTokens,
    prompt_cache_hit_tokens: cacheHit,
    prompt_cache_miss_tokens: cacheMiss,
    completion_tokens: completionTokens,
    estimated_usd: (
      cacheHit * rates.input_cache_hit +
      cacheMiss * rates.input_cache_miss +
      completionTokens * rates.output
    ) / 1_000_000,
  };
}

function loadPricing() {
  const pricing = JSON.parse(readFileSync(PRICING_PATH, 'utf8'));
  if (pricing.currency !== 'USD' || pricing.unit !== 'per_1m_tokens') {
    throw new Error(`unsupported pricing schema in ${PRICING_PATH}`);
  }
  for (const model of ['deepseek-v4-flash', 'deepseek-v4-pro']) {
    for (const field of ['input_cache_hit', 'input_cache_miss', 'output']) {
      if (!(pricing.models?.[model]?.[field] >= 0)) {
        throw new Error(`pricing missing ${model}.${field}`);
      }
    }
  }
  return pricing;
}

function selectItems(all, phase) {
  return phase === 'smoke' ? seededShuffle(all, 42).slice(0, 50) : all;
}

function estimateArm(arm, items, pricing) {
  const inputByteUpperBound = items.reduce((total, item) => {
    const messages = task.messages(item, { promptProfile: arm.profile });
    const contentBytes = messages.reduce((sum, message) => sum + Buffer.byteLength(message.content, 'utf8'), 0);
    return total + contentBytes + 32;
  }, 0);
  const outputTokenUpperBound = items.length * MAX_OUTPUT_TOKENS;
  const rates = pricing.models[arm.model];
  if (!rates) throw new Error(`pricing missing for ${arm.model}`);
  const inputUsd = (inputByteUpperBound / 1_000_000) * rates.input_cache_miss;
  const outputUsd = (outputTokenUpperBound / 1_000_000) * rates.output;
  return {
    calls: items.length,
    input_token_upper_bound: inputByteUpperBound,
    output_token_upper_bound: outputTokenUpperBound,
    single_attempt_upper_usd: inputUsd + outputUsd,
  };
}

function buildPlan(args) {
  const all = task.load({ split: 'valid' });
  task.assert(all, { split: 'valid' });
  const items = selectItems(all, args.phase);
  const pricing = loadPricing();
  const arms = ARMS.map((arm) => {
    const id = runId(args.batchId || 'DRY_RUN', args.phase, arm);
    const estimate = estimateArm(arm, items, pricing);
    const command = [
      process.execPath, 'run.mjs', 'psysuicide',
      '--split', 'valid',
      '--prompt-profile', arm.profile,
      '--model', arm.model,
      '--thinking', 'enabled',
      '--reasoning-effort', 'high',
      '--seed', '42',
      '--sample', args.phase === 'smoke' ? '50' : '0',
      '--concurrency', String(args.concurrency),
      '--run-id', id,
    ];
    return {
      ...arm,
      run_id: id,
      result_path: resultPath(arm, id),
      summary_path: summaryPath(arm, id),
      estimate,
      command,
    };
  });
  const singleAttemptUpperUsd = arms.reduce(
    (sum, arm) => sum + arm.estimate.single_attempt_upper_usd, 0
  );
  return {
    schema_version: 1,
    phase: args.phase,
    batch_id: args.batchId || null,
    seed: 42,
    sample: args.phase === 'smoke' ? 50 : 'full',
    pricing_snapshot: pricing,
    estimate_method: 'UTF-8 input bytes plus 32 tokens/request; all input cache-miss; 2048 output tokens/request',
    single_attempt_upper_usd: singleAttemptUpperUsd,
    retry_reserve_factor: RETRY_RESERVE_FACTOR,
    required_approved_budget_usd: singleAttemptUpperUsd * RETRY_RESERVE_FACTOR,
    arms,
  };
}

function validateSmokeGate(args) {
  for (const arm of ARMS) {
    const id = runId(args.smokeBatchId, 'smoke', arm);
    validateSummary(summaryPath(arm, id), arm, id, 50);
  }
}

function executePlan(args, plan) {
  if (args.phase === 'full') validateSmokeGate(args);
  if (args.approvedBudgetUsd + 1e-9 < plan.required_approved_budget_usd) {
    throw new Error(
      `approved budget $${args.approvedBudgetUsd.toFixed(2)} is below the reserved estimate ` +
      `$${plan.required_approved_budget_usd.toFixed(2)}`
    );
  }
  mkdirSync(RESULTS, { recursive: true });
  const manifestPath = join(RESULTS, `${args.batchId}-${args.phase}.matrix-plan.json`);
  if (existsSync(manifestPath)) throw new Error(`refusing to overwrite matrix plan: ${manifestPath}`);
  for (const arm of plan.arms) {
    if (existsSync(arm.result_path) || existsSync(arm.summary_path)) {
      throw new Error(`refusing to overwrite existing arm output for ${arm.key}`);
    }
  }
  writeFileSync(manifestPath, `${JSON.stringify({
    ...plan,
    generated_at: new Date().toISOString(),
    commands: plan.arms.map((arm) => arm.command.slice(1).join(' ')),
  }, null, 2)}\n`);
  const completedArms = [];
  let actualUsd = 0;
  for (let armIndex = 0; armIndex < plan.arms.length; armIndex++) {
    const arm = plan.arms[armIndex];
    console.log(`\n===== ${arm.key} =====`);
    const result = spawnSync(arm.command[0], arm.command.slice(1), {
      cwd: ROOT,
      env: process.env,
      stdio: 'inherit',
    });
    if (result.status !== 0) throw new Error(`${arm.key} exited with status ${result.status}`);
    const summary = validateSummary(
      arm.summary_path, arm, arm.run_id, plan.sample === 'full' ? 1459 : 50
    );
    const cost = actualCost(summary, plan.pricing_snapshot, arm.model);
    actualUsd += cost.estimated_usd;
    completedArms.push({ key: arm.key, summary_path: arm.summary_path, cost });
    const remainingReserved = plan.arms.slice(armIndex + 1).reduce(
      (sum, pending) => sum + pending.estimate.single_attempt_upper_usd * RETRY_RESERVE_FACTOR,
      0
    );
    if (
      armIndex + 1 < plan.arms.length &&
      actualUsd + remainingReserved > args.approvedBudgetUsd + 1e-9
    ) {
      throw new Error(
        `stopping before the next arm: actual $${actualUsd.toFixed(4)} plus remaining reserve ` +
        `$${remainingReserved.toFixed(4)} exceeds approved $${args.approvedBudgetUsd.toFixed(2)}`
      );
    }
  }
  const completionPath = join(RESULTS, `${args.batchId}-${args.phase}.matrix-summary.json`);
  if (existsSync(completionPath)) throw new Error(`refusing to overwrite matrix summary: ${completionPath}`);
  writeFileSync(completionPath, `${JSON.stringify({
    schema_version: 1,
    phase: args.phase,
    batch_id: args.batchId,
    completed_at: new Date().toISOString(),
    approved_budget_usd: args.approvedBudgetUsd,
    actual_estimated_usd: actualUsd,
    budget_overrun: actualUsd > args.approvedBudgetUsd,
    pricing_snapshot: plan.pricing_snapshot,
    arms: completedArms,
  }, null, 2)}\n`);
  console.log(`matrix ${args.phase} PASS -> ${completionPath}`);
}

function selftest() {
  const parsed = parseArgs(['--phase', 'smoke', '--batch-id', 'x', '--approved-budget-usd', '1', '--execute']);
  if (!parsed.execute || parsed.batchId !== 'x') throw new Error('argument parser selftest failed');
  const pricing = {
    models: {
      'deepseek-v4-pro': { input_cache_hit: 0.5, input_cache_miss: 1, output: 2 },
    },
  };
  const estimate = estimateArm(
    { model: 'deepseek-v4-pro', profile: 'baseline' },
    [{ text: 'fixture' }],
    pricing,
  );
  if (estimate.calls !== 1 || estimate.single_attempt_upper_usd <= 0) {
    throw new Error('cost estimator selftest failed');
  }
  const cost = actualCost(
    { usage: { totals: { prompt_tokens: 10, completion_tokens: 2 } } },
    pricing,
    'deepseek-v4-pro',
  );
  if (
    cost.prompt_cache_miss_tokens !== 10 ||
    !Number.isFinite(cost.estimated_usd) ||
    cost.estimated_usd <= 0
  ) {
    throw new Error('actual cost selftest failed');
  }
  for (const argv of [
    ['--execute', '--batch-id', 'x'],
    ['--phase', 'full', '--batch-id', 'x'],
    ['--batch-id', '../unsafe'],
  ]) {
    let rejected = false;
    try { parseArgs(argv); } catch { rejected = true; }
    if (!rejected) throw new Error(`expected argument rejection: ${argv.join(' ')}`);
  }
  console.log('PsySUICIDE valid matrix selftest PASS: dry-run default, paid confirmation, budget and smoke gates');
}

const args = parseArgs(process.argv.slice(2));
if (args.selftest) {
  selftest();
} else {
  const plan = buildPlan(args);
  console.log(JSON.stringify({
    phase: plan.phase,
    sample: plan.sample,
    calls: plan.arms.reduce((sum, arm) => sum + arm.estimate.calls, 0),
    single_attempt_upper_usd: Number(plan.single_attempt_upper_usd.toFixed(4)),
    required_approved_budget_usd: Number(plan.required_approved_budget_usd.toFixed(4)),
    pricing_retrieved_at: plan.pricing_snapshot.retrieved_at,
    dry_run: !args.execute,
    arms: plan.arms.map((arm) => ({
      key: arm.key,
      model: arm.model,
      profile: arm.profile,
      run_id: arm.run_id,
      command: arm.command.slice(1).join(' '),
    })),
  }, null, 2));
  if (args.execute) executePlan(args, plan);
}
