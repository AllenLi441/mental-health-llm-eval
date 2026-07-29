#!/usr/bin/env node
// Dry-run by default. Paid execution requires an explicit budget and a completed
// same-50 smoke gate before the full official-valid selection run.
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
const REASONING_OUTPUT_TOKEN_UPPER_BOUND = 2048;
const ARMS = [
  { key: 'taxonomy-control', model: 'deepseek-v4-pro', profile: 'taxonomy' },
  { key: 'taxonomy-v2', model: 'deepseek-v4-pro', profile: 'taxonomy-v2' },
  { key: 'fewshot-balanced', model: 'deepseek-v4-pro', profile: 'fewshot-balanced' },
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
    if (value && !/^[A-Za-z0-9._-]+$/.test(value)) {
      throw new Error(`--${name} contains unsafe characters`);
    }
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

function loadPricing() {
  const pricing = JSON.parse(readFileSync(PRICING_PATH, 'utf8'));
  const rates = pricing.models?.['deepseek-v4-pro'];
  if (
    pricing.currency !== 'USD'
    || pricing.unit !== 'per_1m_tokens'
    || !rates
    || ['input_cache_hit', 'input_cache_miss', 'output'].some((key) => !(rates[key] >= 0))
  ) {
    throw new Error('invalid DeepSeek V4 Pro pricing snapshot');
  }
  return pricing;
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

function estimateArm(arm, items, pricing) {
  const inputByteUpperBound = items.reduce((total, item) => {
    const messages = task.messages(item, { promptProfile: arm.profile });
    return total + messages.reduce(
      (subtotal, message) => subtotal + Buffer.byteLength(message.content, 'utf8') + 16,
      0,
    );
  }, 0);
  const outputTokenUpperBound = items.length * REASONING_OUTPUT_TOKEN_UPPER_BOUND;
  const rates = pricing.models[arm.model];
  return {
    calls: items.length,
    input_token_upper_bound: inputByteUpperBound,
    output_token_upper_bound: outputTokenUpperBound,
    single_attempt_upper_usd: (
      inputByteUpperBound * rates.input_cache_miss
      + outputTokenUpperBound * rates.output
    ) / 1_000_000,
  };
}

function buildPlan(args, allItems) {
  const items = args.phase === 'smoke' ? seededShuffle(allItems, 42).slice(0, 50) : allItems;
  const pricing = loadPricing();
  const arms = ARMS.map((arm) => {
    const id = runId(args.batchId || 'DRY_RUN', args.phase, arm);
    return {
      ...arm,
      run_id: id,
      summary_path: summaryPath(arm, id),
      estimate: estimateArm(arm, items, pricing),
      command: [
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
      ],
    };
  });
  const singleAttemptUpperUsd = arms.reduce(
    (total, arm) => total + arm.estimate.single_attempt_upper_usd,
    0,
  );
  return {
    schema_version: 1,
    protocol: 'PsySUICIDE v2 prompt selection on official valid; frozen train holdout excluded',
    phase: args.phase,
    batch_id: args.batchId || null,
    seed: 42,
    sample: args.phase === 'smoke' ? 50 : 'full',
    pricing_snapshot: pricing,
    estimate_method: 'UTF-8 input bytes as conservative token bound; all cache-miss; 2048 reasoning/output tokens per request',
    single_attempt_upper_usd: singleAttemptUpperUsd,
    retry_reserve_factor: RETRY_RESERVE_FACTOR,
    required_approved_budget_usd: singleAttemptUpperUsd * RETRY_RESERVE_FACTOR,
    arms,
  };
}

function readAndValidateSummary(path, arm, id, expectedN) {
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
    [summary.n === expectedN, 'n'],
    [summary.errors === 0, 'errors'],
    [summary.usage?.requests_with_usage === expectedN, 'usage coverage'],
  ];
  const failed = checks.filter(([ok]) => !ok).map(([, name]) => name);
  if (failed.length) throw new Error(`${path}: protocol checks failed: ${failed.join(', ')}`);
  return summary;
}

function validateSmokeGate(smokeBatchId) {
  for (const arm of ARMS) {
    const id = runId(smokeBatchId, 'smoke', arm);
    readAndValidateSummary(summaryPath(arm, id), arm, id, 50);
  }
}

function actualCost(summary, pricing, model) {
  const rates = pricing.models[model];
  const totals = summary.usage?.totals || {};
  const promptTokens = Number(totals.prompt_tokens) || 0;
  let cacheHit = Number(totals.prompt_cache_hit_tokens) || 0;
  let cacheMiss = Number(totals.prompt_cache_miss_tokens) || 0;
  if (cacheHit + cacheMiss === 0) cacheMiss = promptTokens;
  const completionTokens = Number(totals.completion_tokens) || 0;
  return (
    cacheHit * rates.input_cache_hit
    + cacheMiss * rates.input_cache_miss
    + completionTokens * rates.output
  ) / 1_000_000;
}

function executePlan(args, plan) {
  if (args.approvedBudgetUsd + 1e-12 < plan.required_approved_budget_usd) {
    throw new Error(
      `approved budget $${args.approvedBudgetUsd.toFixed(4)} is below reserved estimate `
      + `$${plan.required_approved_budget_usd.toFixed(4)}`,
    );
  }
  if (args.phase === 'full') validateSmokeGate(args.smokeBatchId);
  mkdirSync(RESULTS, { recursive: true });
  const planPath = join(RESULTS, `psysuicide-v2-${args.batchId}-${args.phase}.plan.json`);
  writeFileSync(planPath, `${JSON.stringify(plan, null, 2)}\n`);
  for (const arm of plan.arms) {
    if (existsSync(arm.summary_path)) {
      throw new Error(`refusing to overwrite existing completed arm: ${arm.summary_path}`);
    }
    const child = spawnSync(arm.command[0], arm.command.slice(1), {
      cwd: ROOT,
      env: process.env,
      encoding: 'utf8',
      stdio: 'inherit',
    });
    if (child.status !== 0) throw new Error(`${arm.key} failed with exit ${child.status}`);
  }
  const expectedN = args.phase === 'smoke' ? 50 : 1459;
  const summaries = plan.arms.map((arm) => {
    const summary = readAndValidateSummary(arm.summary_path, arm, arm.run_id, expectedN);
    return {
      key: arm.key,
      model: arm.model,
      profile: arm.profile,
      n: summary.n,
      accuracy: summary.accuracy,
      macro_f1: summary.macroF1,
      weighted_f1: summary.weightedF1,
      invalid: summary.invalid,
      errors: summary.errors,
      estimated_usd: actualCost(summary, plan.pricing_snapshot, arm.model),
      prompt_template_sha256: summary.prompt_template_sha256,
      dataset_manifest_sha256: summary.dataset_manifest_sha256,
    };
  });
  const completion = {
    schema_version: 1,
    protocol: plan.protocol,
    phase: args.phase,
    batch_id: args.batchId,
    summaries,
    total_estimated_usd: summaries.reduce((total, row) => total + row.estimated_usd, 0),
    publishing_boundary: 'aggregate metrics only; raw valid rows and model outputs remain ignored',
  };
  const completionPath = join(RESULTS, `psysuicide-v2-${args.batchId}-${args.phase}.completion.json`);
  writeFileSync(completionPath, `${JSON.stringify(completion, null, 2)}\n`);
  console.log(JSON.stringify(completion, null, 2));
}

function selftest() {
  const args = parseArgs(['--phase', 'smoke']);
  const fixture = {
    id: 'fixture',
    gold: task.labels[0],
    text: '仅用于离线计划自检的中性合成句。',
  };
  const plan = buildPlan(args, Array.from({ length: 50 }, (_, index) => ({
    ...fixture,
    id: `fixture-${index}`,
  })));
  if (plan.arms.length !== 3 || plan.arms.some((arm) => arm.estimate.calls !== 50)) {
    throw new Error('unexpected v2 smoke plan');
  }
  if (!(plan.required_approved_budget_usd > 0)) throw new Error('budget estimate is not positive');
  console.log('PsySUICIDE v2 valid matrix selftest PASS: 3 arms, same-50 smoke, full gate, explicit budget');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selftest) {
    selftest();
    return;
  }
  const all = task.load({ split: 'valid' });
  task.assert(all, { split: 'valid' });
  const plan = buildPlan(args, all);
  console.log(JSON.stringify(plan, null, 2));
  if (!args.execute) {
    console.log('Dry-run only. Add --execute --batch-id ID --approved-budget-usd N for paid calls.');
    return;
  }
  executePlan(args, plan);
}

main();
