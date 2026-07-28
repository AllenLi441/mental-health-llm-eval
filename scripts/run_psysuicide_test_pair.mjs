#!/usr/bin/env node
// Prepare or execute one frozen paired confirmatory test campaign.
// Execution is dry-run unless --execute and --confirm-test-pair are both present.
import {
  existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { committedArtifactRevision } from '../lib.mjs';
import * as task from '../tasks/psysuicide.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const RESULTS = join(ROOT, 'results');
const PRICING_PATH = join(ROOT, 'lib', 'deepseek_v4_pricing_2026-07-27.json');
const RETRY_RESERVE_FACTOR = 1.25;
const MAX_OUTPUT_TOKENS = 2048;
const EXPECTED_TEST_N = 1464;
const ARM_CONFIG = {
  A_flash_baseline: {
    model: 'deepseek-v4-flash', profile: 'baseline', role: 'reference',
  },
  B_pro_baseline: {
    model: 'deepseek-v4-pro', profile: 'baseline', role: 'candidate',
  },
  C_pro_taxonomy: {
    model: 'deepseek-v4-pro', profile: 'taxonomy', role: 'candidate',
  },
  D_pro_hierarchical: {
    model: 'deepseek-v4-pro', profile: 'hierarchical', role: 'candidate',
  },
};

function parseArgs(argv) {
  const args = {
    mode: '', campaignId: '', validAnalysis: '', campaign: '',
    approvedBudgetUsd: 0, concurrency: 8, execute: false,
    confirmTestPair: false, selftest: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const next = () => argv[++index];
    if (flag === '--mode') args.mode = next();
    else if (flag === '--campaign-id') args.campaignId = next();
    else if (flag === '--valid-analysis') args.validAnalysis = next();
    else if (flag === '--campaign') args.campaign = next();
    else if (flag === '--approved-budget-usd') args.approvedBudgetUsd = Number(next());
    else if (flag === '--concurrency') args.concurrency = Number(next());
    else if (flag === '--execute') args.execute = true;
    else if (flag === '--confirm-test-pair') args.confirmTestPair = true;
    else if (flag === '--selftest') args.selftest = true;
    else throw new Error(`unknown flag: ${flag}`);
  }
  if (args.selftest) return args;
  if (!['prepare', 'execute'].includes(args.mode)) {
    throw new Error('--mode must be prepare or execute');
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 16) {
    throw new Error('--concurrency must be an integer from 1 to 16');
  }
  if (args.campaignId && !/^[A-Za-z0-9._-]+$/.test(args.campaignId)) {
    throw new Error('--campaign-id contains unsafe characters');
  }
  if (args.mode === 'prepare') {
    if (!args.campaignId || !args.validAnalysis || !args.campaign) {
      throw new Error('prepare requires --campaign-id, --valid-analysis, and --campaign');
    }
    if (args.execute || args.confirmTestPair) {
      throw new Error('prepare does not accept paid execution flags');
    }
  }
  if (args.mode === 'execute') {
    if (!args.campaign) throw new Error('execute requires --campaign');
    if (args.execute && !args.confirmTestPair) {
      throw new Error('--execute requires --confirm-test-pair');
    }
    if (args.execute && !(args.approvedBudgetUsd > 0)) {
      throw new Error('--execute requires a positive --approved-budget-usd');
    }
  }
  return args;
}

function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function relativeRepoPath(path) {
  const absolute = resolve(path);
  const rel = relative(ROOT, absolute);
  if (!rel || rel.startsWith('..')) throw new Error(`path must be inside repository: ${absolute}`);
  return rel;
}

function validateValidSelection(analysis) {
  if (analysis.scope !== 'PsySUICIDE official valid split; developmental selection only') {
    throw new Error('valid analysis has the wrong scope');
  }
  if (analysis.n !== 1459 || analysis.selection?.confirmatory_claim_allowed !== false) {
    throw new Error('valid analysis is not the complete developmental selection artifact');
  }
  const winner = analysis.selection?.winner;
  if (!ARM_CONFIG[winner]) throw new Error(`unknown valid winner: ${winner}`);
  if (winner === 'A_flash_baseline') {
    throw new Error('valid winner equals the current reference; no superiority test campaign should run');
  }
  const selected = analysis.arms?.[winner];
  const expected = ARM_CONFIG[winner];
  if (
    selected?.model !== expected.model ||
    selected?.prompt_profile !== expected.profile ||
    selected?.errors !== 0 ||
    selected?.rows !== 1459
  ) {
    throw new Error('valid winner metadata is incomplete or inconsistent');
  }
  return winner;
}

function loadCommittedSelection(path) {
  const absolute = resolve(path);
  const commit = committedArtifactRevision(absolute);
  const analysis = JSON.parse(readFileSync(absolute, 'utf8'));
  const winner = validateValidSelection(analysis);
  return {
    absolute, relative: relativeRepoPath(absolute), commit,
    sha256: sha256File(absolute), analysis, winner,
  };
}

function preregPath(campaignPath, role) {
  const base = campaignPath.replace(/\.json$/i, '');
  return `${base}.${role}.prereg.json`;
}

function runId(campaignId, role) {
  return `${campaignId}-${role}`;
}

function resultPath(campaignId, role, config) {
  return join(
    RESULTS,
    `psysuicide-test-${config.profile}-${config.model}-${runId(campaignId, role)}.jsonl`,
  );
}

function summaryPath(campaignId, role, config) {
  return resultPath(campaignId, role, config).replace(/\.jsonl$/, '.summary.json');
}

function prepareArm(campaignId, role, config, outputPath) {
  const command = [
    process.execPath, 'run.mjs', 'psysuicide',
    '--split', 'test',
    '--prompt-profile', config.profile,
    '--model', config.model,
    '--thinking', 'enabled',
    '--reasoning-effort', 'high',
    '--seed', '42',
    '--sample', '0',
    '--run-id', runId(campaignId, role),
    '--prepare-prereg', outputPath,
  ];
  const result = spawnSync(command[0], command.slice(1), {
    cwd: ROOT, env: process.env, stdio: 'inherit',
  });
  if (result.status !== 0) throw new Error(`${role} preregistration failed`);
}

function prepareCampaign(args) {
  const campaignPath = resolve(args.campaign);
  relativeRepoPath(campaignPath);
  if (existsSync(campaignPath)) throw new Error(`refusing to overwrite campaign: ${campaignPath}`);
  const selection = loadCommittedSelection(args.validAnalysis);
  const referenceConfig = ARM_CONFIG.A_flash_baseline;
  const candidateConfig = ARM_CONFIG[selection.winner];
  const referencePrereg = preregPath(campaignPath, 'reference');
  const candidatePrereg = preregPath(campaignPath, 'candidate');
  for (const path of [referencePrereg, candidatePrereg]) {
    if (existsSync(path)) throw new Error(`refusing to overwrite preregistration: ${path}`);
  }
  prepareArm(args.campaignId, 'reference', referenceConfig, referencePrereg);
  prepareArm(args.campaignId, 'candidate', candidateConfig, candidatePrereg);
  const campaign = {
    schema_version: 1,
    campaign_id: args.campaignId,
    prepared_at: new Date().toISOString(),
    protocol: 'one-frozen-paired-confirmatory-test-campaign',
    valid_selection: {
      file: selection.relative,
      sha256: selection.sha256,
      commit: selection.commit,
      winner: selection.winner,
    },
    primary_inference: {
      metric: 'macro_f1',
      alpha: 0.05,
      paired_randomization_repetitions: 20000,
      paired_randomization_seed: 20260727,
      paired_bootstrap_repetitions: 20000,
      paired_bootstrap_seed: 20260727,
      superiority_rule: 'candidate delta > 0, two-sided paired-randomization p < 0.05, and paired-bootstrap CI95 lower bound > 0',
    },
    reference: {
      arm: 'A_flash_baseline',
      ...referenceConfig,
      run_id: runId(args.campaignId, 'reference'),
      preregistration: relativeRepoPath(referencePrereg),
      preregistration_sha256: sha256File(referencePrereg),
    },
    candidate: {
      arm: selection.winner,
      ...candidateConfig,
      run_id: runId(args.campaignId, 'candidate'),
      preregistration: relativeRepoPath(candidatePrereg),
      preregistration_sha256: sha256File(candidatePrereg),
    },
    test_policy: 'reference and candidate each run all 1464 retained single-label test cases once; resumes may only continue the same frozen files',
    publishing_boundary: 'no counseling text, per-case gold/prediction, raw output, or credential is included',
  };
  mkdirSync(dirname(campaignPath), { recursive: true });
  writeFileSync(campaignPath, `${JSON.stringify(campaign, null, 2)}\n`);
  console.log(`paired test campaign prepared without API calls -> ${campaignPath}`);
  console.log('Review and commit the campaign plus both preregistration files before execution.');
}

function loadPricing() {
  const pricing = JSON.parse(readFileSync(PRICING_PATH, 'utf8'));
  return pricing;
}

function estimateArm(items, config, pricing) {
  const inputUpper = items.reduce((total, item) => {
    const messages = task.messages(item, { promptProfile: config.profile });
    return total + messages.reduce(
      (sum, message) => sum + Buffer.byteLength(message.content, 'utf8'), 32
    );
  }, 0);
  const outputUpper = items.length * MAX_OUTPUT_TOKENS;
  const rates = pricing.models[config.model];
  return (
    inputUpper * rates.input_cache_miss + outputUpper * rates.output
  ) / 1_000_000;
}

function actualCost(summary, pricing, model) {
  const totals = summary.usage?.totals || {};
  const prompt = Number(totals.prompt_tokens) || 0;
  let hit = Number(totals.prompt_cache_hit_tokens) || 0;
  let miss = Number(totals.prompt_cache_miss_tokens) || 0;
  if (hit + miss === 0) miss = prompt;
  const completion = Number(totals.completion_tokens) || 0;
  const rates = pricing.models[model];
  return (hit * rates.input_cache_hit + miss * rates.input_cache_miss +
    completion * rates.output) / 1_000_000;
}

function loadCommittedCampaign(path) {
  const absolute = resolve(path);
  const executionCommit = committedArtifactRevision(absolute);
  const campaign = JSON.parse(readFileSync(absolute, 'utf8'));
  if (
    campaign.schema_version !== 1 ||
    campaign.protocol !== 'one-frozen-paired-confirmatory-test-campaign'
  ) throw new Error('invalid paired test campaign');
  const selectionPath = resolve(ROOT, campaign.valid_selection.file);
  committedArtifactRevision(selectionPath);
  if (sha256File(selectionPath) !== campaign.valid_selection.sha256) {
    throw new Error('valid selection artifact hash changed after campaign preparation');
  }
  const selection = JSON.parse(readFileSync(selectionPath, 'utf8'));
  if (validateValidSelection(selection) !== campaign.valid_selection.winner) {
    throw new Error('valid winner changed after campaign preparation');
  }
  if (campaign.candidate?.arm !== campaign.valid_selection.winner) {
    throw new Error('campaign candidate is not the frozen valid winner');
  }
  for (const role of ['reference', 'candidate']) {
    const arm = campaign[role];
    const config = ARM_CONFIG[arm.arm];
    if (
      !config || arm.model !== config.model || arm.profile !== config.profile ||
      arm.run_id !== runId(campaign.campaign_id, role)
    ) throw new Error(`${role} arm differs from frozen campaign`);
    const path = resolve(ROOT, arm.preregistration);
    committedArtifactRevision(path);
    if (sha256File(path) !== arm.preregistration_sha256) {
      throw new Error(`${role} preregistration hash differs from campaign`);
    }
  }
  return { absolute, campaign, executionCommit };
}

function validateSummary(path, campaign, role) {
  if (!existsSync(path)) throw new Error(`${role} summary missing: ${path}`);
  const summary = JSON.parse(readFileSync(path, 'utf8'));
  const arm = campaign[role];
  const responseModels = [...new Set(summary.response_models || [])];
  const checks = [
    [summary.task === 'psysuicide', 'task'],
    [summary.split === 'test', 'split'],
    [summary.prompt_profile === arm.profile, 'profile'],
    [summary.requested_model === arm.model, 'requested model'],
    [responseModels.length === 1 && responseModels[0] === arm.model, 'response model'],
    [summary.run_id === arm.run_id, 'run id'],
    [summary.n === EXPECTED_TEST_N, 'row count'],
    [summary.errors === 0, 'errors'],
    [summary.usage?.requests_with_usage === EXPECTED_TEST_N, 'usage coverage'],
    [summary.preregistration_sha256 === arm.preregistration_sha256, 'preregistration hash'],
    [/^[0-9a-f]{40}$/.test(summary.preregistration_commit || ''), 'preregistration commit'],
  ];
  const failed = checks.filter(([ok]) => !ok).map(([, label]) => label);
  if (failed.length) throw new Error(`${role} summary failed: ${failed.join(', ')}`);
  return summary;
}

function executeArm(args, loaded, role) {
  const { campaign } = loaded;
  const arm = campaign[role];
  const config = ARM_CONFIG[arm.arm];
  const output = resultPath(campaign.campaign_id, role, config);
  const command = [
    process.execPath, 'run.mjs', 'psysuicide',
    '--split', 'test',
    '--prompt-profile', arm.profile,
    '--model', arm.model,
    '--thinking', 'enabled',
    '--reasoning-effort', 'high',
    '--seed', '42',
    '--sample', '0',
    '--concurrency', String(args.concurrency),
    '--run-id', arm.run_id,
    '--confirm-test',
    '--prereg', resolve(ROOT, arm.preregistration),
  ];
  if (existsSync(output)) command.push('--resume', output);
  const result = spawnSync(command[0], command.slice(1), {
    cwd: ROOT, env: process.env, stdio: 'inherit',
  });
  if (result.status !== 0) throw new Error(`${role} test arm exited with status ${result.status}`);
  return validateSummary(summaryPath(campaign.campaign_id, role, config), campaign, role);
}

function executeCampaign(args) {
  const loaded = loadCommittedCampaign(args.campaign);
  const { campaign } = loaded;
  const items = task.load({ split: 'test' });
  task.assert(items, { split: 'test' });
  const pricing = loadPricing();
  const referenceEstimate = estimateArm(items, campaign.reference, pricing);
  const candidateEstimate = estimateArm(items, campaign.candidate, pricing);
  const requiredBudget = (referenceEstimate + candidateEstimate) * RETRY_RESERVE_FACTOR;
  const plan = {
    campaign_id: campaign.campaign_id,
    calls: items.length * 2,
    single_attempt_upper_usd: referenceEstimate + candidateEstimate,
    retry_reserve_factor: RETRY_RESERVE_FACTOR,
    required_approved_budget_usd: requiredBudget,
    dry_run: !args.execute,
    reference: campaign.reference,
    candidate: campaign.candidate,
  };
  console.log(JSON.stringify(plan, null, 2));
  if (!args.execute) return;
  if (args.approvedBudgetUsd + 1e-9 < requiredBudget) {
    throw new Error(
      `approved budget $${args.approvedBudgetUsd.toFixed(2)} is below reserved ` +
      `$${requiredBudget.toFixed(2)}`
    );
  }
  mkdirSync(RESULTS, { recursive: true });
  const referenceSummary = executeArm(args, loaded, 'reference');
  const referenceCost = actualCost(referenceSummary, pricing, campaign.reference.model);
  if (referenceCost + candidateEstimate * RETRY_RESERVE_FACTOR > args.approvedBudgetUsd) {
    throw new Error('stopping before candidate: actual reference cost plus candidate reserve exceeds approved budget');
  }
  const candidateSummary = executeArm(args, loaded, 'candidate');
  const candidateCost = actualCost(candidateSummary, pricing, campaign.candidate.model);
  const output = join(RESULTS, `${campaign.campaign_id}.test-pair-summary.json`);
  if (existsSync(output)) throw new Error(`refusing to overwrite test pair summary: ${output}`);
  writeFileSync(output, `${JSON.stringify({
    schema_version: 1,
    campaign: relativeRepoPath(loaded.absolute),
    campaign_sha256: sha256File(loaded.absolute),
    execution_commit: loaded.executionCommit,
    completed_at: new Date().toISOString(),
    approved_budget_usd: args.approvedBudgetUsd,
    actual_estimated_usd: referenceCost + candidateCost,
    reference_summary: relativeRepoPath(summaryPath(campaign.campaign_id, 'reference', campaign.reference)),
    candidate_summary: relativeRepoPath(summaryPath(campaign.campaign_id, 'candidate', campaign.candidate)),
  }, null, 2)}\n`);
  console.log(`paired confirmatory test campaign PASS -> ${output}`);
}

function selftest() {
  const analysis = {
    scope: 'PsySUICIDE official valid split; developmental selection only',
    n: 1459,
    selection: { winner: 'D_pro_hierarchical', confirmatory_claim_allowed: false },
    arms: {
      D_pro_hierarchical: {
        model: 'deepseek-v4-pro', prompt_profile: 'hierarchical',
        errors: 0, rows: 1459,
      },
    },
  };
  if (validateValidSelection(analysis) !== 'D_pro_hierarchical') {
    throw new Error('valid selection selftest failed');
  }
  let rejected = false;
  try {
    validateValidSelection({
      ...analysis,
      selection: { winner: 'A_flash_baseline', confirmatory_claim_allowed: false },
    });
  } catch { rejected = true; }
  if (!rejected) throw new Error('reference winner must stop confirmatory campaign');
  for (const argv of [
    ['--mode', 'execute', '--campaign', 'x', '--execute'],
    ['--mode', 'prepare', '--campaign-id', '../x', '--valid-analysis', 'a', '--campaign', 'b'],
  ]) {
    rejected = false;
    try { parseArgs(argv); } catch { rejected = true; }
    if (!rejected) throw new Error(`expected argument rejection: ${argv.join(' ')}`);
  }
  console.log('PsySUICIDE paired test campaign selftest PASS: committed selection, two frozen arms, paid confirmation');
}

const args = parseArgs(process.argv.slice(2));
if (args.selftest) selftest();
else if (args.mode === 'prepare') prepareCampaign(args);
else executeCampaign(args);
