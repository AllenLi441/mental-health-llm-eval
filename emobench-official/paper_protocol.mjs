#!/usr/bin/env node
/**
 * Closest executable EmoBench paper protocol:
 *   5 stochastic samples -> per-field majority, repeated for original + three
 *   deterministic option permutations -> mean accuracy across four runs.
 *
 * The ACL paper specifies 5-sample majority and four option orderings but the
 * released repository does not include the aggregation/permutation code.
 * Therefore the deterministic permutation seed, EU per-field vote, and tie
 * rule below are explicit implementation assumptions, not hidden as exact code
 * reproduction. Dry-run by default; paid calls require --execute and budget.
 */
import { createHash } from 'node:crypto';
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadTask, renderMessages, resultForSample } from './eval.mjs';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(SCRIPT_DIR);
const RESULTS = join(ROOT, 'results', 'emobench-paper');
const PRICING = JSON.parse(
  readFileSync(join(ROOT, 'lib', 'deepseek_v4_pricing_2026-07-27.json'), 'utf8')
);
const TASKS = ['EA', 'EU'];
const LANGS = ['en', 'zh'];
const PERMUTATIONS = 4;
const REPEATS = 5;
const SMOKE_PER_CELL = 2;
const FULL_ITEMS_PER_CELL = 200;
const FULL_ALL_CALLS = TASKS.length * LANGS.length * FULL_ITEMS_PER_CELL * PERMUTATIONS * REPEATS;
const SMOKE_ALL_CALLS = TASKS.length * LANGS.length * SMOKE_PER_CELL * PERMUTATIONS * REPEATS;
const STATIC_RESERVE = 1.25;
const FULL_SMOKE_CALIBRATION = 1.5;
const MAX_OUTPUT_TOKENS = 2048;

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function parseDotEnv() {
  const path = join(ROOT, '.env');
  if (!existsSync(path)) return {};
  return Object.fromEntries(
    readFileSync(path, 'utf8')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#') && line.includes('='))
      .map((line) => {
        const index = line.indexOf('=');
        return [line.slice(0, index).trim(), line.slice(index + 1).trim()];
      }),
  );
}

function parseArgs(argv) {
  const dot = parseDotEnv();
  const args = {
    phase: 'smoke',
    task: 'all',
    lang: 'all',
    model: dot.EVAL_MODEL || 'deepseek-v4-pro',
    baseUrl: (dot.EVAL_BASE_URL || 'https://api.deepseek.com/v1').replace(/\/+$/, ''),
    concurrency: 8,
    runId: '',
    smokeRunId: '',
    approvedBudgetUsd: 0,
    execute: false,
    selftest: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const next = () => argv[++index];
    if (flag === '--phase') args.phase = next();
    else if (flag === '--task') args.task = next();
    else if (flag === '--lang') args.lang = next();
    else if (flag === '--model') args.model = next();
    else if (flag === '--base-url') args.baseUrl = next().replace(/\/+$/, '');
    else if (flag === '--concurrency') args.concurrency = Number(next());
    else if (flag === '--run-id') args.runId = next();
    else if (flag === '--smoke-run-id') args.smokeRunId = next();
    else if (flag === '--approved-budget-usd') args.approvedBudgetUsd = Number(next());
    else if (flag === '--execute') args.execute = true;
    else if (flag === '--selftest') args.selftest = true;
    else throw new Error(`unknown flag: ${flag}`);
  }
  if (!['smoke', 'full'].includes(args.phase)) throw new Error('--phase must be smoke or full');
  if (![...TASKS, 'all'].includes(args.task)) throw new Error('--task must be EA, EU, or all');
  if (![...LANGS, 'all'].includes(args.lang)) throw new Error('--lang must be en, zh, or all');
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 16) {
    throw new Error('--concurrency must be an integer from 1 to 16');
  }
  if (args.runId && !/^[A-Za-z0-9._-]+$/.test(args.runId)) throw new Error('unsafe --run-id');
  if (args.smokeRunId && !/^[A-Za-z0-9._-]+$/.test(args.smokeRunId)) {
    throw new Error('unsafe --smoke-run-id');
  }
  if (args.execute && (!args.runId || !(args.approvedBudgetUsd > 0))) {
    throw new Error('--execute requires --run-id and positive --approved-budget-usd');
  }
  if (args.phase === 'full' && !args.smokeRunId) {
    throw new Error('--phase full requires --smoke-run-id');
  }
  if (args.model !== 'deepseek-v4-pro' && args.model !== 'deepseek-v4-flash') {
    throw new Error('paper protocol requires an explicit deepseek-v4-pro or deepseek-v4-flash model');
  }
  return args;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFrom(value) {
  return Number.parseInt(sha256(value).slice(0, 8), 16);
}

function permutation(length, key, permutationIndex) {
  const indices = Array.from({ length }, (_, index) => index);
  if (permutationIndex === 0) return indices;
  const random = mulberry32(seedFrom(`emobench-paper-1234\0${key}\0${permutationIndex}`));
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random() * (index + 1));
    [indices[index], indices[selected]] = [indices[selected], indices[index]];
  }
  return indices;
}

function reordered(values, indices) {
  return indices.map((index) => values[index]);
}

function permuteSample(task, sample, permutationIndex) {
  const key = `${task}\0${sample.language}\0${sample.qid}`;
  if (task === 'EA') {
    const order = permutation(sample.choices.length, `${key}\0choices`, permutationIndex);
    return { ...sample, choices: reordered(sample.choices, order) };
  }
  const emotionOrder = permutation(
    sample.emotion_choices.length, `${key}\0emotion`, permutationIndex
  );
  const causeOrder = permutation(
    sample.cause_choices.length, `${key}\0cause`, permutationIndex
  );
  return {
    ...sample,
    emotion_choices: reordered(sample.emotion_choices, emotionOrder),
    cause_choices: reordered(sample.cause_choices, causeOrder),
  };
}

function majority(values) {
  const valid = values.filter(Boolean);
  if (!valid.length) return '';
  const counts = new Map();
  for (const value of valid) counts.set(value, (counts.get(value) || 0) + 1);
  const maximum = Math.max(...counts.values());
  // Released code/paper do not define ties. Freeze first-observed among tied
  // maxima so the result is deterministic and auditable.
  return valid.find((value) => counts.get(value) === maximum);
}

function selected(values, requested) {
  return requested === 'all' ? values : [requested];
}

function chooseSmoke(items, task, lang) {
  return [...items]
    .sort((left, right) => sha256(`smoke\0${task}\0${lang}\0${left.qid}`)
      .localeCompare(sha256(`smoke\0${task}\0${lang}\0${right.qid}`)))
    .slice(0, SMOKE_PER_CELL);
}

async function buildCases(args) {
  const cases = [];
  const datasetCommitments = {};
  for (const task of selected(TASKS, args.task)) {
    const all = await loadTask(task);
    datasetCommitments[task] = sha256(
      all.map((sample) => JSON.stringify(sample)).join('\n')
    );
    for (const lang of selected(LANGS, args.lang)) {
      const cell = all.filter((sample) => sample.language === lang);
      if (cell.length !== 200) throw new Error(`${task}/${lang}: expected 200 rows`);
      const items = args.phase === 'smoke' ? chooseSmoke(cell, task, lang) : cell;
      for (const sample of items) {
        for (let permutationIndex = 0; permutationIndex < PERMUTATIONS; permutationIndex += 1) {
          const permuted = permuteSample(task, sample, permutationIndex);
          for (let repeat = 0; repeat < REPEATS; repeat += 1) {
            cases.push({
              key: `${task}:${lang}:${sample.qid}:p${permutationIndex}:r${repeat}`,
              task,
              lang,
              qid: String(sample.qid),
              permutation: permutationIndex,
              repeat,
              sample: permuted,
            });
          }
        }
      }
    }
  }
  return { cases, datasetCommitments };
}

function aggregateUsage(rows) {
  const totals = {};
  let covered = 0;
  for (const row of rows) {
    if (!row.usage || typeof row.usage !== 'object') continue;
    covered += 1;
    for (const [key, value] of Object.entries(row.usage)) {
      if (typeof value === 'number') totals[key] = (totals[key] || 0) + value;
    }
  }
  return { covered, totals };
}

function usageCost(usage, model) {
  const rates = PRICING.models[model];
  const prompt = Number(usage.totals.prompt_tokens) || 0;
  let hit = Number(usage.totals.prompt_cache_hit_tokens) || 0;
  let miss = Number(usage.totals.prompt_cache_miss_tokens) || 0;
  if (hit + miss === 0) miss = prompt;
  const output = Number(usage.totals.completion_tokens) || 0;
  return (hit * rates.input_cache_hit + miss * rates.input_cache_miss + output * rates.output) / 1e6;
}

function staticEstimate(cases, model) {
  const rates = PRICING.models[model];
  const inputBytes = cases.reduce((total, item) => {
    const messages = renderMessages(item.task, item.sample);
    return total + messages.reduce(
      (subtotal, message) => subtotal + Buffer.byteLength(message.content, 'utf8') + 16,
      0,
    );
  }, 0);
  return {
    input_byte_token_upper_bound: inputBytes,
    output_token_upper_bound: cases.length * MAX_OUTPUT_TOKENS,
    usd: (
      inputBytes * rates.input_cache_miss
      + cases.length * MAX_OUTPUT_TOKENS * rates.output
    ) / 1e6,
  };
}

function paths(runId) {
  return {
    rows: join(RESULTS, `${runId}.jsonl`),
    summary: join(RESULTS, `${runId}.summary.json`),
  };
}

function loadExisting(path) {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function validateRows(rows, args, caseKeys) {
  const seen = new Set();
  for (const row of rows) {
    if (seen.has(row.key)) throw new Error(`duplicate result key: ${row.key}`);
    seen.add(row.key);
    if (!caseKeys.has(row.key)) throw new Error(`result outside current plan: ${row.key}`);
    if (
      row.run_id !== args.runId
      || row.requested_model !== args.model
      || row.response_model !== args.model
      || row.temperature !== 0.6
      || row.protocol !== 'emobench-paper-5x4-v1'
    ) {
      throw new Error(`result provenance mismatch: ${row.key}`);
    }
    if (row.error || !row.usage) throw new Error(`incomplete paid row: ${row.key}`);
  }
  return seen;
}

async function callModel(args, messages) {
  const env = { ...parseDotEnv(), ...process.env };
  const key = env.EVAL_API_KEY;
  if (!key) throw new Error('EVAL_API_KEY is required; production credentials are ignored');
  const url = `${args.baseUrl}/chat/completions`;
  const body = {
    model: args.model,
    messages,
    temperature: 0.6,
    max_tokens: MAX_OUTPUT_TOKENS,
    thinking: { type: 'enabled' },
    reasoning_effort: 'high',
  };
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await response.text();
      if (!response.ok) {
        if ((response.status === 429 || response.status >= 500) && attempt < 3) {
          await new Promise((accept) => setTimeout(accept, 500 * attempt));
          continue;
        }
        throw new Error(`fatal HTTP ${response.status}; paper-protocol run aborted`);
      }
      const data = JSON.parse(text);
      return {
        content: data.choices?.[0]?.message?.content || '',
        responseModel: data.model || null,
        fingerprint: data.system_fingerprint || null,
        usage: data.usage || null,
        attempts: attempt,
      };
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error('unreachable retry state');
}

async function runWithConcurrency(items, concurrency, worker) {
  let index = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (index < items.length) {
      const current = items[index++];
      await worker(current);
    }
  });
  await Promise.all(workers);
}

function summarize(args, rows, datasetCommitments) {
  const taskSummaries = {};
  for (const task of selected(TASKS, args.task)) {
    const taskRows = rows.filter((row) => row.task === task);
    const majorityRows = [];
    const groupKeys = [...new Set(taskRows.map(
      (row) => `${row.lang}:${row.qid}:p${row.permutation}`
    ))].sort();
    for (const groupKey of groupKeys) {
      const votes = taskRows
        .filter((row) => `${row.lang}:${row.qid}:p${row.permutation}` === groupKey)
        .sort((left, right) => left.repeat - right.repeat);
      if (votes.length !== REPEATS) throw new Error(`${groupKey}: expected ${REPEATS} votes`);
      if (task === 'EA') {
        const answer = majority(votes.map((row) => row.answer));
        majorityRows.push({
          task, lang: votes[0].lang, qid: votes[0].qid,
          permutation: votes[0].permutation,
          correct: answer === votes[0].label,
          invalid: !answer,
        });
      } else {
        const emotion = majority(votes.map((row) => row.emo_answer));
        const cause = majority(votes.map((row) => row.cause_answer));
        majorityRows.push({
          task, lang: votes[0].lang, qid: votes[0].qid,
          permutation: votes[0].permutation,
          correct: emotion === votes[0].emo_label && cause === votes[0].cause_label,
          invalid: !emotion || !cause,
        });
      }
    }
    const permutations = {};
    for (let index = 0; index < PERMUTATIONS; index += 1) {
      const current = majorityRows.filter((row) => row.permutation === index);
      permutations[index] = {
        n: current.length,
        correct: current.filter((row) => row.correct).length,
        accuracy: current.filter((row) => row.correct).length / current.length,
        invalid_majorities: current.filter((row) => row.invalid).length,
        by_language: Object.fromEntries(selected(LANGS, args.lang).map((lang) => {
          const languageRows = current.filter((row) => row.lang === lang);
          return [lang, {
            n: languageRows.length,
            accuracy: languageRows.filter((row) => row.correct).length / languageRows.length,
          }];
        })),
      };
    }
    taskSummaries[task] = {
      dataset_commitment_sha256: datasetCommitments[task],
      calls: taskRows.length,
      unique_items: new Set(taskRows.map((row) => `${row.lang}:${row.qid}`)).size,
      permutations,
      paper_score_mean_accuracy: Object.values(permutations)
        .reduce((total, value) => total + value.accuracy, 0) / PERMUTATIONS,
      paper_score_by_language: Object.fromEntries(selected(LANGS, args.lang).map((lang) => [
        lang,
        Object.values(permutations)
          .reduce((total, value) => total + value.by_language[lang].accuracy, 0) / PERMUTATIONS,
      ])),
    };
  }
  const usage = aggregateUsage(rows);
  return {
    schema_version: 1,
    protocol: 'emobench-paper-5x4-v1',
    protocol_fidelity: {
      paper_specified: '5 stochastic samples with majority vote; original plus three random option orderings; mean of four accuracies; temperature 0.6',
      implementation_assumptions: 'deterministic permutation seed=1234; EU majority voted independently per field; ties choose first observed tied maximum',
      released_repository_gap: 'official repository commit 3b4f84312541f8469e909965e1fff3691ef85c62 does not implement the paper aggregation or permutations',
    },
    phase: args.phase,
    run_id: args.runId,
    model: args.model,
    temperature: 0.6,
    permutations: PERMUTATIONS,
    repeats: REPEATS,
    calls: rows.length,
    response_models: [...new Set(rows.map((row) => row.response_model))],
    fingerprints: [...new Set(rows.map((row) => row.fingerprint).filter(Boolean))],
    usage,
    estimated_usd: usageCost(usage, args.model),
    tasks: taskSummaries,
    publishing_boundary: 'aggregate summary public after review; per-call rows and model outputs stay ignored',
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selftest) {
    const sample = {
      qid: 'fixture', language: 'en', category: 'x', 'question type': 'Action',
      scenario: 'A friend is upset.', subject: 'Alex',
      choices: ['listen', 'mock', 'leave', 'ignore'], label: 'listen',
    };
    const permutations = Array.from({ length: 4 }, (_, index) => (
      permuteSample('EA', sample, index).choices
    ));
    if (new Set(permutations.map((value) => JSON.stringify(value))).size !== 4) {
      throw new Error('fixture permutations are not unique');
    }
    if (majority(['A', 'B', 'A', 'B', 'C']) !== 'A') throw new Error('tie rule mismatch');
    const parsed = resultForSample('EA', sample, '{"answer":"A"}');
    if (!parsed.correct) throw new Error('official parser integration failed');
    if (FULL_ALL_CALLS !== 16000 || SMOKE_ALL_CALLS !== 160) {
      throw new Error(`call-count invariant failed: full=${FULL_ALL_CALLS} smoke=${SMOKE_ALL_CALLS}`);
    }
    console.log('EmoBench paper protocol selftest PASS: 5x4 plan, full=16000 calls, smoke=160 calls, deterministic permutations, frozen tie rule, official prompt/parser');
    return;
  }

  const { cases, datasetCommitments } = await buildCases(args);
  const expectedCalls = selected(TASKS, args.task).length
    * selected(LANGS, args.lang).length
    * (args.phase === 'smoke' ? SMOKE_PER_CELL : FULL_ITEMS_PER_CELL)
    * PERMUTATIONS
    * REPEATS;
  if (cases.length !== expectedCalls) {
    throw new Error(`call-count invariant failed: built=${cases.length} expected=${expectedCalls}`);
  }
  const staticCost = staticEstimate(cases, args.model);
  let requiredBudget = staticCost.usd * STATIC_RESERVE;
  let budgetBasis = 'static conservative byte/token bound with 1.25x reserve';
  if (args.phase === 'full') {
    const smokePath = paths(args.smokeRunId).summary;
    if (!existsSync(smokePath)) throw new Error(`smoke summary not found: ${smokePath}`);
    const smoke = JSON.parse(readFileSync(smokePath, 'utf8'));
    if (
      smoke.protocol !== 'emobench-paper-5x4-v1'
      || smoke.phase !== 'smoke'
      || smoke.model !== args.model
      || smoke.calls <= 0
      || smoke.response_models?.[0] !== args.model
    ) {
      throw new Error('smoke summary does not satisfy the full-run gate');
    }
    requiredBudget = smoke.estimated_usd * (cases.length / smoke.calls) * FULL_SMOKE_CALIBRATION;
    budgetBasis = 'observed smoke usage extrapolated to full calls with 1.5x reserve';
  }
  const plan = {
    schema_version: 1,
    protocol: 'emobench-paper-5x4-v1',
    phase: args.phase,
    task: args.task,
    lang: args.lang,
    model: args.model,
    calls: cases.length,
    unique_items: cases.length / (PERMUTATIONS * REPEATS),
    permutations: PERMUTATIONS,
    repeats: REPEATS,
    temperature: 0.6,
    dataset_commitments: datasetCommitments,
    static_estimate: staticCost,
    budget_basis: budgetBasis,
    required_approved_budget_usd: requiredBudget,
    dry_run_default: true,
  };
  console.log(JSON.stringify(plan, null, 2));
  if (!args.execute) {
    console.log('Dry-run only. Add --execute --run-id ID --approved-budget-usd N for paid calls.');
    return;
  }
  if (args.approvedBudgetUsd + 1e-12 < requiredBudget) {
    throw new Error(
      `approved budget $${args.approvedBudgetUsd.toFixed(4)} below required $${requiredBudget.toFixed(4)}`
    );
  }
  mkdirSync(RESULTS, { recursive: true });
  const output = paths(args.runId);
  if (existsSync(output.summary)) throw new Error(`refusing to overwrite summary: ${output.summary}`);
  const rows = loadExisting(output.rows);
  const caseKeys = new Set(cases.map((item) => item.key));
  const seen = validateRows(rows, args, caseKeys);
  const pending = cases.filter((item) => !seen.has(item.key));
  console.log(`EmoBench paper protocol pending=${pending.length}/${cases.length}`);
  await runWithConcurrency(pending, args.concurrency, async (item) => {
    const response = await callModel(args, renderMessages(item.task, item.sample));
    const parsed = resultForSample(item.task, item.sample, response.content);
    const row = {
      key: item.key,
      task: item.task,
      lang: item.lang,
      qid: item.qid,
      permutation: item.permutation,
      repeat: item.repeat,
      label: parsed.label,
      answer: parsed.answer,
      emo_label: parsed.emo_label,
      emo_answer: parsed.emo_answer,
      cause_label: parsed.cause_label,
      cause_answer: parsed.cause_answer,
      correct: parsed.correct,
      invalid: parsed.invalid,
      protocol: 'emobench-paper-5x4-v1',
      run_id: args.runId,
      requested_model: args.model,
      response_model: response.responseModel,
      fingerprint: response.fingerprint,
      temperature: 0.6,
      attempts: response.attempts,
      usage: response.usage,
      error: null,
    };
    if (row.response_model !== args.model || !row.usage) {
      throw new Error(`response identity/usage missing for ${item.key}`);
    }
    rows.push(row);
    appendFileSync(output.rows, `${JSON.stringify(row)}\n`);
    if (rows.length % 25 === 0 || rows.length === cases.length) {
      console.log(`EmoBench paper protocol ${rows.length}/${cases.length}`);
    }
  });
  const summary = summarize(args, rows, datasetCommitments);
  if (summary.calls !== cases.length || summary.usage.covered !== cases.length) {
    throw new Error('completion gate failed: calls or usage coverage incomplete');
  }
  writeFileSync(output.summary, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify({
    calls: summary.calls,
    estimated_usd: summary.estimated_usd,
    tasks: Object.fromEntries(Object.entries(summary.tasks).map(([key, value]) => [
      key,
      {
        paper_score_mean_accuracy: value.paper_score_mean_accuracy,
        paper_score_by_language: value.paper_score_by_language,
      },
    ])),
    summary: output.summary,
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
