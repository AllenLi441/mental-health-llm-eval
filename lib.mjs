// eval-suite/lib.mjs — shared infra for all dataset tasks. Zero deps, Node >= 18.
import { readFileSync, existsSync, mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const SUITE = dirname(fileURLToPath(import.meta.url));
export const DATASETS = process.env.EVAL_DATASETS_DIR
  ? resolve(process.env.EVAL_DATASETS_DIR)
  : join(SUITE, '..');

// ---------- args / config ----------
export function parseArgs(argv) {
  const a = {
    sample: -1, concurrency: 8, seed: 42, runId: 'run', selftest: false,
    dataCheck: false, resume: '', thinking: '', reasoningEffort: 'high',
  };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const next = () => argv[++i];
    if (k === '--sample') a.sample = Number(next());
    else if (k === '--concurrency') a.concurrency = Math.min(16, Number(next()));
    else if (k === '--model') a.model = next();
    else if (k === '--base-url') a.baseUrl = next();
    else if (k === '--seed') a.seed = Number(next());
    else if (k === '--run-id') a.runId = next();
    else if (k === '--resume') a.resume = next();
    else if (k === '--thinking') a.thinking = next();
    else if (k === '--reasoning-effort') a.reasoningEffort = next();
    else if (k === '--selftest') a.selftest = true;
    else if (k === '--data-check') a.dataCheck = true;
    else { console.error(`unknown flag: ${k}`); process.exit(2); }
  }
  return a;
}

function loadDotEnv() {
  if (process.env.EVAL_IGNORE_DOTENV === '1') return {};
  const p = join(SUITE, '.env');
  if (!existsSync(p)) return {};
  const out = {};
  for (const line of readFileSync(p, 'utf8').split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('#')) continue;
    const eq = t.indexOf('=');
    if (eq > 0) out[t.slice(0, eq).trim()] = t.slice(eq + 1).trim();
  }
  return out;
}

export function resolveConfig(args) {
  const dot = loadDotEnv();
  const pick = (...names) => {
    for (const n of names) { if (process.env[n]) return process.env[n]; if (dot[n]) return dot[n]; }
    return '';
  };
  // Batch evaluation must never silently borrow a production/application key.
  // Operators must provide a credential whose name makes its scope explicit.
  const apiKey = pick('EVAL_API_KEY');
  const baseUrl = (args.baseUrl || pick('EVAL_BASE_URL') || 'https://api.deepseek.com/v1').replace(/\/+$/, '');
  const model = args.model || pick('EVAL_MODEL') || 'deepseek-chat';
  const wireApi = pick('EVAL_WIRE_API') || 'chat'; // 'chat' | 'responses'
  const thinking = args.thinking || pick('EVAL_THINKING') || '';
  const reasoningEffort = args.reasoningEffort || pick('EVAL_REASONING_EFFORT') || 'high';
  if (thinking && !['enabled', 'disabled'].includes(thinking)) {
    throw new Error(`--thinking must be enabled or disabled, got ${thinking}`);
  }
  if (!['high', 'max'].includes(reasoningEffort)) {
    throw new Error(`--reasoning-effort must be high or max, got ${reasoningEffort}`);
  }
  let provider = 'openai-compatible';
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    if (host === 'api.deepseek.com') provider = 'deepseek';
    else if (host.endsWith('openai.com')) provider = 'openai';
  } catch { /* resolveConfig already preserves the caller's URL for fetch diagnostics */ }
  return {
    apiKey, baseUrl, model, wireApi, thinking, reasoningEffort, provider,
    credentialScope: 'EVAL_API_KEY',
  };
}

// ---------- CSV (RFC 4180: quoted fields, embedded commas/newlines/quotes) ----------
export function parseCSV(text) {
  const rows = [];
  let row = [], field = '', inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQ = false;
      } else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field); field = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; }
    else if (c !== '\r') field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

export function csvObjects(text) {
  const rows = parseCSV(text);
  const header = rows[0];
  return rows.slice(1).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}

// ---------- deterministic sampling ----------
export function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededShuffle(items, seed) {
  const rng = mulberry32(seed);
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---------- standalone selftest fixtures ----------
// Public releases intentionally omit sensitive/licensed datasets. These fixtures exercise
// each task's prompt and strict parser without reaching outside the repository or making API calls.
export function selftestTask(task) {
  let item;
  if (task.key === 'emobench-ea') item = { id: 'fixture', gold: 'A', lang: 'en', scenario: 'A teammate is upset.', subject: 'Alex', choices: ['listen', 'ignore', 'mock', 'leave'] };
  else if (task.key === 'emobench-eu') item = { id: 'fixture', gold: 'A+A', lang: 'en', scenario: 'Alex received good news.', subject: 'Alex', emotionChoices: ['joy', 'anger'], causeChoices: ['good news', 'rain'] };
  else if (task.key === 'mdd5k-diagnosis') item = { id: 'fixture', gold: '抑郁障碍', text: '来访者持续数周情绪低落并失去兴趣。' };
  else if (task.key === 'psysuicide') item = { id: 'fixture', gold: task.labels[0], text: '这是用于解析自检的中性合成句。' };
  else if (task.key.startsWith('cbt-')) item = { id: 'fixture', gold: task.labels[0], golds: [task.labels[0]], situation: 'A plan changed.', thoughts: 'Nothing ever works.' };
  else if (task.key === 'mentalmanip') item = { id: 'fixture', gold: 'yes', dialogue: 'If you cared, you would obey me.' };
  else if (task.key.startsWith('imhi-')) item = { id: 'fixture', gold: task.labels[0], post: 'Synthetic post for parser selftest.', question: 'Does the post match the requested category?' };
  else if (task.key === 'cpsyexam') item = { id: 'fixture', gold: 'A', qtype: 'single', kind: 'knowledge', question: '自检题', options: { A: '正确', B: '错误' } };
  else if (task.key === 'eatd-depression') item = { id: 'fixture', gold: 'depressed', positive: '没有兴趣', neutral: '难以集中', negative: '持续低落' };
  else throw new Error(`${task.key}: no standalone selftest fixture`);

  const messages = task.messages(item);
  if (!Array.isArray(messages) || messages.length < 2 || messages.some((m) => !m.role || !m.content)) {
    throw new Error(`${task.key}: invalid messages contract`);
  }
  const raw = task.key === 'emobench-eu' ? 'A A' : task.key === 'eatd-depression' ? '抑郁' : item.gold;
  const parsed = task.parse(raw, item);
  const ok = task.ok ? task.ok(item, parsed.predicted) : parsed.predicted === item.gold;
  if (parsed.invalid || !ok) throw new Error(`${task.key}: parser selftest failed for ${JSON.stringify(raw)} -> ${JSON.stringify(parsed)}`);
  if (task.group) task.group(item);
  console.log(`selftest OK: ${task.key} (prompt + parser; dataset intentionally not loaded)`);
}

export function dataCheckTask(task) {
  const items = task.load();
  if (!Array.isArray(items) || !items.length) throw new Error(`${task.key}: dataset loader returned no items`);
  if (task.assert) task.assert(items);
  const keys = items.map((item) => item.id);
  if (keys.some((key) => key === undefined || key === null || key === '')) throw new Error(`${task.key}: missing item id`);
  if (new Set(keys).size !== keys.length) throw new Error(`${task.key}: item ids are not unique; resume would be unsafe`);
  console.log(`data-check OK: ${task.key} n=${items.length}`);
}

// ---------- API ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function extractResponsesText(data) {
  if (typeof data.output_text === 'string') return data.output_text;
  for (const item of data.output || []) {
    if (item.type !== 'message') continue;
    for (const c of item.content || []) {
      if (c.type === 'output_text' && typeof c.text === 'string') return c.text;
    }
  }
  return '';
}

export async function chat(cfg, messages, maxTokens = 16) {
  // Responses API (wire_api=responses, e.g. gpt-5.x relays): reasoning tokens count
  // toward the output budget, so give generous headroom; omit temperature (rejected
  // by some reasoning models).
  const deepSeekThinking = cfg.wireApi === 'chat' && cfg.model.startsWith('deepseek-v4-') && cfg.thinking;
  const body = JSON.stringify(cfg.wireApi === 'responses'
    ? {
        model: cfg.model,
        input: messages.map((m) => ({ role: m.role, content: m.content })),
        max_output_tokens: 2048,
      }
    : {
        model: cfg.model,
        messages,
        ...(deepSeekThinking ? { thinking: { type: cfg.thinking } } : {}),
        ...(deepSeekThinking === 'enabled' ? { reasoning_effort: cfg.reasoningEffort } : { temperature: 0 }),
        max_tokens: deepSeekThinking === 'enabled' || cfg.model.includes('reasoner') ? 2048 : maxTokens,
      });
  const path = cfg.wireApi === 'responses' ? 'responses' : 'chat/completions';
  const backoffs = [2000, 8000, 20000];
  let lastErr = '';
  for (let attempt = 0; attempt <= backoffs.length; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), cfg.wireApi === 'responses' ? 180000 : 60000);
    try {
      const res = await fetch(`${cfg.baseUrl}/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.apiKey}` },
        body, signal: ctrl.signal,
      });
      if (res.status === 429 || res.status >= 500) {
        lastErr = `HTTP ${res.status}`;
        const ra = Number(res.headers.get('retry-after'));
        if (attempt < backoffs.length) await sleep(ra > 0 ? ra * 1000 : backoffs[attempt]);
        continue;
      }
      if (res.status === 401 || res.status === 402) {
        const fatal = new Error(`fatal HTTP ${res.status}; run aborted before recording this case`);
        fatal.fatal = true;
        throw fatal;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const data = await res.json();
      const raw = cfg.wireApi === 'responses'
        ? extractResponsesText(data)
        : (data.choices?.[0]?.message?.content ?? '');
      const apiModel = data.model; const fingerprint = data.system_fingerprint;
      if (cfg.wireApi === 'responses' && !raw && data.status && data.status !== 'completed') {
        return { error: `responses status=${data.status}` };
      }
      return { raw, apiModel, fingerprint, usage: data.usage ?? null, attempts: attempt + 1 };
    } catch (e) {
      if (e.fatal) throw e;
      if (e.message?.startsWith('HTTP 4')) return { error: e.message };
      lastErr = e.name === 'AbortError' ? 'timeout' : String(e.message || e);
      if (attempt < backoffs.length) await sleep(backoffs[attempt]);
    } finally {
      clearTimeout(timer);
    }
  }
  return { error: lastErr, attempts: backoffs.length + 1 };
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

// ---------- metrics ----------
export function wilson95(k, n) {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n;
  const d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n);
  const m = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [(c - m) / d, (c + m) / d];
}

function f1For(results, lab) {
  let tp = 0, fp = 0, fn = 0;
  for (const r of results) {
    if (r.predicted === lab && r.gold === lab) tp++;
    else if (r.predicted === lab) fp++;
    else if (r.gold === lab) fn++;
  }
  return tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
}

export function macroF1(results, labels) {
  return labels.reduce((s, lab) => s + f1For(results, lab), 0) / labels.length;
}

export function weightedF1(results, labels) {
  const n = results.length;
  let sum = 0;
  for (const lab of labels) {
    const support = results.filter((r) => r.gold === lab).length;
    sum += (support / n) * f1For(results, lab);
  }
  return sum;
}

export function positiveF1(results, lab) {
  return f1For(results, lab);
}

// ---------- runner ----------
export async function runTask(task, args) {
  const all = task.load();
  console.log(`[${task.key}] pool=${all.length}${task.labels ? ` labels=${task.labels.length}` : ''}`);
  if (task.assert) task.assert(all);

  if (args.selftest) {
    const dist = {};
    for (const it of all) dist[it.gold] = (dist[it.gold] || 0) + 1;
    console.log('gold distribution:', Object.entries(dist).sort((a, b) => b[1] - a[1])
      .map(([l, c]) => `${l}=${c}`).join(', '));
    const ex = all[0];
    console.log('--- sample messages ---');
    for (const m of task.messages(ex)) console.log(`[${m.role}] ${m.content.slice(0, 500)}`);
    console.log(`selftest OK: ${task.key}`);
    return null;
  }

  const cfg = resolveConfig(args);

  const sampleN = args.sample >= 0 ? args.sample : (task.defaultSample ?? 0);
  let items = sampleN > 0 && sampleN < all.length ? seededShuffle(all, args.seed).slice(0, sampleN) : all;

  const outDir = join(SUITE, 'results');
  mkdirSync(outDir, { recursive: true });
  const outPath = args.resume ||
    join(outDir, `${task.key}-${cfg.model.replace(/[^\w.-]/g, '_')}-${args.runId}.jsonl`);
  const done = new Set();
  const existingResults = [];
  if (args.resume && existsSync(outPath)) {
    for (const line of readFileSync(outPath, 'utf8').split('\n')) if (line) {
      const row = JSON.parse(line);
      existingResults.push(row);
      done.add(row.id);
    }
    const currentIds = new Set(items.map((item) => item.id));
    if (existingResults.length !== done.size) {
      throw new Error(
        `${task.key}: resume file contains duplicate ids (${existingResults.length} rows / ${done.size} unique). ` +
        'This is a legacy non-unique result and cannot be resumed safely; start a new run with normalized task ids.'
      );
    }
    const unknownIds = [...done].filter((id) => !currentIds.has(id));
    if (unknownIds.length) {
      throw new Error(
        `${task.key}: resume file has ${unknownIds.length} ids outside the current sampled dataset ` +
        `(first: ${unknownIds[0]}). Reuse the original --sample/--seed or start a new run.`
      );
    }
    console.log(`resume: ${done.size} unique ids / ${existingResults.length} existing rows`);
  }
  const allItems = items;
  items = items.filter((it) => !done.has(it.id));

  const caseDigests = new Map(allItems.map((item) => [item.id, sha256(canonicalJson(item))]));
  const datasetManifestSha256 = sha256(allItems.map((item) => caseDigests.get(item.id)).join('\n'));
  const promptTemplateSha256 = sha256([
    task.messages.toString(), task.parse.toString(), task.ok?.toString() ?? '',
    canonicalJson({ model: cfg.model, thinking: cfg.thinking || null, reasoning_effort: cfg.reasoningEffort }),
  ].join('\n'));

  if (items.length > 0 && !cfg.apiKey) {
    console.error('ERROR: no dedicated evaluation key. Set EVAL_API_KEY in env;');
    console.error('DEEPSEEK_API_KEY / OPENAI_API_KEY are intentionally ignored to protect production credentials,');
    console.error(`or write "EVAL_API_KEY=sk-..." to ${join(SUITE, '.env')}`);
    process.exit(1);
  }

  console.log(`[${task.key}] evaluating ${items.length} | model=${cfg.model} | out=${outPath}`);
  const t0 = Date.now();
  const runStartedAt = new Date(t0).toISOString();
  const results = [];
  let idx = 0, completed = 0;

  async function worker() {
    while (idx < items.length) {
      const it = items[idx++];
      const s0 = Date.now();
      const requestStartedAt = new Date(s0).toISOString();
      let rec;
      const r = await chat(cfg, task.messages(it), task.maxTokens ?? 16);
      const common = {
        requested_model: cfg.model,
        response_model: r.apiModel ?? null,
        provider: cfg.provider,
        fingerprint: r.fingerprint ?? null,
        run_id: args.runId,
        seed: args.seed,
        thinking: cfg.thinking || null,
        reasoning_effort: cfg.thinking === 'enabled' ? cfg.reasoningEffort : null,
        attempt: r.attempts ?? null,
        request_started_at: requestStartedAt,
        completed_at: new Date().toISOString(),
        case_sha256: caseDigests.get(it.id),
        prompt_template_sha256: promptTemplateSha256,
        dataset_manifest_sha256: datasetManifestSha256,
        usage: r.usage ?? null,
      };
      if (r.error) {
        rec = { id: it.id, gold: it.gold, predicted: null, raw: '', ok: false, invalid: false, error: r.error, ms: Date.now() - s0, ...common };
      } else {
        const { predicted, invalid } = task.parse(r.raw, it);
        const ok = task.ok ? task.ok(it, predicted) : predicted === it.gold;
        rec = { id: it.id, gold: it.gold, predicted, raw: r.raw.slice(0, 120), ok, invalid, error: null, ms: Date.now() - s0, api_model: r.apiModel ?? null, ...common };
      }
      results.push(rec);
      appendFileSync(outPath, JSON.stringify(rec) + '\n');
      completed++;
      if (completed % 25 === 0 || completed === items.length) {
        const acc = results.filter((x) => x.ok).length / results.length;
        console.log(`[${task.key}] ${completed}/${items.length}  acc=${(acc * 100).toFixed(1)}%  ` +
          `invalid=${results.filter((x) => x.invalid).length}  err=${results.filter((x) => x.error).length}  ` +
          `${((Date.now() - t0) / 1000).toFixed(0)}s`);
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(args.concurrency, items.length) }, worker));

  const combinedResults = [...existingResults, ...results];
  const n = combinedResults.length;
  const correct = combinedResults.filter((r) => r.ok).length;
  const summary = {
    task: task.key, description: task.description, model: cfg.model,
    credential_scope: cfg.credentialScope,
    requested_model: cfg.model,
    response_models: [...new Set(combinedResults.map((r) => r.response_model ?? r.api_model).filter(Boolean))],
    provider: cfg.provider,
    thinking: cfg.thinking || null,
    reasoning_effort: cfg.thinking === 'enabled' ? cfg.reasoningEffort : null,
    run_id: args.runId,
    seed: args.seed,
    sample: sampleN > 0 ? sampleN : 'full',
    concurrency: args.concurrency,
    started_at: runStartedAt,
    completed_at: new Date().toISOString(),
    prompt_template_sha256: promptTemplateSha256,
    dataset_manifest_sha256: datasetManifestSha256,
    api_models: [...new Set(combinedResults.map((r) => r.response_model ?? r.api_model).filter(Boolean))],
    fingerprints: [...new Set(combinedResults.map((r) => r.fingerprint).filter(Boolean))],
    n, correct,
    accuracy: n ? correct / n : 0,
    wilson95: wilson95(correct, n),
    invalid: combinedResults.filter((r) => r.invalid).length,
    errors: combinedResults.filter((r) => r.error).length,
    elapsedMs: Date.now() - t0,
  };
  if (task.labels) {
    summary.macroF1 = macroF1(combinedResults, task.labels);
    summary.weightedF1 = weightedF1(combinedResults, task.labels);
    const confusion = {};
    for (const r of combinedResults) {
      if (!r.ok && r.predicted) confusion[`${r.gold} -> ${r.predicted}`] = (confusion[`${r.gold} -> ${r.predicted}`] || 0) + 1;
    }
    summary.topConfusion = Object.entries(confusion).sort((a, b) => b[1] - a[1]).slice(0, 10)
      .map(([pair, count]) => ({ pair, count }));
  }
  if (task.positive) summary.positiveF1 = positiveF1(combinedResults, task.positive);
  if (task.group) {
    const groups = {};
    for (let i = 0; i < combinedResults.length; i++) {
      const item = itemsById(allItems, combinedResults[i].id);
      if (!item) throw new Error(`${task.key}: result id ${combinedResults[i].id} is absent from the current dataset`);
      const g = task.group(item);
      if (!groups[g]) groups[g] = { n: 0, correct: 0 };
      groups[g].n++;
      if (combinedResults[i].ok) groups[g].correct++;
    }
    summary.groups = Object.fromEntries(Object.entries(groups)
      .map(([g, v]) => [g, { ...v, accuracy: v.correct / v.n }]));
  }
  if (task.comparisons) summary.comparisons = task.comparisons;

  const sumPath = outPath.replace(/\.jsonl$/, '.summary.json');
  writeFileSync(sumPath, JSON.stringify(summary, null, 2));
  console.log(`\n===== ${task.key} =====`);
  console.log(`n=${n} accuracy=${(summary.accuracy * 100).toFixed(2)}% ` +
    `CI[${(summary.wilson95[0] * 100).toFixed(1)},${(summary.wilson95[1] * 100).toFixed(1)}]` +
    (summary.weightedF1 !== undefined ? ` weightedF1=${summary.weightedF1.toFixed(3)}` : '') +
    (summary.positiveF1 !== undefined ? ` positiveF1=${summary.positiveF1.toFixed(3)}` : '') +
    ` invalid=${summary.invalid} err=${summary.errors}`);
  if (summary.groups) for (const [g, v] of Object.entries(summary.groups)) {
    console.log(`  ${g}: ${(v.accuracy * 100).toFixed(1)}% (${v.correct}/${v.n})`);
  }
  console.log(`summary -> ${sumPath}`);
  return summary;
}

const _idCache = new WeakMap();
function itemsById(items, id) {
  let m = _idCache.get(items);
  if (!m) { m = new Map(items.map((it) => [it.id, it])); _idCache.set(items, m); }
  return m.get(id);
}

// generic single-label parser: exact match else unique label mention.
// Normalizes curly quotes/dashes/whitespace so surface variants still match.
function normalizeText(s) {
  return s.replace(/[‘’]/g, "'").replace(/[–—]/g, '-').replace(/\s+/g, ' ').trim();
}
export function makeLabelParser(labels, { lowercase = true } = {}) {
  const canon = labels.map((l) => normalizeText(lowercase ? l.toLowerCase() : l));
  return (raw) => {
    if (!raw) return { predicted: null, invalid: true };
    const s = normalizeText(lowercase ? raw.toLowerCase() : raw);
    const cleaned = s.replace(/^[^a-z一-鿿]+|[^a-z一-鿿]+$/gi, '');
    const exact = canon.indexOf(cleaned);
    if (exact >= 0) return { predicted: labels[exact], invalid: false };
    const hits = [];
    for (let i = 0; i < canon.length; i++) if (s.includes(canon[i])) hits.push(i);
    if (hits.length === 1) return { predicted: labels[hits[0]], invalid: false };
    return { predicted: null, invalid: true };
  };
}
