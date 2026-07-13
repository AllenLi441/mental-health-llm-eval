// eval-suite/lib.mjs — shared infra for all dataset tasks. Zero deps, Node >= 18.
import { readFileSync, existsSync, mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SUITE = dirname(fileURLToPath(import.meta.url));
export const DATASETS = join(SUITE, '..');

// ---------- args / config ----------
export function parseArgs(argv) {
  const a = { sample: -1, concurrency: 8, seed: 42, runId: 'run', selftest: false, resume: '' };
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
    else if (k === '--selftest') a.selftest = true;
    else { console.error(`unknown flag: ${k}`); process.exit(2); }
  }
  return a;
}

function loadDotEnv() {
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
  const apiKey = pick('EVAL_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY');
  const baseUrl = (args.baseUrl || pick('EVAL_BASE_URL') || 'https://api.deepseek.com/v1').replace(/\/+$/, '');
  const model = args.model || pick('EVAL_MODEL') || 'deepseek-chat';
  const wireApi = pick('EVAL_WIRE_API') || 'chat'; // 'chat' | 'responses'
  return { apiKey, baseUrl, model, wireApi };
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
  const body = JSON.stringify(cfg.wireApi === 'responses'
    ? {
        model: cfg.model,
        input: messages.map((m) => ({ role: m.role, content: m.content })),
        max_output_tokens: 2048,
      }
    : {
        model: cfg.model, messages, temperature: 0,
        max_tokens: cfg.model.includes('reasoner') ? 2048 : maxTokens,
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
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const data = await res.json();
      const raw = cfg.wireApi === 'responses'
        ? extractResponsesText(data)
        : (data.choices?.[0]?.message?.content ?? '');
      const apiModel = data.model; const fingerprint = data.system_fingerprint;
      if (cfg.wireApi === 'responses' && !raw && data.status && data.status !== 'completed') {
        return { error: `responses status=${data.status}` };
      }
      return { raw, apiModel, fingerprint };
    } catch (e) {
      if (e.message?.startsWith('HTTP 4')) return { error: e.message };
      lastErr = e.name === 'AbortError' ? 'timeout' : String(e.message || e);
      if (attempt < backoffs.length) await sleep(backoffs[attempt]);
    } finally {
      clearTimeout(timer);
    }
  }
  return { error: lastErr };
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
  if (!cfg.apiKey) {
    console.error('ERROR: no API key. Set EVAL_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY) in env,');
    console.error(`or write "EVAL_API_KEY=sk-..." to ${join(SUITE, '.env')}`);
    process.exit(1);
  }

  const sampleN = args.sample >= 0 ? args.sample : (task.defaultSample ?? 0);
  let items = sampleN > 0 && sampleN < all.length ? seededShuffle(all, args.seed).slice(0, sampleN) : all;

  const outDir = join(SUITE, 'results');
  mkdirSync(outDir, { recursive: true });
  const outPath = args.resume ||
    join(outDir, `${task.key}-${cfg.model.replace(/[^\w.-]/g, '_')}-${args.runId}.jsonl`);
  const done = new Set();
  if (args.resume && existsSync(outPath)) {
    for (const line of readFileSync(outPath, 'utf8').split('\n')) if (line) done.add(JSON.parse(line).id);
    console.log(`resume: ${done.size} already done`);
  }
  items = items.filter((it) => !done.has(it.id));

  console.log(`[${task.key}] evaluating ${items.length} | model=${cfg.model} | out=${outPath}`);
  const t0 = Date.now();
  const results = [];
  let idx = 0, completed = 0;

  async function worker() {
    while (idx < items.length) {
      const it = items[idx++];
      const s0 = Date.now();
      let rec;
      const r = await chat(cfg, task.messages(it), task.maxTokens ?? 16);
      if (r.error) {
        rec = { id: it.id, gold: it.gold, predicted: null, raw: '', ok: false, invalid: false, error: r.error, ms: Date.now() - s0 };
      } else {
        const { predicted, invalid } = task.parse(r.raw, it);
        const ok = task.ok ? task.ok(it, predicted) : predicted === it.gold;
        rec = { id: it.id, gold: it.gold, predicted, raw: r.raw.slice(0, 120), ok, invalid, error: null, ms: Date.now() - s0, api_model: r.apiModel ?? null, fingerprint: r.fingerprint ?? null };
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

  const n = results.length;
  const correct = results.filter((r) => r.ok).length;
  const summary = {
    task: task.key, description: task.description, model: cfg.model,
    api_models: [...new Set(results.map((r) => r.api_model).filter(Boolean))],
    fingerprints: [...new Set(results.map((r) => r.fingerprint).filter(Boolean))],
    n, correct,
    accuracy: n ? correct / n : 0,
    wilson95: wilson95(correct, n),
    invalid: results.filter((r) => r.invalid).length,
    errors: results.filter((r) => r.error).length,
    elapsedMs: Date.now() - t0,
  };
  if (task.labels) {
    summary.macroF1 = macroF1(results, task.labels);
    summary.weightedF1 = weightedF1(results, task.labels);
    const confusion = {};
    for (const r of results) {
      if (!r.ok && r.predicted) confusion[`${r.gold} -> ${r.predicted}`] = (confusion[`${r.gold} -> ${r.predicted}`] || 0) + 1;
    }
    summary.topConfusion = Object.entries(confusion).sort((a, b) => b[1] - a[1]).slice(0, 10)
      .map(([pair, count]) => ({ pair, count }));
  }
  if (task.positive) summary.positiveF1 = positiveF1(results, task.positive);
  if (task.group) {
    const groups = {};
    for (let i = 0; i < results.length; i++) {
      const g = task.group(itemsById(items, results[i].id));
      if (!groups[g]) groups[g] = { n: 0, correct: 0 };
      groups[g].n++;
      if (results[i].ok) groups[g].correct++;
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
