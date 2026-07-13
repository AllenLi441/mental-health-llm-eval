// MentalManip (ACL 2024 main) — mental manipulation detection in dialogues.
// Main task: binary manipulative / non-manipulative on the consensus subset
// (mentalmanip_con.csv, 2,915 dialogues). Metric: accuracy + F1.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, csvObjects } from '../lib.mjs';

const POS = 'yes', NEG = 'no';

export const key = 'mentalmanip';
export const description = 'MentalManip manipulation detection (binary, consensus subset, zero-shot)';
export const labels = [POS, NEG];
export const positive = POS;
export const defaultSample = 500;
export const maxTokens = 4;

export function load() {
  const path = join(DATASETS, 'MentalManip', 'repo', 'mentalmanip_dataset', 'mentalmanip_con.csv');
  const rows = csvObjects(readFileSync(path, 'utf8'));
  const items = [];
  for (const r of rows) {
    if (r.Manipulative !== '0' && r.Manipulative !== '1') continue;
    items.push({ id: r.ID, gold: r.Manipulative === '1' ? POS : NEG, dialogue: r.Dialogue });
  }
  return items;
}

export function assert(items) {
  if (items.length !== 2915) throw new Error(`expected 2915 consensus dialogues, got ${items.length}`);
}

export function messages(item) {
  return [
    { role: 'system', content: 'You are an expert in detecting mental manipulation in conversations.' },
    {
      role: 'user',
      content: `Read the following dialogue.\n\n${item.dialogue}\n\nDoes this dialogue contain elements of mental manipulation (e.g. gaslighting, guilt-tripping, intimidation, or other manipulation techniques)? Answer with only "yes" or "no".`,
    },
  ];
}

export function parse(raw) {
  const s = (raw || '').trim().toLowerCase();
  if (/^yes/.test(s)) return { predicted: POS, invalid: false };
  if (/^no/.test(s)) return { predicted: NEG, invalid: false };
  if (s.includes('yes') !== s.includes('no')) return { predicted: s.includes('yes') ? POS : NEG, invalid: false };
  return { predicted: null, invalid: true };
}

export const comparisons = [
  { method: 'GPT-4 zero-shot（论文 Table，binary detection）', metric: 'accuracy/F1', value: '论文含 GPT-4/Llama 基线，本地 PDF 可对表' },
  { method: 'majority class (manipulative, 69.2%)', metric: 'accuracy', value: 69.2 },
];
