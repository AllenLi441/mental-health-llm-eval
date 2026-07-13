// EATD-Corpus (ICASSP 2022) — Chinese depression detection from interview
// transcripts. Text-only, validation split (79 volunteers, same split the paper
// reports on). Gold: SDS standard score (new_label.txt) >= 53 -> depressed.
// Paper text-only baselines: BiLSTM F1 0.65; multimodal fusion 0.71.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS } from '../lib.mjs';

const POS = 'depressed', NEG = 'non-depressed';

export const key = 'eatd-depression';
export const description = 'EATD depression detection (binary, text-only, validation split, zero-shot)';
export const labels = [POS, NEG];
export const positive = POS;
export const defaultSample = 0; // all 79
export const maxTokens = 8;

export function load() {
  const dir = join(DATASETS, 'EATD', 'data', 'EATD-Corpus', 'validation');
  const items = [];
  for (const v of readdirSync(dir).sort((a, b) => Number(a) - Number(b))) {
    const vd = join(dir, v);
    if (!statSync(vd).isDirectory()) continue;
    const read = (f) => readFileSync(join(vd, f), 'utf8').trim();
    const sds = Number(read('new_label.txt'));
    if (!Number.isFinite(sds)) throw new Error(`bad SDS score for volunteer ${v}`);
    items.push({
      id: `val-${v}`,
      gold: sds >= 53 ? POS : NEG,
      sds,
      positive: read('positive.txt'),
      neutral: read('neutral.txt'),
      negative: read('negative.txt'),
    });
  }
  return items;
}

export function assert(items) {
  if (items.length !== 79) throw new Error(`expected 79 validation volunteers, got ${items.length}`);
  const dep = items.filter((i) => i.gold === POS).length;
  if (dep < 10 || dep > 40) throw new Error(`suspicious depressed count: ${dep}`);
}

export function messages(item) {
  return [
    { role: 'system', content: '你是心理健康评估专家。' },
    {
      role: 'user',
      content: `以下是一位受访者在访谈中对三类问题的回答（均为口语转写）。\n\n【正性问题的回答】${item.positive}\n\n【中性问题的回答】${item.neutral}\n\n【负性问题的回答】${item.negative}\n\n请根据这些回答判断该受访者是否表现出抑郁倾向。只回答「抑郁」或「非抑郁」，不要解释。`,
    },
  ];
}

export function parse(raw) {
  if (!raw) return { predicted: null, invalid: true };
  const s = raw.trim();
  if (/非抑郁|不抑郁|无抑郁/.test(s)) return { predicted: NEG, invalid: false };
  if (/抑郁/.test(s)) return { predicted: POS, invalid: false };
  return { predicted: null, invalid: true };
}

export const comparisons = [
  { method: 'paper BiLSTM text-only (trained on train split)', metric: 'F1 (depressed)', value: 0.65 },
  { method: 'paper GRU audio-only', metric: 'F1', value: 0.66 },
  { method: 'paper multimodal fusion (SOTA in paper)', metric: 'F1', value: 0.71 },
];
