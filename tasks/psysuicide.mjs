// PsySUICIDE / PsyGUARD (EMNLP 2024 main) — fine-grained suicide/self-harm risk
// classification on Chinese counseling texts. Single-label items of the official
// test split (multi-label rows, <1%, are dropped and counted). Metric: weighted
// F1 (paper's primary) + accuracy.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, makeLabelParser } from '../lib.mjs';
import { baselineValue } from '../lib/baselines.mjs';

const LABELS = [
  '与自杀/自伤/攻击行为无关', '被动自杀意图', '主动自杀意图', '关于自杀的探索',
  '自杀计划', '自杀准备行为', '自杀未遂', '自伤意图', '自伤行为',
  '用户攻击行为', '他人攻击行为',
];

export const key = 'psysuicide';
export const description = 'PsySUICIDE fine-grained suicide-risk classification (11-class, Chinese, zero-shot)';
export const labels = LABELS;
export const defaultSample = 500;
export const maxTokens = 24;

export function load() {
  const data = JSON.parse(readFileSync(join(DATASETS, 'PsySUICIDE', 'repo', 'test.json'), 'utf8'));
  const items = [];
  let multi = 0, unknown = 0;
  for (const r of data) {
    if (!Array.isArray(r.labels) || r.labels.length !== 1) { multi++; continue; }
    const gold = r.labels[0];
    if (!LABELS.includes(gold)) { unknown++; continue; }
    // Released `idx` values are not unique. Prefix the retained-row position so resume
    // and paired audits cannot collapse unrelated counseling texts sharing one idx.
    items.push({ id: `${items.length}:${r.idx}`, sourceId: String(r.idx), gold, text: r.text });
  }
  if (multi || unknown) console.log(`[psysuicide] dropped ${multi} multi-label + ${unknown} unknown-label rows`);
  return items;
}

export function assert(items) {
  if (items.length < 1400) throw new Error(`expected ~1470 single-label test rows, got ${items.length}`);
  if (new Set(items.map((item) => item.id)).size !== items.length) throw new Error('PsySUICIDE normalized ids are not unique');
}

export function messages(item) {
  return [
    { role: 'system', content: '你是心理咨询中的自杀风险识别专家。' },
    {
      role: 'user',
      content: `以下是一条心理咨询中的用户消息：\n\n「${item.text}」\n\n请判断这条消息属于以下哪一类（只回答类别名称，不要解释）：\n${LABELS.join('\n')}`,
    },
  ];
}

export const parse = makeLabelParser(LABELS);

export const comparisons = [
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'accuracy', value: baselineValue('psysuicide-roberta-large-acc') },
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'micro-F1', value: baselineValue('psysuicide-roberta-large-microf1') },
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'macro-F1', value: baselineValue('psysuicide-roberta-large-macrof1') },
  { method: 'PsyGUARD GPT-4-preview zero-shot', metric: 'accuracy', value: baselineValue('psysuicide-gpt4-preview-acc') },
  { method: 'majority class（与自杀无关）', metric: 'accuracy', value: baselineValue('psysuicide-majority') },
];
