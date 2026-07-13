// CPsyExam (COLING 2025) — Chinese psychology examination MCQ, accuracy.
// Full answered test split (3,902 questions), zero-shot, exact-match scoring
// (multi-answer questions must match the full answer set, as in the paper).
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS } from '../lib.mjs';

export const key = 'cpsyexam';
export const description = 'CPsyExam MCQ answering (KG+CA, single+multiple choice, zero-shot)';
export const labels = null; // free letter combinations; accuracy is the metric
export const defaultSample = 0; // full test set
export const maxTokens = 8;

export function load() {
  const dir = join(DATASETS, 'CPsyExam', 'data', 'extracted_with_answer', 'test');
  const items = [];
  for (const f of readdirSync(dir).sort()) {
    if (!f.endsWith('.json')) continue;
    const arr = JSON.parse(readFileSync(join(dir, f), 'utf8'));
    for (const q of arr) {
      const gold = String(q.answer || '').toUpperCase().match(/[A-E]/g);
      if (!gold) throw new Error(`no answer letters in ${f} id=${q.id}`);
      const qtype = q.question_type === 'single' ? 'single' : 'multiple'; // data uses single|multi
      items.push({
        id: q.id,
        gold: [...new Set(gold)].sort().join(''),
        qtype,
        kind: q.kind, // knowledge | analyse
        question: q.question,
        options: q.options,
      });
    }
  }
  return items;
}

export function assert(items) {
  if (items.length !== 3902) throw new Error(`expected 3902 questions, got ${items.length}`);
  for (const it of items) {
    if (!['single', 'multiple'].includes(it.qtype)) throw new Error(`bad qtype ${it.qtype}`);
    if (!['knowledge', 'analyse'].includes(it.kind)) throw new Error(`bad kind ${it.kind}`);
  }
}

export function messages(item) {
  const opts = Object.entries(item.options)
    .filter(([, v]) => v && v.trim())
    .map(([k, v]) => `${k}. ${v}`).join('\n');
  const inst = item.qtype === 'single'
    ? '这是一道单项选择题，只回答一个正确选项的字母，不要解释。'
    : '这是一道多项选择题，回答所有正确选项的字母（连写，如 ABD），不要解释。';
  return [
    { role: 'system', content: '你是心理学考试答题专家。' },
    { role: 'user', content: `${inst}\n\n题目：${item.question}\n${opts}\n\n答案：` },
  ];
}

export function parse(raw, item) {
  const letters = (raw || '').toUpperCase().match(/[A-E]/g);
  if (!letters) return { predicted: null, invalid: true };
  const uniq = [...new Set(letters)].sort();
  if (item.qtype === 'single') return { predicted: letters[0], invalid: false };
  return { predicted: uniq.join(''), invalid: false };
}

export function group(item) {
  return `${item.kind === 'knowledge' ? 'KG' : 'CA'}-${item.qtype === 'single' ? 'SCQ' : 'MAQ'}`;
}

export const comparisons = [
  { method: 'GPT-4 (paper Table 2, zero-shot)', metric: 'acc KG-SCQ/KG-MAQ/CA-SCQ/CA-MAQ', value: '76.56 / 10.76 / 60.33 / 13.00 (Avg. 67.43 incl. few-shot)' },
  { method: 'ChatGLM-Turbo (paper)', metric: 'Avg.', value: 64.58 },
  { method: 'ChatGPT (paper, zero-shot)', metric: 'acc KG-SCQ/KG-MAQ/CA-SCQ/CA-MAQ', value: '57.43 / 11.14 / 47.33 / 9.00 (Avg. 51.15)' },
];
