// CBT-Bench (NAACL 2025 main) — cognitive model understanding tasks.
// Most items carry MULTIPLE gold labels, so the protocol here is top-1 hit
// accuracy: the model answers ONE label; correct iff it is in the gold set.
//   cbt-cd: cognitive distortion classification (10 classes, 146 items)
//   cbt-pc: primary core belief (3 classes, 184 items)
//   cbt-fc: fine-grained core belief (19 classes, 112 items)
// (CBT-QA is skipped: the released qa_test.json has no answer key.)
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, makeLabelParser } from '../lib.mjs';

const REPO = join(DATASETS, 'CBT-Bench', 'repo');

function loadTask(file, field) {
  const data = JSON.parse(readFileSync(join(REPO, file), 'utf8'));
  const items = [];
  for (const r of data) {
    const golds = Array.isArray(r[field]) ? r[field] : [r[field]];
    if (!golds.length || !golds[0]) continue;
    items.push({
      id: String(r.id), gold: golds[0], golds,
      situation: r.situation, thoughts: r.thoughts,
    });
  }
  return items;
}

function makeVariant({ key, description, file, field, labels, question, minItems }) {
  return {
    key, description, labels,
    defaultSample: 0, // all items (tasks are small)
    maxTokens: 24,
    load: () => loadTask(file, field),
    assert(items) {
      if (items.length < minItems) throw new Error(`${key}: expected >=${minItems} items, got ${items.length}`);
      const set = new Set(labels);
      for (const it of items) for (const g of it.golds) {
        if (!set.has(g)) throw new Error(`${key}: unknown gold "${g}"`);
      }
    },
    messages(item) {
      return [
        { role: 'system', content: 'You are an expert in cognitive behavioral therapy (CBT).' },
        {
          role: 'user',
          content: `A patient describes the following.\n\nSituation: ${item.situation}\nThoughts: ${item.thoughts}\n\n${question}\nAnswer with exactly one item from this list, and nothing else:\n${labels.join('\n')}`,
        },
      ];
    },
    parse: makeLabelParser(labels),
    // multi-label gold: correct iff prediction hits any gold label
    ok: (item, predicted) => predicted !== null && item.golds.includes(predicted),
    comparisons: [
      { method: '协议说明', metric: 'note', value: '多标签金标 → 采用 top-1 命中率；论文 Table 报告微调/零样本模型的 multi-label F1，口径不同，跑分后如需对表可加严格集合匹配' },
    ],
  };
}

const cd = makeVariant({
  key: 'cbt-cd',
  description: 'CBT-Bench cognitive distortion classification (10-class, top-1 hit)',
  file: 'distortions_test.json',
  field: 'distortions',
  minItems: 140,
  question: 'Which cognitive distortion is most present in the patient\'s thoughts?',
  labels: [
    'all-or-nothing thinking', 'overgeneralization', 'mental filter', 'should statements',
    'labeling', 'personalization', 'magnification', 'emotional reasoning',
    'mind reading', 'fortune-telling',
  ],
});

const pc = makeVariant({
  key: 'cbt-pc',
  description: 'CBT-Bench primary core belief classification (3-class, top-1 hit)',
  file: 'core_major_test.json',
  field: 'core_belief_major',
  minItems: 180,
  question: 'Which primary core belief category underlies the patient\'s thoughts?',
  labels: ['helpless', 'unlovable', 'worthless'],
});

const fc = makeVariant({
  key: 'cbt-fc',
  description: 'CBT-Bench fine-grained core belief classification (19-class, top-1 hit)',
  file: 'core_fine_test.json',
  field: 'core_belief_fine_grained',
  minItems: 100,
  question: 'Which fine-grained core belief best matches the patient\'s thoughts?',
  labels: [
    'I am incompetent', 'I am helpless', 'I am powerless, weak, vulnerable', 'I am a victim',
    'I am needy', 'I am trapped', 'I am out of control', 'I am a failure, loser',
    'I am defective', 'I am unlovable', 'I am unattractive', 'I am undesirable, unwanted',
    'I am bound to be rejected', 'I am bound to be abandoned', 'I am bound to be alone',
    'I am worthless, waste', 'I am immoral', 'I am bad - dangerous, toxic, evil',
    'I don’t deserve to live',
  ],
});

export const variants = [cd, pc, fc];
