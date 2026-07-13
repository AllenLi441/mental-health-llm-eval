// EmoBench (ACL 2024) — Emotional Intelligence benchmark, EN+ZH, multiple choice.
//   emobench-ea: Emotional Application (400 items, 4 choices) — official metric accuracy
//   emobench-eu: Emotional Understanding (400 items) — emotion AND cause must both be
//                correct (official scoring), asked in one prompt as two lettered answers.
// Paper (Table 1/2): GPT-4 EA 75.50 (en) / 73.75 (zh); EU 59.75 (en) / 54.12 (zh).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS } from '../lib.mjs';

const DATA = join(DATASETS, 'EmoBench', 'repo', 'data');
const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

function loadJsonl(file) {
  return readFileSync(join(DATA, file), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

function lettered(choices) {
  return choices.map((c, i) => `${LETTERS[i]}. ${c}`).join('\n');
}

const ea = {
  key: 'emobench-ea',
  description: 'EmoBench Emotional Application (MCQ, en+zh, zero-shot)',
  labels: LETTERS.slice(0, 4),
  defaultSample: 0, // all 400
  maxTokens: 4,
  load() {
    return loadJsonl('EA.jsonl').map((r) => {
      const gold = r.choices.indexOf(r.label);
      if (gold < 0) throw new Error(`EA qid ${r.qid}: label not in choices`);
      return { id: `${r.language}-${r.qid}`, gold: LETTERS[gold], lang: r.language, scenario: r.scenario, subject: r.subject, choices: r.choices };
    });
  },
  assert(items) {
    if (items.length !== 400) throw new Error(`expected 400 EA items, got ${items.length}`);
  },
  messages(item) {
    const zh = item.lang === 'zh';
    return [
      { role: 'system', content: zh ? '你是情商测评的答题者。' : 'You are taking an emotional intelligence test.' },
      {
        role: 'user',
        content: zh
          ? `场景：${item.scenario}\n\n在这个场景中，${item.subject}最有效的做法是哪一项？\n${lettered(item.choices)}\n\n只回答一个选项字母。`
          : `Scenario: ${item.scenario}\n\nWhat is the most effective action for ${item.subject} in this scenario?\n${lettered(item.choices)}\n\nAnswer with only one option letter.`,
      },
    ];
  },
  parse(raw) {
    const m = (raw || '').toUpperCase().match(/[A-D]/);
    return m ? { predicted: m[0], invalid: false } : { predicted: null, invalid: true };
  },
  group: (item) => item.lang,
  comparisons: [
    { method: 'GPT-4 (paper, zero-shot)', metric: 'accuracy en/zh', value: '75.50 / 73.75' },
    { method: 'ChatGLM3-66B (paper, best open-source)', metric: 'accuracy en/zh', value: '65.50 / 59.12' },
    { method: 'human average (paper Fig. 5)', metric: 'note', value: 'above all LLMs' },
  ],
};

const eu = {
  key: 'emobench-eu',
  description: 'EmoBench Emotional Understanding (emotion + cause both correct, en+zh, zero-shot)',
  labels: null, // compound answers; accuracy is the metric
  defaultSample: 0, // all 400
  maxTokens: 8,
  load() {
    return loadJsonl('EU.jsonl').map((r) => {
      const e = r.emotion_choices.indexOf(r.emotion_label);
      const c = r.cause_choices.indexOf(r.cause_label);
      if (e < 0 || c < 0) throw new Error(`EU qid ${r.qid}: label not in choices`);
      return {
        id: `${r.language}-${r.qid}`, gold: `${LETTERS[e]}+${LETTERS[c]}`, lang: r.language,
        scenario: r.scenario, subject: r.subject,
        emotionChoices: r.emotion_choices, causeChoices: r.cause_choices,
      };
    });
  },
  assert(items) {
    if (items.length !== 400) throw new Error(`expected 400 EU items, got ${items.length}`);
  },
  messages(item) {
    const zh = item.lang === 'zh';
    return [
      { role: 'system', content: zh ? '你是情绪理解测评的答题者。' : 'You are taking an emotion understanding test.' },
      {
        role: 'user',
        content: zh
          ? `场景：${item.scenario}\n\n问题1：${item.subject}在场景结尾最主要的情绪是？\n${lettered(item.emotionChoices)}\n\n问题2：产生这种情绪的原因是？\n${lettered(item.causeChoices)}\n\n按「情绪字母 原因字母」格式回答，例如 "B D"，不要解释。`
          : `Scenario: ${item.scenario}\n\nQuestion 1: What is ${item.subject}'s main emotion at the end of the scenario?\n${lettered(item.emotionChoices)}\n\nQuestion 2: What is the cause of this emotion?\n${lettered(item.causeChoices)}\n\nAnswer in the format "emotion-letter cause-letter", e.g. "B D". No explanation.`,
      },
    ];
  },
  parse(raw, item) {
    const letters = (raw || '').toUpperCase().match(/[A-F]/g);
    if (!letters || letters.length < 2) return { predicted: null, invalid: true };
    const e = letters[0], c = letters[1];
    if (LETTERS.indexOf(e) >= item.emotionChoices.length) return { predicted: null, invalid: true };
    if (LETTERS.indexOf(c) >= item.causeChoices.length) return { predicted: null, invalid: true };
    return { predicted: `${e}+${c}`, invalid: false };
  },
  group: (item) => item.lang,
  comparisons: [
    { method: 'GPT-4 (paper, zero-shot, emotion+cause both correct)', metric: 'accuracy en/zh', value: '59.75 / 54.12' },
    { method: 'ChatGLM3-66B (paper)', metric: 'accuracy en/zh', value: '47.45 / 42.86' },
  ],
};

export const variants = [ea, eu];
