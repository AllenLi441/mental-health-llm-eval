// CPsyExam (COLING 2025) — Chinese psychology examination MCQ, accuracy.
// Full answered test split (3,902 questions), zero-shot, exact-match scoring
// (multi-answer questions must match the full answer set, as in the paper).
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS } from '../lib.mjs';
import { baselineValue } from '../lib/baselines.mjs';

export const key = 'cpsyexam';
export const description = 'CPsyExam MCQ answering (KG+CA, single+multiple choice, zero-shot)';
export const labels = null; // free letter combinations; accuracy is the metric
export const defaultSample = 0; // full test set
export const maxTokens = 32;
export const promptProfiles = ['legacy-zero-shot-v1', 'subject-json-v1'];
export const defaultPromptProfile = 'legacy-zero-shot-v1';
export const promptVersion = 'cpsyexam-prompts-v2';

export function load() {
  const dir = join(DATASETS, 'CPsyExam', 'data', 'extracted_with_answer', 'test');
  const items = [];
  const idOccurrences = new Map();
  for (const f of readdirSync(dir).sort()) {
    if (!f.endsWith('.json')) continue;
    const arr = JSON.parse(readFileSync(join(dir, f), 'utf8'));
    for (const q of arr) {
      const gold = String(q.answer || '').toUpperCase().match(/[A-E]/g);
      if (!gold) throw new Error(`no answer letters in ${f} id=${q.id}`);
      const qtype = q.question_type === 'single' ? 'single' : 'multiple'; // data uses single|multi
      const occurrence = (idOccurrences.get(q.id) ?? 0) + 1;
      idOccurrences.set(q.id, occurrence);
      items.push({
        // The released test split contains repeated ids. Keep the first id stable and
        // suffix later occurrences so resume/accounting cannot collapse distinct rows.
        id: occurrence === 1 ? q.id : `${q.id}#dup${occurrence}`,
        sourceId: q.id,
        gold: [...new Set(gold)].sort().join(''),
        qtype,
        kind: q.kind, // knowledge | analyse
        // Keep the released subject metadata available for prospective prompt
        // profiles. Historical result artifacts are not rewritten or rescored.
        subject_name: q.subject_name,
        question: q.question,
        options: q.options,
      });
    }
  }
  return items;
}

export function assert(items) {
  if (items.length !== 3902) throw new Error(`expected 3902 questions, got ${items.length}`);
  if (new Set(items.map((item) => item.id)).size !== items.length) throw new Error('CPsyExam normalized ids are not unique');
  for (const it of items) {
    if (!['single', 'multiple'].includes(it.qtype)) throw new Error(`bad qtype ${it.qtype}`);
    if (!['knowledge', 'analyse'].includes(it.kind)) throw new Error(`bad kind ${it.kind}`);
  }
}

function optionsText(item) {
  return Object.entries(item.options)
    .filter(([, value]) => value && value.trim())
    .map(([letter, value]) => `${letter}. ${value}`)
    .join('\n');
}

function legacyZeroShotMessages(item) {
  const opts = optionsText(item);
  const inst = item.qtype === 'single'
    ? '这是一道单项选择题，只回答一个正确选项的字母，不要解释。'
    : '这是一道多项选择题，回答所有正确选项的字母（连写，如 ABD），不要解释。';
  return [
    { role: 'system', content: '你是心理学考试答题专家。' },
    { role: 'user', content: `${inst}\n\n题目：${item.question}\n${opts}\n\n答案：` },
  ];
}

function subjectJsonMessages(item) {
  const opts = Object.entries(item.options)
    .filter(([, v]) => v && v.trim())
    .map(([k, v]) => `${k}. ${v}`).join('\n');
  const subject = String(item.subject_name || '').trim() || '未提供';
  const output = item.qtype === 'single'
    ? '{"answer":"B"}'
    : '{"answer":["A","B","D"]}';
  const typeRule = item.qtype === 'single'
    ? '这是单项选择题，answer 必须是一个 A-E 字母。'
    : '这是多项选择题，answer 必须是包含全部正确选项的 A-E 字母数组。';
  return [
    {
      role: 'system',
      content: '你是心理学考试答题专家。只依据题目、科目和选项作答，不输出解释。',
    },
    {
      role: 'user',
      content: `考试科目：${subject}\n${typeRule}\n只输出严格 JSON，格式为 ${output}；不要 Markdown 代码块，不要添加其他字段。\n\n题目：${item.question}\n${opts}`,
    },
  ];
}

export function messages(item, args = {}) {
  const profile = args.promptProfile || defaultPromptProfile;
  if (profile === 'legacy-zero-shot-v1') return legacyZeroShotMessages(item);
  if (profile === 'subject-json-v1') return subjectJsonMessages(item);
  throw new Error(`unsupported CPsyExam prompt profile: ${profile}`);
}

export function promptFingerprint(args = {}) {
  return {
    prompt_version: promptVersion,
    prompt_profile: args.promptProfile || defaultPromptProfile,
    subject_name_included: (args.promptProfile || defaultPromptProfile) === 'subject-json-v1',
    output_contract: (args.promptProfile || defaultPromptProfile) === 'subject-json-v1'
      ? 'strict-json-answer-v1'
      : 'legacy-letter-only-v1',
  };
}

function lettersFromJson(raw) {
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!value || Array.isArray(value) || typeof value !== 'object') return null;
  const keys = Object.keys(value);
  if (keys.length !== 1 || !['answer', 'ans'].includes(keys[0])) return null;
  const answer = value[keys[0]];
  if (Array.isArray(answer)) {
    if (!answer.length || answer.some((entry) => typeof entry !== 'string' || !/^[A-E]$/i.test(entry.trim()))) {
      return null;
    }
    return answer.map((entry) => entry.trim().toUpperCase());
  }
  if (typeof answer !== 'string') return null;
  const text = answer.trim().toUpperCase();
  if (!/^[A-E](?:[\s,，、/]*[A-E])*$/u.test(text)) return null;
  return text.match(/[A-E]/g);
}

function lettersFromAnchoredAnswer(raw) {
  const match = raw.match(
    /^(?:FINAL\s+ANSWER|ANSWER|最终答案|答案)\s*(?:是\s*)?[:：]?\s*([A-E](?:[\s,，、/]*[A-E])*)\s*[.。]?$/iu,
  );
  return match ? match[1].toUpperCase().match(/[A-E]/g) : null;
}

function lettersFromBareAnswer(raw) {
  if (!/^[A-E](?:[\s,，、/]*[A-E])*$/iu.test(raw)) return null;
  return raw.toUpperCase().match(/[A-E]/g);
}

export function parse(raw, item) {
  const text = typeof raw === 'string' ? raw.trim() : '';
  if (!text) return { predicted: null, invalid: true };
  const letters = lettersFromJson(text)
    || lettersFromAnchoredAnswer(text)
    || lettersFromBareAnswer(text);
  if (!letters) return { predicted: null, invalid: true };
  const uniq = [...new Set(letters)].sort();
  if (item.qtype === 'single') {
    if (uniq.length !== 1) return { predicted: null, invalid: true };
    return { predicted: uniq[0], invalid: false };
  }
  return { predicted: uniq.join(''), invalid: false };
}

export function group(item) {
  return `${item.kind === 'knowledge' ? 'KG' : 'CA'}-${item.qtype === 'single' ? 'SCQ' : 'MAQ'}`;
}

export const comparisons = [
  { method: 'GPT-4 (paper Table 2, zero-shot groups)', metric: 'acc KG-SCQ/KG-MAQ/CA-SCQ/CA-MAQ', value: baselineValue('cpsyexam-gpt4-zeroshot-groups') },
  { method: 'GPT-4 (derived strict zero-shot weighted)', metric: 'accuracy', value: baselineValue('cpsyexam-gpt4-zeroshot-weighted') },
  { method: 'ChatGLM-Turbo (paper Table 2 mixed best-of)', metric: 'Avg. across zero/few-shot maxima', value: baselineValue('cpsyexam-chatglm-turbo-avg') },
  { method: 'ChatGPT (paper Table 2 mixed best-of)', metric: 'Avg. across zero/few-shot maxima', value: baselineValue('cpsyexam-chatgpt-avg') },
];
