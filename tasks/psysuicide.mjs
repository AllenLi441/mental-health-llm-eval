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
export const defaultSamples = { valid: 0, test: 0 };
export const maxTokens = 24;
export const splits = ['train', 'valid', 'test'];
export const runSplits = ['valid', 'test'];
export const promptProfiles = ['baseline', 'taxonomy', 'hierarchical'];
export const defaultPromptProfile = 'baseline';
export const promptVersion = 'psysuicide-prompts-v1';

const OPERATIONAL_DEFINITIONS = [
  '与自杀/自伤/攻击行为无关：消息不表达用户自身或他人的相关意图、计划、准备、尝试、行为或攻击；不要因一般性的痛苦、绝望或负面情绪自动判为风险类别。',
  '被动自杀意图：希望死亡、消失或不再存在，但没有表达主动采取行动的意图。',
  '主动自杀意图：表达用户自身主动结束生命的想法或意图，但没有明确的计划、准备行为或已经开始的尝试。',
  '关于自杀的探索：以询问、了解、讨论或探索自杀相关信息为主，不能仅凭话题出现就推断为用户自身的主动意图。',
  '自杀计划：表达结束生命的意图，并明确存在具体计划；不要把只有主动想法但没有计划的文本归入此类。',
  '自杀准备行为：已经实施为自杀尝试做准备或创造条件的行为，但尚未开始自杀尝试。',
  '自杀未遂：已经开始实施自杀尝试，或明确叙述已经发生的未遂事件。',
  '自伤意图：表达故意伤害自己的想法或意图，但没有明确已经实施。',
  '自伤行为：明确已经发生故意伤害自己的行为。',
  '用户攻击行为：当前用户明确实施、威胁或意图对他人实施攻击。',
  '他人攻击行为：消息中的攻击行为由他人针对当前用户或其他人实施。',
];

const HIERARCHICAL_RULES = [
  '先判断是否确有自杀、自伤或攻击相关证据；一般痛苦、绝望和负面情绪本身不足以定类。',
  '再区分自杀、自伤、用户攻击、他人攻击四个类别族。',
  '自杀类别中区分：探索、被动意图、主动意图、计划、准备行为、未遂。',
  '自伤类别中区分：意图与已发生行为。',
  '以文本里最直接、最明确的主体和行为证据为准，不要只按风险严重程度猜测。',
  '在内部完成层级判断，但最终只输出一个完整类别名称，不输出推理。',
];

export function load(args = {}) {
  const split = args.split;
  if (!splits.includes(split)) {
    throw new Error(`psysuicide loader requires split=${splits.join('|')}`);
  }
  const data = JSON.parse(readFileSync(join(DATASETS, 'PsySUICIDE', 'repo', `${split}.json`), 'utf8'));
  const items = [];
  let multi = 0, unknown = 0;
  for (const r of data) {
    if (!Array.isArray(r.labels) || r.labels.length !== 1) { multi++; continue; }
    const gold = r.labels[0];
    if (!LABELS.includes(gold)) { unknown++; continue; }
    // Released `idx` values are not unique. Prefix the retained-row position so resume
    // and paired audits cannot collapse unrelated counseling texts sharing one idx.
    items.push({ id: `${split}:${items.length}:${r.idx}`, sourceId: String(r.idx), gold, text: r.text });
  }
  if (multi || unknown) {
    console.log(`[psysuicide/${split}] dropped ${multi} multi-label + ${unknown} unknown-label rows`);
  }
  return items;
}

export function assert(items, args = {}) {
  const minimum = { train: 11000, valid: 1400, test: 1400 }[args.split];
  if (!minimum || items.length < minimum) {
    throw new Error(`expected ${args.split} to contain at least ${minimum} single-label rows, got ${items.length}`);
  }
  if (new Set(items.map((item) => item.id)).size !== items.length) throw new Error('PsySUICIDE normalized ids are not unique');
}

export function leakageKey(item) {
  return { text: item.text };
}

function baselineMessages(item) {
  return [
    { role: 'system', content: '你是心理咨询中的自杀风险识别专家。' },
    {
      role: 'user',
      content: `以下是一条心理咨询中的用户消息：\n\n「${item.text}」\n\n请判断这条消息属于以下哪一类（只回答类别名称，不要解释）：\n${LABELS.join('\n')}`,
    },
  ];
}

function taxonomyMessages(item) {
  return [
    {
      role: 'system',
      content: '你负责按固定标注体系分类心理咨询文本。只依据消息中的明确证据判断，不进行临床诊断，也不要补充文本中没有的信息。',
    },
    {
      role: 'user',
      content: `请按下面的工作定义，将用户消息分到唯一一个类别。\n\n工作定义：\n${OPERATIONAL_DEFINITIONS.join('\n')}\n\n用户消息：\n「${item.text}」\n\n只回答一个完整类别名称，不要解释。`,
    },
  ];
}

function hierarchicalMessages(item) {
  return [
    {
      role: 'system',
      content: '你负责按固定层级标签体系分类心理咨询文本。不得推测消息未表达的意图；只输出最终类别名称。',
    },
    {
      role: 'user',
      content: `请在内部按以下层级完成判断：\n${HIERARCHICAL_RULES.map((rule, i) => `${i + 1}. ${rule}`).join('\n')}\n\n类别工作定义：\n${OPERATIONAL_DEFINITIONS.join('\n')}\n\n用户消息：\n「${item.text}」\n\n最终只回答以下类别中的一个完整名称，不要输出步骤或解释：\n${LABELS.join('\n')}`,
    },
  ];
}

export function messages(item, args = {}) {
  const profile = args.promptProfile || defaultPromptProfile;
  if (profile === 'baseline') return baselineMessages(item);
  if (profile === 'taxonomy') return taxonomyMessages(item);
  if (profile === 'hierarchical') return hierarchicalMessages(item);
  throw new Error(`unsupported PsySUICIDE prompt profile: ${profile}`);
}

export function promptFingerprint(args = {}) {
  const profile = args.promptProfile || defaultPromptProfile;
  const fingerprint = {
    prompt_version: promptVersion,
    prompt_profile: profile,
    labels: LABELS,
  };
  if (profile !== 'baseline') fingerprint.operational_definitions = OPERATIONAL_DEFINITIONS;
  if (profile === 'hierarchical') fingerprint.hierarchical_rules = HIERARCHICAL_RULES;
  return fingerprint;
}

export const parse = makeLabelParser(LABELS);

export const comparisons = [
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'accuracy', value: baselineValue('psysuicide-roberta-large-acc') },
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'micro-F1', value: baselineValue('psysuicide-roberta-large-microf1') },
  { method: 'PsyGUARD RoBERTa-large fine-tuned', metric: 'macro-F1', value: baselineValue('psysuicide-roberta-large-macrof1') },
  { method: 'PsyGUARD GPT-4-preview zero-shot', metric: 'accuracy', value: baselineValue('psysuicide-gpt4-preview-acc') },
  { method: 'majority class（与自杀无关）', metric: 'accuracy', value: baselineValue('psysuicide-majority') },
];
