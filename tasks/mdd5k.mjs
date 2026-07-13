// MDD-5k (AAAI 2025) — Chinese psychiatric diagnostic conversations (synthesized
// from 1,000 real Shanghai Mental Health Center cases). Task defined by this
// module (the dataset is new; no standard leaderboard yet): predict the coarse
// diagnostic category from the conversation. Gold derived from the ICD code:
//   F31->双相障碍  F32/F33->抑郁障碍  F41.2->焦虑抑郁混合  F40/F41->焦虑障碍  else->其他精神障碍
// One conversation per patient (the first of five), truncated to 6000 chars.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, makeLabelParser } from '../lib.mjs';

const CLASSES = ['抑郁障碍', '焦虑障碍', '焦虑抑郁混合', '双相障碍', '其他精神障碍'];
const MAXCHARS = 6000;

function coarseClass(icd) {
  const code = String(icd || '').toUpperCase();
  if (code.startsWith('F31')) return '双相障碍';
  if (code.startsWith('F32') || code.startsWith('F33')) return '抑郁障碍';
  if (code.startsWith('F41.2')) return '焦虑抑郁混合';
  if (code.startsWith('F40') || code.startsWith('F41')) return '焦虑障碍';
  return '其他精神障碍';
}

function convText(entry) {
  const c = entry?.conversation ?? entry;
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) {
    return c.map((t) => {
      if (typeof t === 'string') return t;
      const who = t.role || t.speaker || '';
      return `${who}${who ? ': ' : ''}${t.content || t.text || JSON.stringify(t)}`;
    }).join('\n');
  }
  if (c && typeof c === 'object') {
    return Object.entries(c).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');
  }
  return '';
}

export const key = 'mdd5k-diagnosis';
export const description = 'MDD-5k coarse diagnosis classification (5-class, Chinese, zero-shot; protocol defined by this module)';
export const labels = CLASSES;
export const defaultSample = 400;
export const maxTokens = 16;

export function load() {
  const repo = join(DATASETS, 'MDD-5k', 'repo');
  const labelDir = join(repo, 'Label');
  const convDir = join(repo, 'MDD_5k');
  const items = [];
  for (const f of readdirSync(labelDir).sort()) {
    if (!f.endsWith('_label.json')) continue;
    const pid = f.replace('_label.json', '');
    const lab = JSON.parse(readFileSync(join(labelDir, f), 'utf8'));
    let convs;
    try {
      convs = JSON.parse(readFileSync(join(convDir, `${pid}.json`), 'utf8'));
    } catch { continue; } // label without conversation file
    const text = convText(Array.isArray(convs) ? convs[0] : convs).slice(0, MAXCHARS);
    if (!text) continue;
    items.push({ id: pid, gold: coarseClass(lab.ICD_Code), icd: lab.ICD_Code, rawDiagnosis: lab.Diagnosis_Result, text });
  }
  return items;
}

export function assert(items) {
  if (items.length < 800) throw new Error(`expected ~926 patients, got ${items.length}`);
  const set = new Set(CLASSES);
  for (const it of items) if (!set.has(it.gold)) throw new Error(`bad class ${it.gold}`);
}

export function messages(item) {
  return [
    { role: 'system', content: '你是精神科诊断助手。' },
    {
      role: 'user',
      content: `以下是一段精神科门诊的医患对话（可能被截断）：\n\n${item.text}\n\n根据对话内容，患者最可能的诊断类别是以下哪一类？只回答类别名称，不要解释：\n${CLASSES.join('、')}`,
    },
  ];
}

export const parse = makeLabelParser(CLASSES);

export const comparisons = [
  { method: '协议说明', metric: 'note', value: '数据集 2025 年发布、无统一诊断分类榜单；本 5 类协议由本模块定义（由 ICD 编码归并），数字用于横向比较模型，不与论文直接对表' },
];
