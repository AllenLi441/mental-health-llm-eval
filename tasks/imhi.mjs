// IMHI benchmark (MentaLLaMA, WWW 2024) — 9 mental-health analysis test sets.
// Gold labels are parsed from the gold-informed response prefix in
// test_data/test_complete/*.csv (the label the explanation was conditioned on).
// Metric: weighted F1 (paper's primary) + accuracy.
// Paper Table 2 reference points are attached per subtask.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, csvObjects } from '../lib.mjs';

const DIR = join(DATASETS, 'MentaLLaMA', 'test_data', 'test_complete');

const YESNO = ['yes', 'no'];

const TASKS = [
  {
    name: 'DR', file: 'DR.csv', type: 'binary',
    question: 'Does the poster suffer from depression?',
    decisionGuide: [
      'Judge clinical depression, not ordinary sadness or a single bad day.',
      'Answer yes when the poster shows sustained low mood, loss of interest or pleasure, hopelessness, worthlessness, persistent fatigue, or an explicit statement of being depressed over time.',
      'Answer no for transient sadness, venting about one specific event, ordinary stress, or posts that only discuss depression in other people or in the abstract.',
      'Do not infer depression merely because the post is negative, emotional, or about an unpleasant situation.',
    ].join(' '),
    cmp: { chatgptZS: 82.41, bestDiscriminative: 'RoBERTa 95.11', mentallama13b: 85.68 },
  },
  {
    name: 'dreaddit', file: 'dreaddit.csv', type: 'binary',
    question: 'Does the poster suffer from stress?',
    decisionGuide: [
      'Use the dataset\'s stress-detection meaning, not a broad clinical-safety screen.',
      'Answer yes when the poster expresses meaningful psychological strain, such as sustained overwhelm, helplessness, fear, anxiety, inability to cope, or an explicit statement of being stressed.',
      'Answer no for neutral descriptions, ordinary advice requests, isolated annoyance or disagreement, and descriptions of a difficult event that do not show the poster experiencing psychological strain.',
      'Do not infer stress merely because the topic is unpleasant or could be stressful.',
    ].join(' '),
    cmp: { chatgptZS: 71.79, bestDiscriminative: 'MentalRoBERTa 81.76', mentallama13b: 75.79 },
  },
  {
    name: 'loneliness', file: 'loneliness.csv', type: 'binary',
    question: 'Does the poster suffer from loneliness?',
    decisionGuide: [
      'Judge subjective loneliness — distress from lacking meaningful connection — not merely being alone.',
      'Answer yes when the poster expresses feeling isolated, disconnected, unseen, left out, or having no one to turn to.',
      'Answer no when the poster is simply by themselves, mentions relationships or social activity without distress, or describes other problems without expressed loneliness.',
      'Do not infer loneliness merely because the poster is physically alone or mentions other people.',
    ].join(' '),
    cmp: { chatgptZS: 58.40, bestDiscriminative: 'MentalRoBERTa 85.33', mentallama13b: 85.1 },
  },
  {
    name: 'Irf', file: 'Irf.csv', type: 'binary', questionFromRow: true,
    decisionGuide: [
      'The question names one interpersonal risk factor from the interpersonal theory of suicide.',
      'Thwarted belongingness = an unmet need to belong: feeling disconnected, excluded, alienated, or without reciprocal caring relationships.',
      'Perceived burdensomeness = the belief that one is a burden and that others would be better off without them, often with self-hatred.',
      'Answer yes only when the post expresses the specific factor named in the question; answer no when it is absent or only another kind of distress is shown.',
      'Do not infer the factor merely from general negativity, sadness, or the mention of relationships or self-worth.',
    ].join(' '),
    cmp: { chatgptZS: 41.33, bestDiscriminative: 'MentalBERT 76.73', mentallama13b: 76.49 },
  },
  {
    name: 'MultiWD', file: 'MultiWD.csv', type: 'binary', questionFromRow: true,
    decisionGuide: [
      'Decide whether the specific wellness dimension named in the question is substantively present in the post.',
      'Physical = bodily health, illness, sleep, nutrition, exercise, or physical self-care.',
      'Emotional = awareness, expression, acceptance, or regulation of feelings.',
      'Social = relationships, support, belonging, isolation, or social interaction.',
      'Vocational = work, career, education-to-career direction, or occupational purpose.',
      'Spiritual = meaning, purpose, values, faith, or one\'s place in life.',
      'Intellectual = learning, curiosity, creativity, problem solving, or cultural and intellectual activity.',
      'Answer yes for explicit or clearly demonstrated evidence, including impairment or absence in that dimension; answer no when the dimension is only remotely implied or when evidence belongs to another dimension.',
    ].join(' '),
    cmp: { chatgptZS: 62.72, bestDiscriminative: 'BERT 76.69', mentallama13b: 75.11 },
  },
  {
    name: 'SAD', file: 'SAD.csv', type: 'multi',
    question: 'What is the cause of the poster\'s stress?',
    labels: ['school', 'financial problem', 'family issues', 'social relationships', 'work',
      'health issues', 'emotional turmoil', 'everyday decision making', 'other causes'],
    cmp: { chatgptZS: 54.05, bestDiscriminative: 'MentalRoBERTa 68.44', mentallama13b: 63.62 },
  },
  {
    name: 'CAMS', file: 'CAMS.csv', type: 'multi',
    question: 'What is the cause of the poster\'s mental disorder? (answer "none" if no cause is shown)',
    labels: ['bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation', 'none'],
    cmp: { chatgptZS: 33.85, bestDiscriminative: 'MentalRoBERTa 47.62', mentallama13b: 45.52 },
  },
  {
    name: 'swmh', file: 'swmh.csv', type: 'multi',
    question: 'Which mental disorder symptoms does this post show?',
    labels: ['depression', 'suicide', 'anxiety', 'bipolar disorder', 'no mental disorders'],
    cmp: { chatgptZS: 49.32, bestDiscriminative: 'MentalRoBERTa 72.16', mentallama13b: 71.7 },
  },
  {
    name: 't-sid', file: 't-sid.csv', type: 'multi',
    question: 'Which mental disorder symptoms does this post show?',
    labels: ['depression', 'suicide or self-harm tendency', 'ptsd', 'no mental disorders'],
    cmp: { chatgptZS: 33.30, bestDiscriminative: 'MentalRoBERTa 89.01', mentallama13b: 75.31 },
  },
];

function extractPost(query) {
  const m = query.match(/(?:consider this post:|post:)\s*"?([\s\S]*?)"?\s*question:/i);
  if (m) return m[1].trim();
  const qi = query.toLowerCase().lastIndexOf('question:');
  return (qi > 0 ? query.slice(0, qi) : query).trim();
}

function extractQuestion(query) {
  const m = query.match(/the answer to the question:?\s*"+(.+?)"+\s*is\s*$/is);
  return m ? m[1].trim() : null;
}

function goldFrom(resp, t) {
  let head = String(resp || '').split(/reasoning/i)[0].trim()
    .replace(/^["'\s]+|["'\s.:]+$/g, '').toLowerCase();
  if (t.type === 'binary') {
    if (head.startsWith('yes')) return 'yes';
    if (head.startsWith('no')) return 'no';
    return null;
  }
  for (const lab of t.labels) {
    if (head === lab || head.startsWith(lab)) return lab;
  }
  return null;
}

function makeVariant(t) {
  const labels = t.type === 'binary' ? YESNO : t.labels;
  return {
    key: `imhi-${t.name.toLowerCase()}`,
    description: `IMHI ${t.name} (${t.type === 'binary' ? 'binary' : `${labels.length}-class`}, zero-shot)`,
    labels,
    defaultSample: 500,
    maxTokens: t.type === 'binary' ? 4 : 16,
    load() {
      const rows = csvObjects(readFileSync(join(DIR, t.file), 'utf8'));
      const items = [];
      let dropped = 0;
      for (let i = 0; i < rows.length; i++) {
        const query = rows[i].query;
        const gold = goldFrom(rows[i]['gpt-3.5-turbo'], t);
        if (!gold) { dropped++; continue; }
        const post = extractPost(query);
        if (!post) { dropped++; continue; }
        const q = t.questionFromRow ? extractQuestion(query) : t.question;
        if (!q) { dropped++; continue; }
        items.push({ id: `${t.name}-${i}`, gold, post, question: q });
      }
      if (dropped) console.log(`[imhi-${t.name}] dropped ${dropped}/${rows.length} rows (unparseable gold/post/question)`);
      if (dropped / rows.length > 0.05) throw new Error(`imhi-${t.name}: too many dropped rows`);
      return items;
    },
    messages(item) {
      const guide = t.decisionGuide ? `\nDecision criteria: ${t.decisionGuide}` : '';
      const inst = t.type === 'binary'
        ? `${item.question}${guide}\nAnswer with only "yes" or "no".`
        : `${item.question} Answer with exactly one of: ${labels.join('; ')}. No explanation.`;
      return [
        { role: 'system', content: 'You are an expert in mental health analysis of social media posts.' },
        { role: 'user', content: `Post: "${item.post}"\n\n${inst}` },
      ];
    },
    parse(raw) {
      const s = (raw || '').trim().toLowerCase();
      if (!s) return { predicted: null, invalid: true };
      if (t.type === 'binary') {
        if (/^yes/.test(s)) return { predicted: 'yes', invalid: false };
        if (/^no\b/.test(s)) return { predicted: 'no', invalid: false };
        if (s.includes('yes') !== s.includes('no')) return { predicted: s.includes('yes') ? 'yes' : 'no', invalid: false };
        return { predicted: null, invalid: true };
      }
      const cleaned = s.replace(/^[^a-z]+|[^a-z]+$/g, '');
      for (const lab of labels) if (cleaned === lab) return { predicted: lab, invalid: false };
      const hits = labels.filter((lab) => s.includes(lab));
      if (hits.length === 1) return { predicted: hits[0], invalid: false };
      // prefix leniency: unique label starting with the reply
      const pre = labels.filter((lab) => cleaned.length >= 4 && lab.startsWith(cleaned));
      if (pre.length === 1) return { predicted: pre[0], invalid: false };
      return { predicted: null, invalid: true };
    },
    comparisons: [
      { method: 'ChatGPT zero-shot (paper Table 2)', metric: 'weighted F1', value: t.cmp.chatgptZS },
      { method: `best fine-tuned discriminative (${t.cmp.bestDiscriminative.split(' ')[0]})`, metric: 'weighted F1', value: Number(t.cmp.bestDiscriminative.split(' ')[1]) },
      { method: 'MentaLLaMA-chat-13B (paper)', metric: 'weighted F1', value: t.cmp.mentallama13b },
    ],
  };
}

export const variants = TASKS.map(makeVariant);
