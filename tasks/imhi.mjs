// IMHI benchmark (MentaLLaMA, WWW 2024) — 9 mental-health analysis test sets.
// Gold labels are parsed from the gold-informed response prefix in
// test_data/test_complete/*.csv (the label the explanation was conditioned on).
// Metric: weighted F1 (paper's primary) + accuracy.
// Paper Table 2 reference points are attached per subtask.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATASETS, csvObjects } from '../lib.mjs';
import { imhiBaseline } from '../lib/baselines.mjs';

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
  },
  {
    name: 'SAD', file: 'SAD.csv', type: 'multi',
    question: 'What is the cause of the poster\'s stress?',
    labels: ['school', 'financial problem', 'family issues', 'social relationships', 'work',
      'health issues', 'emotional turmoil', 'everyday decision making', 'other causes'],
    candidateGuide: 'school = academic demands or study; financial problem = money, debt, bills, or basic costs; family issues = relatives or household conflict; social relationships = friends, partners, rejection, or interpersonal conflict outside the family; work = job or workplace demands; health issues = physical or mental health; emotional turmoil = internal emotional distress without a more direct listed cause; everyday decision making = ordinary choices or daily organization; other causes = only when none of the specific causes is directly supported.',
  },
  {
    name: 'CAMS', file: 'CAMS.csv', type: 'multi',
    question: 'What is the cause of the poster\'s mental disorder? (answer "none" if no cause is shown)',
    labels: ['bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation', 'none'],
    candidateGuide: 'bias or abuse = discrimination, harassment, violence, or maltreatment; jobs and career = employment, workplace, or career pressure; medication = psychiatric or other medicine effects, withdrawal, or treatment; relationship = conflict, loss, or difficulty with a partner, family member, or friend; alienation = isolation, exclusion, or disconnection; none = the post does not state a cause. Prefer the most directly stated cause and do not infer one from symptoms alone.',
  },
  {
    name: 'swmh', file: 'swmh.csv', type: 'multi',
    question: 'Which mental disorder symptoms does this post show?',
    labels: ['depression', 'suicide', 'anxiety', 'bipolar disorder', 'no mental disorders'],
    candidateGuide: 'depression = sustained depressed mood, anhedonia, hopelessness, or related depressive symptoms; suicide = suicidal thought, plan, attempt, or self-harm tendency; anxiety = persistent fear, worry, panic, or physiological anxiety; bipolar disorder = explicit mania/hypomania or bipolar diagnosis, not ordinary mood change; no mental disorders = no direct evidence for the listed symptom groups. Use only evidence in the post.',
  },
  {
    name: 't-sid', file: 't-sid.csv', type: 'multi',
    question: 'Which mental disorder symptoms does this post show?',
    labels: ['depression', 'suicide or self-harm tendency', 'ptsd', 'no mental disorders'],
    candidateGuide: 'depression = sustained depressed mood, anhedonia, hopelessness, or related depressive symptoms; suicide or self-harm tendency = suicidal or deliberate self-injury thought, intent, plan, attempt, or behavior; ptsd = trauma-linked intrusion, avoidance, hyperarousal, or explicit PTSD; no mental disorders = no direct evidence for the listed groups. Do not infer a diagnosis from a negative event alone.',
  },
];

const EVIDENCE_PROTOCOL_V4 = [
  'Identify the exact phrase or behavior that supports each plausible label.',
  'Actively test the closest alternative label and reject labels supported only by topic, negativity, or speculation.',
  'For binary questions, answer yes only when the named construct is substantively present; otherwise answer no.',
  'For multiclass questions, choose the single most directly supported label and use the fallback label only when no specific label is evidenced.',
  'Perform this comparison internally, then output only the exact allowed label.',
].join(' ');

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
  const published = imhiBaseline(t.name);
  return {
    key: `imhi-${t.name.toLowerCase()}`,
    description: `IMHI ${t.name} (${t.type === 'binary' ? 'binary' : `${labels.length}-class`}, zero-shot)`,
    labels,
    defaultSample: 500,
    maxTokens: t.type === 'binary' ? 4 : 16,
    promptProfiles: ['uniform-v3u', 'contrastive-v4'],
    defaultPromptProfile: 'uniform-v3u',
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
    messages(item, args = {}) {
      const profile = args.promptProfile || 'uniform-v3u';
      const guide = t.decisionGuide ? `\nDecision criteria: ${t.decisionGuide}` : '';
      if (profile === 'contrastive-v4') {
        const candidateGuide = t.decisionGuide || t.candidateGuide;
        const inst = t.type === 'binary'
          ? `${item.question}\nConstruct definition: ${candidateGuide}\nUnified evidence protocol: ${EVIDENCE_PROTOCOL_V4}\nAnswer with only "yes" or "no".`
          : `${item.question}\nLabel boundaries: ${candidateGuide}\nUnified evidence protocol: ${EVIDENCE_PROTOCOL_V4}\nAnswer with exactly one of: ${labels.join('; ')}. No explanation.`;
        return [
          { role: 'system', content: 'You are a benchmark labeler. Apply the same evidence-first contrastive decision process to every IMHI task. Do not diagnose or add facts absent from the post.' },
          { role: 'user', content: `Post: "${item.post}"\n\n${inst}` },
        ];
      }
      if (profile !== 'uniform-v3u') throw new Error(`unsupported IMHI prompt profile: ${profile}`);
      const inst = t.type === 'binary'
        ? `${item.question}${guide}\nAnswer with only "yes" or "no".`
        : `${item.question} Answer with exactly one of: ${labels.join('; ')}. No explanation.`;
      return [
        { role: 'system', content: 'You are an expert in mental health analysis of social media posts.' },
        { role: 'user', content: `Post: "${item.post}"\n\n${inst}` },
      ];
    },
    promptFingerprint(args = {}) {
      const profile = args.promptProfile || 'uniform-v3u';
      return {
        prompt_version: 'imhi-uniform-protocols-v1',
        prompt_profile: profile,
        subtask: t.name,
        labels,
        decision_guide: profile === 'contrastive-v4'
          ? (t.decisionGuide || t.candidateGuide)
          : (t.decisionGuide || null),
        unified_evidence_protocol: profile === 'contrastive-v4'
          ? EVIDENCE_PROTOCOL_V4
          : null,
      };
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
      { method: 'ChatGPT zero-shot (paper Table 2)', metric: 'weighted F1', value: published.chatgpt_zs },
      { method: `best fine-tuned discriminative (${published.best_finetuned.name})`, metric: 'weighted F1', value: published.best_finetuned.value },
      { method: 'MentaLLaMA-chat-13B (paper)', metric: 'weighted F1', value: published.mentallama13b },
    ],
  };
}

export const variants = TASKS.map(makeVariant);
