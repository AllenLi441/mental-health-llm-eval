#!/usr/bin/env node
// Offline synthetic regression guard for CPsyExam prompt profiles and parsing.
// This script intentionally does not load the licensed dataset or call an API.
process.env.EVAL_IGNORE_DOTENV = '1';

const lib = await import('../lib.mjs');
const task = await import('../tasks/cpsyexam.mjs');

const single = {
  id: 'fixture-single',
  gold: 'B',
  qtype: 'single',
  kind: 'knowledge',
  subject_name: '发展与教育心理学',
  question: '哪一个选项是合成测试答案？',
  options: { A: '干扰项', B: '正确项', C: '干扰项' },
};
const multiple = {
  ...single,
  id: 'fixture-multiple',
  gold: 'ABD',
  qtype: 'multiple',
};

function expectParse(name, raw, item, expected, invalid = false) {
  const actual = task.parse(raw, item);
  if (actual.predicted !== expected || actual.invalid !== invalid) {
    throw new Error(`${name}: ${JSON.stringify(raw)} -> ${JSON.stringify(actual)}, expected predicted=${JSON.stringify(expected)} invalid=${invalid}`);
  }
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

// Strict JSON supports both documented keys and common string/array encodings.
expectParse('JSON answer single', '{"answer":"B"}', single, 'B');
expectParse('JSON ans single', '{"ans":"b"}', single, 'B');
expectParse('JSON answer multiple array', '{"answer":["D","A","B","A"]}', multiple, 'ABD');
expectParse('JSON answer multiple string', '{"answer":"D, A B"}', multiple, 'ABD');

// Anchors are parsed before letters, so the A in "Answer" is never an answer.
expectParse('English Answer anchor', 'Answer: B', single, 'B');
expectParse('English Final Answer anchor', 'Final Answer: c', single, 'C');
expectParse('Chinese answer anchor', '答案：B', single, 'B');
expectParse('Chinese final answer anchor', '最终答案是 D、A、B', multiple, 'ABD');

// Bare outputs are accepted, with deterministic normalization for multi-choice.
expectParse('bare single', 'b', single, 'B');
expectParse('bare multiple', 'DABAD', multiple, 'ABD');
expectParse('duplicate single letter', 'BBB', single, 'B');

// Ambiguous prose, conflicting choices, and non-strict JSON must be rejected.
expectParse('single conflicting letters', 'Answer: AB', single, null, true);
expectParse('prose with answer', 'I think the answer is B because it fits.', single, null, true);
expectParse('Chinese prose with answer', '我认为答案是B，因为其他选项不对。', single, null, true);
expectParse('JSON extra field', '{"answer":"B","reason":"synthetic"}', single, null, true);
expectParse('JSON wrong key', '{"result":"B"}', single, null, true);
expectParse('JSON conflicting keys', '{"answer":"B","ans":"B"}', single, null, true);
expectParse('Markdown JSON fence', '```json\n{"answer":"B"}\n```', single, null, true);
expectParse('empty', '', single, null, true);

expect(task.defaultPromptProfile === 'legacy-zero-shot-v1', 'legacy profile must remain the default');
expect(
  JSON.stringify(task.messages(single)) === JSON.stringify(task.messages(single, { promptProfile: 'legacy-zero-shot-v1' })),
  'default prompt must equal the explicit legacy profile',
);
const subjectJson = task.messages(single, { promptProfile: 'subject-json-v1' });
expect(subjectJson[1].content.includes(single.subject_name), 'subject JSON profile must include subject_name');
expect(subjectJson[1].content.includes('{"answer":"B"}'), 'subject JSON single profile must request strict JSON');
const subjectJsonMultiple = task.messages(multiple, { promptProfile: 'subject-json-v1' });
expect(subjectJsonMultiple[1].content.includes('{"answer":["A","B","D"]}'), 'subject JSON multiple profile must request a JSON array');

lib.selftestTask(task, lib.parseArgs([]));

console.log('CPsyExam protocol check PASS: 2 prompt profiles; strict JSON/anchored/bare parsing; ambiguous prose rejected; synthetic-only');
