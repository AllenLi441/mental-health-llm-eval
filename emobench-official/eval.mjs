#!/usr/bin/env node

import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = dirname(SCRIPT_DIR);
const DATA_DIR = join(ROOT_DIR, "repo", "data");
const RESULTS_DIR = join(SCRIPT_DIR, "results");

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const TASKS = ["EA", "EU"];
const LANGS = ["en", "zh"];

const SYS = {
  en: `# Instructions

In this task, you are presented with a scenario, a question, and multiple choices. 
Carefully analyze the scenario and take the perspective of the individual involved.
Then, select the option that best reflects their perspective or emotional response.

# Output`,
  zh: `# 说明
在这个任务中，你会面临一个场景、一个问题和多个选项。
仔细分析场景，并从相关人员的角度进行思考。
然后，选择最能反映他们的观点或情绪反应的选项。

# 输出`,
};

const RESPONSE_BASE = {
  en: "Provide only one single correct answer to this question. Do not provide any additional information or explanations. The response should be in the following JSON format:",
  zh: "只提供一个正确的答案。 不要提供任何额外的信息或解释。 回复应采用以下JSON格式：",
};

const RESPONSE_CONDITIONS = {
  EA: {
    en: `"answer": "<Respond with the corresponding letter numbering>"`,
    zh: `"answer": "<用相应的字母进行回答>"`,
  },
  EU: {
    en: `"answer_q1": "<Respond to the Question 1 with the corresponding letter numbering>",
"answer_q2": "<Respond to the Question 2 with the corresponding letter numbering>"`,
    zh: `"answer_q1": "<用相应的字母进行回答问题1>",
"answer_q2": "<用相应的字母进行回答问题2>"`,
  },
};

function parseArgs(argv) {
  const args = {
    base: process.env.DEEPSEEK_BASE_URL || process.env.BASE_URL || "https://api.deepseek.com",
    concurrency: 10,
    lang: "all",
    model: "deepseek-chat",
    selftest: false,
    task: "all",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--selftest") {
      args.selftest = true;
    } else if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const value = argv[i + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`Missing value for ${arg}`);
      }
      i += 1;
      if (key === "concurrency") {
        args.concurrency = Number(value);
      } else if (key in args) {
        args[key] = value;
      } else {
        throw new Error(`Unknown option: ${arg}`);
      }
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!["EA", "EU", "all"].includes(args.task)) {
    throw new Error("--task must be EA, EU, or all");
  }
  if (!["en", "zh", "all"].includes(args.lang)) {
    throw new Error("--lang must be en, zh, or all");
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1) {
    throw new Error("--concurrency must be a positive integer");
  }

  return args;
}

async function readJsonl(filePath) {
  const text = await readFile(filePath, "utf8");
  return text
    .split(/\r?\n/u)
    .filter((line) => line.trim() !== "")
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${filePath}:${index + 1}: ${error.message}`);
      }
    });
}

async function loadTask(task) {
  return readJsonl(join(DATA_DIR, `${task}.jsonl`));
}

function selectedTasks(task) {
  return task === "all" ? TASKS : [task];
}

function selectedLangs(lang) {
  return lang === "all" ? LANGS : [lang];
}

function rankChoices(choices) {
  return choices.map((choice, index) => `${LETTERS[index]}) ${choice}`).join("\n");
}

function responseFormat(task, lang) {
  return `\n${RESPONSE_BASE[lang]}\n\`\`\`json\n    {\n    ${RESPONSE_CONDITIONS[task][lang]}\n    }\n\`\`\``;
}

function systemPrompt(task, lang) {
  return `${SYS[lang]}\n${responseFormat(task, lang)}`;
}

function renderUserPrompt(task, sample) {
  if (task === "EA") {
    const choices = rankChoices(sample.choices);
    if (sample.language === "en") {
      return `## Scenario
${sample.scenario}

## Question 
In this scenario, what is the most effective ${sample["question type"]} for ${sample.subject}?

## Choices
${choices}`;
    }

    return `## 场景
${sample.scenario}

## 问题
在这个场景中，${sample.subject}最有效的${sample["question type"]}是什么？

## 选项
${choices}`;
  }

  const emoChoices = rankChoices(sample.emotion_choices);
  const causeChoices = rankChoices(sample.cause_choices);
  if (sample.language === "en") {
    return `## Scenario
${sample.scenario}

## Question 1
What emotion(s) would ${sample.subject} ultimately feel in this situation?

## Choices for Question 1
${emoChoices}

## Question 2
Why would ${sample.subject} feel these emotions in this situation?

## Choices for Question 2
${causeChoices}`;
  }

  return `## 场景
${sample.scenario}

## 问题 1
在这种情况下，${sample.subject}最终会感受到什么情绪？

## 问题 1 的选项
${emoChoices}

## 问题 2
${sample.subject}为什么会在这种情况下感受到这些情绪？

## 问题 2 的选项
${causeChoices}`;
}

function renderMessages(task, sample) {
  return [
    { role: "system", content: systemPrompt(task, sample.language) },
    { role: "user", content: renderUserPrompt(task, sample) },
  ];
}

function goldLetter(choices, label) {
  const index = choices.indexOf(label);
  if (index === -1) {
    throw new Error(`Label not found in choices: ${label}`);
  }
  return LETTERS[index];
}

function stripJsonFence(content) {
  const text = String(content ?? "").trim();
  const match = text.match(/```json\s*([\s\S]*?)```/iu);
  if (match) {
    return match[1].trim();
  }
  return text;
}

function parseJsonObject(content) {
  try {
    return JSON.parse(stripJsonFence(content));
  } catch {
    return undefined;
  }
}

function extractLetter(value, choices) {
  if (value === undefined || value === null) {
    return "";
  }

  const raw = String(value).trim();
  const letterMatch = raw.match(/(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)/u);
  if (letterMatch) {
    return letterMatch[1];
  }

  const choiceIndex = choices.findIndex((choice) => raw === choice);
  return choiceIndex === -1 ? "" : LETTERS[choiceIndex];
}

function parsePrediction(task, content, sample) {
  const obj = parseJsonObject(content);
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
    return task === "EA"
      ? { answer: "", invalid: true }
      : { answer_q1: "", answer_q2: "", invalid: true };
  }

  if (task === "EA") {
    const answer = extractLetter(obj.answer, sample.choices);
    return { answer, invalid: answer === "" };
  }

  const answerQ1 = extractLetter(obj.answer_q1, sample.emotion_choices);
  const answerQ2 = extractLetter(obj.answer_q2, sample.cause_choices);
  return {
    answer_q1: answerQ1,
    answer_q2: answerQ2,
    invalid: answerQ1 === "" || answerQ2 === "",
  };
}

function resultForSample(task, sample, content) {
  const pred = parsePrediction(task, content, sample);
  if (task === "EA") {
    const label = goldLetter(sample.choices, sample.label);
    return {
      qid: sample.qid,
      lang: sample.language,
      category: sample.category,
      label,
      answer: pred.answer,
      correct: pred.answer === label,
      invalid: pred.invalid,
      raw_response: content,
    };
  }

  const emoLabel = goldLetter(sample.emotion_choices, sample.emotion_label);
  const causeLabel = goldLetter(sample.cause_choices, sample.cause_label);
  return {
    qid: sample.qid,
    lang: sample.language,
    coarse_category: sample.coarse_category,
    finegrained_category: sample.finegrained_category,
    emo_label: emoLabel,
    emo_answer: pred.answer_q1,
    cause_label: causeLabel,
    cause_answer: pred.answer_q2,
    correct: pred.answer_q1 === emoLabel && pred.answer_q2 === causeLabel,
    invalid: pred.invalid,
    raw_response: content,
  };
}

function summarize(task, rows) {
  const summary = {
    task,
    n: rows.length,
    invalid: rows.filter((row) => row.invalid).length,
    combined: {},
    by_language: {},
  };

  summary.combined = summarizeRows(task, rows);
  for (const lang of LANGS) {
    const langRows = rows.filter((row) => row.lang === lang);
    if (langRows.length > 0) {
      summary.by_language[lang] = summarizeRows(task, langRows);
    }
  }
  return summary;
}

function summarizeRows(task, rows) {
  const categoryKey = task === "EA" ? "category" : "coarse_category";
  const categories = {};
  for (const row of rows) {
    const key = row[categoryKey];
    if (!categories[key]) {
      categories[key] = { correct: 0, total: 0, accuracy: 0 };
    }
    categories[key].total += 1;
    if (row.correct) {
      categories[key].correct += 1;
    }
  }

  for (const value of Object.values(categories)) {
    value.accuracy = value.total === 0 ? 0 : value.correct / value.total;
  }

  const correct = rows.filter((row) => row.correct).length;
  return {
    correct,
    total: rows.length,
    accuracy: rows.length === 0 ? 0 : correct / rows.length,
    categories,
  };
}

function resultPaths(model, task) {
  return {
    jsonl: join(RESULTS_DIR, `${model}-${task}.jsonl`),
    summary: join(RESULTS_DIR, `${model}-${task}.summary.json`),
  };
}

async function loadExistingResults(filePath, langSet) {
  try {
    const rows = await readJsonl(filePath);
    const wantedRows = rows.filter((row) => langSet.has(row.lang));
    const seen = new Set(wantedRows.map((row) => `${row.lang}:${row.qid}`));
    return { rows: wantedRows, seen };
  } catch (error) {
    if (error.code === "ENOENT") {
      return { rows: [], seen: new Set() };
    }
    throw error;
  }
}

async function callChatCompletions(args, messages) {
  const url = `${args.base.replace(/\/+$/u, "")}/chat/completions`;
  const maxTokens = args.model.includes("reasoner") ? 2048 : 50;
  const body = {
    model: args.model,
    messages,
    temperature: 0,
    max_tokens: maxTokens,
  };

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      const text = await response.text();
      if (!response.ok) {
        if ((response.status === 429 || response.status >= 500) && attempt < 3) {
          await sleep(500 * attempt);
          continue;
        }
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const parsed = JSON.parse(text);
      return { content: parsed.choices?.[0]?.message?.content ?? "", apiModel: parsed.model ?? null, fingerprint: parsed.system_fingerprint ?? null };
    } catch (error) {
      if (attempt < 3 && (error.name === "AbortError" || /fetch failed/u.test(error.message))) {
        await sleep(500 * attempt);
        continue;
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  throw new Error("Unreachable retry state");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runWithConcurrency(items, concurrency, worker) {
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      await worker(items[index], index);
    }
  });
  await Promise.all(workers);
}

async function runEval(args) {
  if (!process.env.DEEPSEEK_API_KEY) {
    throw new Error("DEEPSEEK_API_KEY is required for API evaluation");
  }

  await mkdir(RESULTS_DIR, { recursive: true });
  const langSet = new Set(selectedLangs(args.lang));
  const finalSummaries = {};

  for (const task of selectedTasks(args.task)) {
    const paths = resultPaths(args.model, task);
    const { rows, seen } = await loadExistingResults(paths.jsonl, langSet);
    const data = (await loadTask(task)).filter((sample) => langSet.has(sample.language));
    const pending = data.filter((sample) => !seen.has(`${sample.language}:${sample.qid}`));

    console.log(`> ${task}: loaded ${data.length}, existing ${seen.size}, pending ${pending.length}`);
    await runWithConcurrency(pending, args.concurrency, async (sample) => {
      const messages = renderMessages(task, sample);
      const resp = await callChatCompletions(args, messages);
      const row = resultForSample(task, sample, resp.content);
      row.api_model = resp.apiModel; row.fingerprint = resp.fingerprint;
      rows.push(row);
      await appendFile(paths.jsonl, `${JSON.stringify(row)}\n`, "utf8");
      console.log(`> ${task} ${sample.language} qid=${sample.qid} ${row.correct ? "correct" : "wrong"}`);
    });

    rows.sort((a, b) => a.lang.localeCompare(b.lang) || Number(a.qid) - Number(b.qid));
    const summary = summarize(task, rows);
    finalSummaries[task] = summary;
    await writeFile(paths.summary, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  }

  printFinalLine(finalSummaries);
  console.log("Official Base comparators: temp-0 single-run here is a deterministic proxy; the paper used Base 5-sample majority@0.6.");
}

function pct(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(2) : "NA";
}

function printFinalLine(summaries) {
  const parts = [];
  for (const task of TASKS) {
    if (summaries[task]) {
      const en = summaries[task].by_language.en?.accuracy;
      const zh = summaries[task].by_language.zh?.accuracy;
      parts.push(`${task} en=${pct(en)} zh=${pct(zh)}`);
    }
  }
  console.log(parts.join(" | "));
}

function countLangs(rows) {
  return {
    total: rows.length,
    en: rows.filter((row) => row.language === "en").length,
    zh: rows.filter((row) => row.language === "zh").length,
  };
}

function assertTaskCounts(task, rows) {
  const counts = countLangs(rows);
  if (counts.total !== 400 || counts.en !== 200 || counts.zh !== 200) {
    throw new Error(`${task} count mismatch: total=${counts.total} en=${counts.en} zh=${counts.zh}`);
  }
  return counts;
}

function formatPromptForPrint(task, sample) {
  const messages = renderMessages(task, sample);
  return [
    `===== ${task} ${sample.language} SYSTEM =====`,
    messages[0].content,
    `===== ${task} ${sample.language} USER =====`,
    messages[1].content,
  ].join("\n");
}

async function selftest() {
  const ea = await loadTask("EA");
  const eu = await loadTask("EU");
  const eaCounts = assertTaskCounts("EA", ea);
  const euCounts = assertTaskCounts("EU", eu);

  console.log(`SELFTEST counts: EA total=${eaCounts.total} en=${eaCounts.en} zh=${eaCounts.zh}; EU total=${euCounts.total} en=${euCounts.en} zh=${euCounts.zh}`);
  console.log("");
  console.log(formatPromptForPrint("EA", ea[0]));
  console.log("");
  console.log(formatPromptForPrint("EU", eu[0]));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selftest) {
    await selftest();
    return;
  }
  await runEval(args);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
