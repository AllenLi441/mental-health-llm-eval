# EmoBench eval harness — build spec (faithful reproduction of the official Base/zero-shot protocol)

Goal: measure 静室's underlying model (DeepSeek) on EmoBench (ACL 2024) and compare to the
paper's Table 1/2 Base numbers. Reproduce the OFFICIAL protocol exactly, but as a light
standalone script (NO langchain/torch — this machine can't take the official heavy stack).

## Output
Write `eval.mjs` (Node, stdlib only, zero deps — same style as
`~/Desktop/静室/datasets/ed-emotion-eval/eval.mjs`). Node >= 18 (global fetch).

## Data (verbatim schemas — do NOT re-derive)
- `~/Desktop/静室/datasets/EmoBench/repo/data/EA.jsonl` (400 lines; 200 en + 200 zh)
  fields: qid, language ("en"|"zh"), category, `question type` ("Action"|"Response"),
  scenario, subject, choices (list[str]), label (str, one of choices).
- `~/Desktop/静室/datasets/EmoBench/repo/data/EU.jsonl` (400 lines; 200 en + 200 zh)
  fields: qid, language, coarse_category, finegrained_category, scenario, subject,
  emotion_choices (list[str]), emotion_label (str∈emotion_choices),
  cause_choices (list[str]), cause_label (str∈cause_choices).

## Choice lettering
Letters = A,B,C,... by index. `rankChoices(choices)` → "A) <c0>\nB) <c1>\n...".
gold letter = the uppercase letter at the index of the label within its choices list.

## Prompts (VERBATIM from repo configs — copy exactly, do not paraphrase)
System prompt = `sys[lang]` + `\n` + responseFormat(task,lang). No CoT.

sys.en:
```
# Instructions

In this task, you are presented with a scenario, a question, and multiple choices. 
Carefully analyze the scenario and take the perspective of the individual involved.
Then, select the option that best reflects their perspective or emotional response.

# Output
```
sys.zh:
```
# 说明
在这个任务中，你会面临一个场景、一个问题和多个选项。
仔细分析场景，并从相关人员的角度进行思考。
然后，选择最能反映他们的观点或情绪反应的选项。

# 输出
```

responseFormat = `\n{statement}\n` + "```json\n    {{\n    {conditions}\n    }}\n```" (the
official `res_format` in utils.get_response_format). statement = base[lang]; conditions per task.

base.en: "Provide only one single correct answer to this question. Do not provide any additional information or explanations. The response should be in the following JSON format:"
base.zh: "只提供一个正确的答案。 不要提供任何额外的信息或解释。 回复应采用以下JSON格式："

conditions EA.en: `"answer": "<Respond with the corresponding letter numbering>"`
conditions EA.zh: `"answer": "<用相应的字母进行回答>"`
conditions EU.en: `"answer_q1": "<Respond to the Question 1 with the corresponding letter numbering>",\n"answer_q2": "<Respond to the Question 2 with the corresponding letter numbering>"`
conditions EU.zh: `"answer_q1": "<用相应的字母进行回答问题1>",\n"answer_q2": "<用相应的字母进行回答问题2>"`

User message:
EA.en:
```
## Scenario
{scenario}

## Question 
In this scenario, what is the most effective {q_type} for {subject}?

## Choices
{choices}
```
EA.zh:
```
## 场景
{scenario}

## 问题
在这个场景中，{subject}最有效的{q_type}是什么？

## 选项
{choices}
```
(q_type = the record's `question type` value: "Action"/"Response". Keep English value even in zh prompt — that's what the official code does: it passes sample["question type"] directly.)

EU.en:
```
## Scenario
{scenario}

## Question 1
What emotion(s) would {subject} ultimately feel in this situation?

## Choices for Question 1
{emo_choices}

## Question 2
Why would {subject} feel these emotions in this situation?

## Choices for Question 2
{cause_choices}
```
EU.zh:
```
## 场景
{scenario}

## 问题 1
在这种情况下，{subject}最终会感受到什么情绪？

## 问题 1 的选项
{emo_choices}

## 问题 2
{subject}为什么会在这种情况下感受到这些情绪？

## 问题 2 的选项
{cause_choices}
```

## Parsing the model reply (mirror utils.parse_json_response, then a robust letter fallback)
1. Strip ```json fences if present; JSON.parse.
2. EA: pred = obj.answer ; EU: predEmo = obj.answer_q1, predCause = obj.answer_q2.
3. Each pred → extract the FIRST standalone A–Z letter (e.g. "B", "B)", "B) text", "Answer: B").
   If no letter but the value exactly matches a choice string, map that choice→its letter.
   If still unresolved → mark invalid (counts as wrong).

## Scoring (EXACT match, mirror data.py.evaluate_results)
- EA: correct = (predLetter == goldLetter). Group accuracy by `category` + Overall.
- EU: correct = (predEmoLetter == emoGoldLetter) AND (predCauseLetter == causeGoldLetter).
  Group by `coarse_category` + Overall. BOTH sub-answers must be right.
- Report accuracy per language separately (en, zh) AND a combined, since the paper reports en/zh.

## API
OpenAI-compatible `${BASE}/chat/completions`. BASE default `https://api.deepseek.com`.
Key from env DEEPSEEK_API_KEY (export via `set -a; . ~/Desktop/静室/app/.env.local; set +a`).
messages = [ {role:"system", content: sysPrompt}, {role:"user", content: userMsg} ].
temperature 0 (deterministic; note in output that official used Base 5-sample majority@0.6 —
temp-0 single is the clean deterministic proxy). max_tokens: 30 for chat, 2048 for reasoner
(reasoner emits reasoning tokens; still parse the final JSON from content). Retry 3× on 429/5xx/timeout.

## CLI
`node eval.mjs --model deepseek-chat --task all --lang all --concurrency 10`
--task EA|EU|all ; --lang en|zh|all ; write results/<model>-<task>.jsonl (per-item) +
results/<model>-<task>.summary.json (overall + per-category + per-lang + invalid count).
Print a final line: `EA en=.. zh=.. | EU en=.. zh=..` accuracies.

## Official comparators (for the report, already extracted from the paper — do not recompute)
EU Overall Base: GPT-4 en 59.75 / zh 54.12 ; GPT-3.5 en 33.12 / zh 26.38 ; Random 2.62
EA Overall Base: GPT-4 en 75.50 / zh 73.75 ; GPT-3.5 en 61.38 / zh 55.75 ; Random 24.12
