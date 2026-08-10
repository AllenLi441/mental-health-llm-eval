import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const source = new URL("../open_response_eval/preregistration.json", import.meta.url);
const target = new URL(
  "../open_response_eval/preregistration_provider_api_v2.json",
  import.meta.url,
);

const preregistration = JSON.parse(await readFile(source, "utf8"));
preregistration.protocol_id = "jingshi-cpcd-esconv-20260808-provider-api-v2";
preregistration.frozen_at = "2026-08-08T00:00:00-07:00";
preregistration.status = "FROZEN_PROVIDER_API_COMPARISON";
preregistration.research_question =
  "Compare the DeepSeek V4-Pro backbone used by Jingshi with the official provider-hosted Qwen3.6-27B API under identical frozen prompts and official CPCD/ESConv evaluation criteria.";

const qwen = preregistration.target_arms.find(
  (arm) => arm.id === "qwen_3_6_27b_base",
);
if (!qwen) throw new Error("Qwen evaluation arm is missing");

qwen.label = "Qwen3.6-27B official provider API";
qwen.model = "qwen3.6-27b";
delete qwen.revision;
delete qwen.canonical_repo_id;
delete qwen.deployment_manifest_path_env;
delete qwen.deployment_manifest_sha256;
qwen.thinking_mode = "disabled";
qwen.request_overrides = { enable_thinking: false };

const structuredEvalPromptPath =
  "open_response_eval/prompts/jingshi_core_en_structured_eval.txt";
const structuredEvalPrompt = await readFile(
  new URL(`../${structuredEvalPromptPath}`, import.meta.url),
);
preregistration.prompts.en = {
  path: structuredEvalPromptPath,
  sha256: createHash("sha256").update(structuredEvalPrompt).digest("hex"),
};

await writeFile(target, `${JSON.stringify(preregistration, null, 2)}\n`);
