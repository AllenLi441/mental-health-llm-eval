import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const registry = JSON.parse(readFileSync(join(ROOT, "reports", "BASELINES.json"), "utf8"));
const byId = new Map(registry.baselines.map((row) => [row.id, Object.freeze({ ...row })]));

export function baseline(id) {
  const row = byId.get(id);
  if (!row) throw new Error(`unknown baseline id: ${id}`);
  return row;
}

export function baselineValue(id) {
  return baseline(id).value;
}

export function imhiBaseline(subtask) {
  const row = baseline("imhi-weighted-f1").subtasks?.[subtask];
  if (!row) throw new Error(`unknown IMHI baseline subtask: ${subtask}`);
  return row;
}

export function allBaselines() {
  return registry.baselines.map((row) => ({ ...row }));
}
