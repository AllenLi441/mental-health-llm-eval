#!/usr/bin/env python3
"""Deterministic safety/portability gate for the public benchmark harness."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")

lib = text("lib.mjs")
required = [
    "const apiKey = pick('EVAL_API_KEY')",
    "credentialScope: 'EVAL_API_KEY'",
    "resume file contains duplicate ids",
    "EVAL_IGNORE_DOTENV",
    "fatal HTTP ${res.status}; run aborted before recording this case",
    "requested_model: cfg.model",
    "prompt_profile: runArgs.promptProfile",
    "split: runArgs.split || null",
    "dataset_manifest_sha256",
    "prompt_template_sha256",
    "preregistration_sha256",
    "preregistration_commit",
    "test split requires explicit --confirm-test",
    "test preregistration is not committed to git",
    "resume metadata mismatch",
    "refusing to append a new run to existing result path",
    "usage: aggregateUsage(combinedResults)",
]
for marker in required:
    if marker not in lib:
        raise SystemExit(f"lib.mjs missing safety marker: {marker}")

code_files = [ROOT / "lib.mjs", ROOT / "contamination_probe.py", ROOT / "emobench-official" / "eval.mjs"]
for path in code_files:
    body = path.read_text(encoding="utf-8")
    for forbidden in ('process.env.DEEPSEEK_API_KEY', 'os.environ["DEEPSEEK_API_KEY"]'):
        if forbidden in body:
            raise SystemExit(f"{path.relative_to(ROOT)} reads production credential: {forbidden}")
if "EMOBENCH_DATA_DIR" not in text("emobench-official/eval.mjs"):
    raise SystemExit("official EmoBench harness has no portable data-root override")

runtime_files = [ROOT / "lib.mjs", ROOT / "run.mjs", *sorted((ROOT / "tasks").glob("*.mjs"))]
private_path_markers = ("/Users/", "/home/", "\\Users\\", "../静室")
for path in runtime_files:
    body = path.read_text(encoding="utf-8")
    if any(marker in body for marker in private_path_markers):
        raise SystemExit(f"{path.relative_to(ROOT)} contains a private app path")

readme = text("README.md")
if "多数类 72.4" in readme or "只统计本次新跑" in readme:
    raise SystemExit("README contains a stale baseline or resume claim")
if "EVAL_MODEL=deepseek-chat" in text(".env.example"):
    raise SystemExit(".env.example uses the deprecated deepseek-chat alias")
scoreboard = text("scripts/scoreboard.py")
if "赢过" in scoreboard or "同口径对照" in scoreboard:
    raise SystemExit("scoreboard contains an unsupported win/same-protocol claim")

emo_spec = text("emobench-official/SPEC.md")
if "faithful reproduction" in emo_spec or "Reproduce the OFFICIAL protocol exactly" in emo_spec:
    raise SystemExit("EmoBench deterministic proxy is mislabeled as an exact paper-protocol reproduction")
emo_paper = text("emobench-official/paper_protocol.mjs")
for marker in (
    "Closest executable EmoBench paper protocol",
    "5 stochastic samples",
    "PERMUTATIONS = 4",
    "REPEATS = 5",
    "temperature: 0.6",
    "implementation_assumptions",
    "--approved-budget-usd",
    "Dry-run only",
):
    if marker not in emo_paper:
        raise SystemExit(f"EmoBench paper protocol missing marker: {marker}")

psysuicide = text("tasks/psysuicide.mjs")
for marker in (
    "export const splits = ['train', 'valid', 'test']",
    "export const runSplits = ['valid', 'test']",
    "export const promptProfiles = ['baseline', 'taxonomy', 'hierarchical', 'taxonomy-v2', 'fewshot-balanced']",
    "fingerprint.holdout_excluded = true",
):
    if marker not in psysuicide:
        raise SystemExit(f"PsySUICIDE protocol missing marker: {marker}")

matrix = text("scripts/run_psysuicide_valid_matrix.mjs")
for marker in (
    "Dry-run by default",
    "--approved-budget-usd",
    "validateSmokeGate",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
):
    if marker not in matrix:
        raise SystemExit(f"PsySUICIDE valid matrix missing marker: {marker}")

analyzer = text("scripts/analyze_psysuicide_valid_matrix.py")
for marker in (
    "CONTRASTS =",
    "exact_mcnemar",
    "holm_adjust",
    "confirmatory_claim_allowed",
    "publishing_boundary",
):
    if marker not in analyzer:
        raise SystemExit(f"PsySUICIDE valid analyzer missing marker: {marker}")

test_pair = text("scripts/run_psysuicide_test_pair.mjs")
for marker in (
    "one-frozen-paired-confirmatory-test-campaign",
    "A_flash_baseline",
    "--confirm-test-pair",
    "committedArtifactRevision",
    "valid winner equals the current reference",
):
    if marker not in test_pair:
        raise SystemExit(f"PsySUICIDE test pair runner missing marker: {marker}")

test_analyzer = text("scripts/analyze_psysuicide_test_pair.py")
for marker in (
    "paired_randomization_macro_f1",
    "primary_metric",
    "CANDIDATE_SUPERIOR_ON_PREREGISTERED_MACRO_F1",
    "NO_DETECTED_PRIMARY_DIFFERENCE_NOT_A_TIE_OR_EQUIVALENCE",
    "equivalence_tested",
):
    if marker not in test_analyzer:
        raise SystemExit(f"PsySUICIDE confirmatory analyzer missing marker: {marker}")

partition = text("scripts/prepare_psysuicide_v2_partition.mjs")
for marker in (
    "cryptographic commitments",
    "never text, source ids, row-level labels, or membership",
    "refusing to overwrite frozen commitment artifact",
    "commit and push this artifact before inspecting optimization rows",
):
    if marker not in partition:
        raise SystemExit(f"PsySUICIDE v2 partition freezer missing marker: {marker}")

v2_matrix = text("scripts/run_psysuicide_v2_valid_matrix.mjs")
for marker in (
    "Dry-run by default",
    "--approved-budget-usd",
    "validateSmokeGate",
    "frozen train holdout excluded",
    "fewshot-balanced",
):
    if marker not in v2_matrix:
        raise SystemExit(f"PsySUICIDE v2 valid matrix missing marker: {marker}")

v2_analyzer = text("scripts/analyze_psysuicide_v2_valid.py")
for marker in (
    "paired_randomization_macro_f1",
    "bootstrap_metric_deltas",
    "REJECT_TAXONOMY_V2",
    "resampling parameters were not preregistered",
    "holdout_scored",
):
    if marker not in v2_analyzer:
        raise SystemExit(f"PsySUICIDE v2 valid analyzer missing marker: {marker}")

trainer = text("scripts/train_psysuicide_roberta.py")
for marker in (
    "hfl/chinese-roberta-wwm-ext-large",
    "full protocol requires exactly --seeds 42,43,44",
    "holdout rows not returned, tokenized, sampled, scored, or selected on",
    "WeightedRandomSampler",
    "class-weighted focal loss",
    "Dry-run only",
):
    if marker not in trainer:
        raise SystemExit(f"PsySUICIDE supervised trainer missing marker: {marker}")

reporter = text("scripts/report_psysuicide_roberta.py")
for marker in (
    "verify_execution_commit",
    "validate_class_weights",
    "seed accuracy recomputation",
    "POST_RUN_LOCAL_ARTIFACT_MATCH",
    "official-valid membership was not cryptographically committed",
):
    if marker not in reporter:
        raise SystemExit(f"PsySUICIDE supervised reporter missing marker: {marker}")

holdout_scorer = text("scripts/score_psysuicide_roberta_holdout.py")
for marker in (
    "Dry-run only. Holdout membership remains unopened.",
    "durable_exclusive_json",
    "\"ls-remote\"",
    "CLAIMED_BEFORE_HOLDOUT_LOAD",
    "global_exactly_once_proven",
    "descriptive only; no paired test was preregistered",
):
    if marker not in holdout_scorer:
        raise SystemExit(f"PsySUICIDE holdout scorer missing marker: {marker}")

holdout_freeze = text("reports/psysuicide-roberta-v1-holdout-freeze.json")
for marker in (
    "psysuicide-roberta-v1-holdout-once",
    "psysuicide-roberta-v1-valid-selection.json",
    "score_psysuicide_roberta_holdout.py",
    "\"row_level_material\": \"not persisted\"",
    "cannot prove no out-of-band data access",
):
    if marker not in holdout_freeze:
        raise SystemExit(f"PsySUICIDE holdout freeze missing marker: {marker}")

holdout_freeze_json = json.loads(holdout_freeze)
holdout_result = json.loads(
    text("reports/psysuicide-roberta-v1-holdout-result.json")
)
if holdout_result.get("status") != "COMPLETED_FROM_EXCLUSIVE_CLAIM":
    raise SystemExit("PsySUICIDE holdout result has no completed claim")
if holdout_result.get("protocol_id") != "psysuicide-roberta-v1-holdout-once":
    raise SystemExit("PsySUICIDE holdout result protocol mismatch")
if holdout_result.get("freeze_commit") != "a531c6c498602e258bf58c98ed71459a4d7b3757":
    raise SystemExit("PsySUICIDE holdout result freeze commit mismatch")
if holdout_result.get("artifacts") != holdout_freeze_json.get("artifacts"):
    raise SystemExit("PsySUICIDE holdout result artifact manifest mismatch")
claim_evidence = holdout_result.get("claim_evidence", {})
if (
    claim_evidence.get("completed_from_exclusive_claim") is not True
    or claim_evidence.get("global_exactly_once_proven") is not False
):
    raise SystemExit("PsySUICIDE holdout claim evidence is overstated or incomplete")
if holdout_result.get("configuration", {}).get("predict_call_count") != 1:
    raise SystemExit("PsySUICIDE holdout predict-call count mismatch")
per_class = holdout_result.get("metrics", {}).get("per_class", {})
expected_labels = holdout_freeze_json["inference"]["label_order"]
if list(per_class) != expected_labels:
    raise SystemExit("PsySUICIDE holdout label order mismatch")
support_total = sum(values["support"] for values in per_class.values())
if support_total != holdout_result["partition"]["holdout_rows"] or support_total != 2329:
    raise SystemExit("PsySUICIDE holdout support total mismatch")
class_f1 = []
weighted_f1 = 0.0
true_positives = 0.0
for label, values in per_class.items():
    support = values["support"]
    for metric in ("precision", "recall", "f1"):
        value = values[metric]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"PsySUICIDE holdout {metric} invalid: {label}")
    class_f1.append(values["f1"])
    weighted_f1 += values["f1"] * support
    true_positives += values["recall"] * support
metrics = holdout_result["metrics"]
checks = {
    "accuracy": true_positives / support_total,
    "macro_f1": sum(class_f1) / len(class_f1),
    "weighted_f1": weighted_f1 / support_total,
}
for metric, recomputed in checks.items():
    if not math.isclose(
        metrics[metric],
        recomputed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(f"PsySUICIDE holdout aggregate mismatch: {metric}")

paired_prereg_path = (
    ROOT / "reports" / "psysuicide-roberta-v1-official-test-paired.prereg.json"
)
paired_result = json.loads(
    text("reports/psysuicide-roberta-v1-official-test-paired-result.json")
)
paired_prereg = json.loads(paired_prereg_path.read_text(encoding="utf-8"))
if (
    paired_result.get("protocol_id")
    != "psysuicide-roberta-v1-official-test-paired-v1"
    or paired_result.get("cohort", {}).get("rows") != 1464
    or paired_result.get("cohort", {}).get("dataset_manifest_sha256")
    != paired_prereg["cohort"]["dataset_manifest_sha256"]
):
    raise SystemExit("PsySUICIDE same-test paired result identity mismatch")
paired_freeze = paired_result.get("preregistration", {})
if (
    paired_freeze.get("freeze_commit")
    != "6d597b35d16f9789274390202d79041cfb9a7936"
    or paired_freeze.get("live_remote_sha") != paired_freeze.get("freeze_commit")
):
    raise SystemExit("PsySUICIDE same-test freeze/remote identity mismatch")
if (
    paired_result.get("reference", {}).get("source_jsonl_sha256")
    != paired_prereg["reference"]["source_jsonl_sha256"]
    or paired_result.get("candidate", {}).get("checkpoint_weight_sha256")
    != paired_prereg["candidate"]["checkpoint_weight_sha256"]
):
    raise SystemExit("PsySUICIDE same-test model artifact identity mismatch")

paired_labels = paired_prereg["cohort"]["label_count"]
for arm_name in ("reference", "candidate"):
    arm = paired_result.get(arm_name, {})
    arm_per_class = arm.get("per_class", {})
    if list(arm_per_class) != expected_labels or len(arm_per_class) != paired_labels:
        raise SystemExit(f"PsySUICIDE same-test {arm_name} label order mismatch")
    arm_support = sum(values["support"] for values in arm_per_class.values())
    arm_tp = sum(values["tp"] for values in arm_per_class.values())
    arm_macro = sum(values["f1"] for values in arm_per_class.values()) / paired_labels
    arm_weighted = (
        sum(values["support"] * values["f1"] for values in arm_per_class.values())
        / arm_support
    )
    if arm_support != 1464:
        raise SystemExit(f"PsySUICIDE same-test {arm_name} support mismatch")
    for metric, recomputed in {
        "accuracy": arm_tp / arm_support,
        "macro_f1": arm_macro,
        "weighted_f1": arm_weighted,
    }.items():
        if not math.isclose(
            arm.get(metric, float("nan")),
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                f"PsySUICIDE same-test {arm_name} {metric} aggregate mismatch"
            )

paired_inference = paired_result.get("paired_inference", {})
paired_delta = paired_inference.get("delta_candidate_minus_reference", {})
for metric in ("accuracy", "macro_f1", "weighted_f1"):
    recomputed = (
        paired_result["candidate"][metric] - paired_result["reference"][metric]
    )
    if not math.isclose(
        paired_delta.get(metric, float("nan")),
        recomputed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(f"PsySUICIDE same-test paired delta mismatch: {metric}")

randomization = paired_inference.get("paired_randomization", {})
bootstrap = paired_inference.get("paired_bootstrap", {})
macro_delta = paired_delta["macro_f1"]
primary_superior = (
    macro_delta > 0
    and randomization.get("two_sided_p", 1.0) < 0.05
    and bootstrap.get("macro_f1_delta_ci95", [float("nan")])[0] > 0
)
primary_inferior = (
    macro_delta < 0
    and randomization.get("two_sided_p", 1.0) < 0.05
    and bootstrap.get("macro_f1_delta_ci95", [float("nan"), float("nan")])[1]
    < 0
)
expected_primary_claim = (
    "CANDIDATE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    if primary_superior
    else (
        "REFERENCE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
        if primary_inferior
        else "NO_DETECTED_PRIMARY_DIFFERENCE_NOT_A_TIE_OR_EQUIVALENCE"
    )
)
if (
    paired_inference.get("claim") != expected_primary_claim
    or paired_inference.get("equivalence_tested") is not False
):
    raise SystemExit("PsySUICIDE same-test primary claim mismatch")

mcnemar = paired_inference.get("accuracy_exact_mcnemar_secondary", {})
candidate_only = mcnemar.get("candidate_only_correct")
reference_only = mcnemar.get("reference_only_correct")
if not isinstance(candidate_only, int) or not isinstance(reference_only, int):
    raise SystemExit("PsySUICIDE same-test discordant counts are invalid")
discordant = candidate_only + reference_only
tail = sum(
    math.comb(discordant, index)
    for index in range(min(candidate_only, reference_only) + 1)
)
exact_p = min(1.0, 2.0 * tail / (2**discordant))
if not math.isclose(
    mcnemar.get("two_sided_p", float("nan")),
    exact_p,
    rel_tol=0.0,
    abs_tol=1e-15,
):
    raise SystemExit("PsySUICIDE same-test exact McNemar mismatch")

serialized_paired = json.dumps(paired_result, ensure_ascii=False)
for forbidden_key in ('"id":', '"text":', '"raw":', '"logits":'):
    if forbidden_key in serialized_paired:
        raise SystemExit(
            f"PsySUICIDE same-test public result contains row payload: {forbidden_key}"
        )

imhi_matrix = text("scripts/run_imhi_uniform_v4_matrix.mjs")
for marker in (
    "IMHI 9-task uniform v4 matrix",
    "no task-wise cherry-picking",
    "uniform-control",
    "contrastive-candidate",
    "unweighted mean of nine task weighted-F1 values",
    "--approved-budget-usd",
    "Dry-run only",
):
    if marker not in imhi_matrix:
        raise SystemExit(f"IMHI uniform v4 matrix missing marker: {marker}")

print("harness safety check PASS: dedicated credentials, split/profile provenance, frozen test, safe resume, portable runtime, honest claims")
