#!/usr/bin/env python3
"""Deterministic safety/portability gate for the public benchmark harness."""
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

print("harness safety check PASS: dedicated credentials, split/profile provenance, frozen test, safe resume, portable runtime, honest claims")
