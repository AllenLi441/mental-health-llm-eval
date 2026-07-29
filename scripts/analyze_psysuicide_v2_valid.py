#!/usr/bin/env python3
"""Aggregate-only paired analysis for the PsySUICIDE taxonomy-v2 valid run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from analyze_psysuicide_test_pair import paired_randomization_macro_f1
from analyze_psysuicide_valid_matrix import (
    LABELS,
    bootstrap_metric_deltas,
    exact_mcnemar,
    f1_metrics,
    sha256_file,
    top_confusions,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_N = 1459
REFERENCE_PROFILE = "taxonomy"
CANDIDATE_PROFILE = "taxonomy-v2"
MODEL = "deepseek-v4-pro"


def require_committed(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode:
        raise SystemExit(f"artifact is not committed: {relative}")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative.as_posix()],
        cwd=ROOT,
        check=False,
    )
    if unchanged.returncode:
        raise SystemExit(f"artifact differs from HEAD: {relative}")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_rows(path: Path, profile: str) -> tuple[dict[str, dict], dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "id", "gold", "predicted", "ok", "invalid", "error",
                "split", "prompt_profile", "requested_model", "response_model",
                "provider", "fingerprint", "run_id", "seed", "thinking",
                "reasoning_effort", "case_sha256", "prompt_template_sha256",
                "dataset_manifest_sha256", "usage",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise SystemExit(f"{path.name}:{line_number}: missing fields {missing}")
            expected = {
                "split": "valid",
                "prompt_profile": profile,
                "requested_model": MODEL,
                "response_model": MODEL,
                "provider": "deepseek",
                "seed": 42,
                "thinking": "enabled",
                "reasoning_effort": "high",
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise SystemExit(
                        f"{path.name}:{line_number}: {field} expected {value!r}, got {row[field]!r}"
                    )
            if row["error"] or not isinstance(row["usage"], dict):
                raise SystemExit(f"{path.name}:{line_number}: API error or missing usage")
            if row["gold"] not in LABELS:
                raise SystemExit(f"{path.name}:{line_number}: unknown gold")
            if row["predicted"] is not None and row["predicted"] not in LABELS:
                raise SystemExit(f"{path.name}:{line_number}: unknown prediction")
            if bool(row["ok"]) != (row["gold"] == row["predicted"]):
                raise SystemExit(f"{path.name}:{line_number}: ok flag mismatch")
            rows.append(row)
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"{path.name}: expected {EXPECTED_N} rows, got {len(rows)}")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{path.name}: duplicate ids")

    def unique(field):
        values = sorted({row[field] for row in rows})
        if len(values) != 1:
            raise SystemExit(f"{path.name}: mixed {field}")
        return values[0]

    return by_id, {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "model": MODEL,
        "prompt_profile": profile,
        "run_id": unique("run_id"),
        "fingerprints": sorted({row["fingerprint"] for row in rows}),
        "prompt_template_sha256": unique("prompt_template_sha256"),
        "dataset_manifest_sha256": unique("dataset_manifest_sha256"),
    }


def analyze(reference, candidate, reference_meta, candidate_meta, prereg_meta):
    if set(reference) != set(candidate):
        raise SystemExit("reference and candidate ID sets differ")
    ordered_ids = sorted(reference)
    for row_id in ordered_ids:
        if reference[row_id]["gold"] != candidate[row_id]["gold"]:
            raise SystemExit("paired gold mismatch")
        if reference[row_id]["case_sha256"] != candidate[row_id]["case_sha256"]:
            raise SystemExit("paired case commitment mismatch")
    if (
        reference_meta["dataset_manifest_sha256"]
        != candidate_meta["dataset_manifest_sha256"]
    ):
        raise SystemExit("dataset manifest mismatch")

    golds = [reference[row_id]["gold"] for row_id in ordered_ids]
    reference_predictions = [reference[row_id]["predicted"] for row_id in ordered_ids]
    candidate_predictions = [candidate[row_id]["predicted"] for row_id in ordered_ids]
    reference_metrics = f1_metrics(golds, reference_predictions)
    candidate_metrics = f1_metrics(golds, candidate_predictions)
    randomization = paired_randomization_macro_f1(
        golds, reference_predictions, candidate_predictions, 20000, 20260728
    )
    bootstrap = bootstrap_metric_deltas(
        golds, reference_predictions, candidate_predictions, 20000, 20260728
    )
    reference_ok = [
        gold == prediction for gold, prediction in zip(golds, reference_predictions)
    ]
    candidate_ok = [
        gold == prediction for gold, prediction in zip(golds, candidate_predictions)
    ]
    candidate_only = sum(
        candidate_value and not reference_value
        for reference_value, candidate_value in zip(reference_ok, candidate_ok)
    )
    reference_only = sum(
        reference_value and not candidate_value
        for reference_value, candidate_value in zip(reference_ok, candidate_ok)
    )
    per_class_delta = {
        label: (
            candidate_metrics["per_class"][label]["f1"]
            - reference_metrics["per_class"][label]["f1"]
        )
        for label in LABELS
    }
    macro_delta = candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
    candidate_passes = (
        macro_delta > 0
        and sum(bool(row["invalid"]) for row in candidate.values())
        <= sum(bool(row["invalid"]) for row in reference.values())
    )
    return {
        "schema_version": 1,
        "scope": "PsySUICIDE official valid full paired developmental comparison",
        "n": EXPECTED_N,
        "preregistration": prereg_meta,
        "reference": {
            **reference_meta,
            "accuracy": reference_metrics["accuracy"],
            "macro_f1": reference_metrics["macro_f1"],
            "weighted_f1": reference_metrics["weighted_f1"],
            "invalid": sum(bool(row["invalid"]) for row in reference.values()),
            "per_class": reference_metrics["per_class"],
            "top_confusions": top_confusions(golds, reference_predictions),
        },
        "candidate": {
            **candidate_meta,
            "accuracy": candidate_metrics["accuracy"],
            "macro_f1": candidate_metrics["macro_f1"],
            "weighted_f1": candidate_metrics["weighted_f1"],
            "invalid": sum(bool(row["invalid"]) for row in candidate.values()),
            "per_class": candidate_metrics["per_class"],
            "top_confusions": top_confusions(golds, candidate_predictions),
        },
        "paired_exploratory_analysis": {
            "delta_candidate_minus_reference": {
                "accuracy": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
                "macro_f1": macro_delta,
                "weighted_f1": candidate_metrics["weighted_f1"] - reference_metrics["weighted_f1"],
            },
            "per_class_f1_delta": per_class_delta,
            "paired_randomization_macro_f1": randomization,
            "paired_bootstrap": bootstrap,
            "accuracy_exact_mcnemar": {
                "candidate_only_correct": candidate_only,
                "reference_only_correct": reference_only,
                "two_sided_p": exact_mcnemar(candidate_only, reference_only),
            },
            "inference_status": "exploratory; resampling parameters were not preregistered",
        },
        "selection": {
            "candidate_passes_committed_rule": candidate_passes,
            "decision": "ADVANCE_TAXONOMY_V2" if candidate_passes else "REJECT_TAXONOMY_V2",
            "holdout_scored": False,
            "fewshot_full_run": "stopped after same-50 smoke for futility and cost",
        },
        "publishing_boundary": "aggregate valid analysis only; no counseling text, ids, per-case gold/prediction assignments, or raw output",
    }


def selftest():
    golds = LABELS * 5
    reference = golds.copy()
    candidate = golds.copy()
    candidate[0] = LABELS[1]
    result = paired_randomization_macro_f1(
        golds, reference, candidate, 200, 1
    )
    assert result["observed_delta"] < 0
    assert exact_mcnemar(0, 1) == 1.0
    print("PsySUICIDE v2 valid analyzer selftest PASS: paired rows, metrics, exploratory inference, rejection rule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=ROOT / "reports" / "psysuicide-v2-valid-full.prereg.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.reference or not args.candidate or not args.out:
        raise SystemExit("--reference, --candidate and --out are required")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite analysis: {args.out}")
    execution_commit = require_committed(args.prereg)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    reference, reference_meta = load_rows(args.reference, REFERENCE_PROFILE)
    candidate, candidate_meta = load_rows(args.candidate, CANDIDATE_PROFILE)
    if candidate_meta["prompt_template_sha256"] != prereg["candidate"]["prompt_template_sha256"]:
        raise SystemExit("candidate prompt hash differs from preregistration")
    if reference_meta["sha256"] != prereg["reference"]["result_sha256"]:
        raise SystemExit("reference result hash differs from preregistration")
    if candidate_meta["dataset_manifest_sha256"] != prereg["dataset_manifest_sha256"]:
        raise SystemExit("candidate dataset hash differs from preregistration")
    prereg_meta = {
        "file": args.prereg.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(args.prereg),
        "execution_commit": execution_commit,
    }
    result = analyze(reference, candidate, reference_meta, candidate_meta, prereg_meta)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["selection"]["decision"],
        "delta": result["paired_exploratory_analysis"]["delta_candidate_minus_reference"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
