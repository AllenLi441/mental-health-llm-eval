#!/usr/bin/env python3
"""Confirmatory paired analysis for one frozen PsySUICIDE test campaign."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

from analyze_psysuicide_valid_matrix import (
    LABELS,
    bootstrap_metric_deltas,
    exact_mcnemar,
    f1_metrics,
    sha256_file,
    top_confusions,
    usage_cost,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_N = 1464


def require_committed(path: Path) -> str:
    path = path.resolve()
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"artifact must be inside repository: {path}") from exc
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path.name}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def result_path(results_dir: Path, campaign: dict, role: str) -> Path:
    arm = campaign[role]
    return (
        results_dir
        / f"psysuicide-test-{arm['profile']}-{arm['model']}-{arm['run_id']}.jsonl"
    )


def validate_campaign(path: Path) -> tuple[dict, dict]:
    execution_commit = require_committed(path)
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        campaign.get("schema_version") != 1
        or campaign.get("protocol")
        != "one-frozen-paired-confirmatory-test-campaign"
    ):
        raise SystemExit("invalid paired test campaign")
    if campaign["reference"]["arm"] != "A_flash_baseline":
        raise SystemExit("confirmatory reference must be A_flash_baseline")
    if campaign["candidate"]["arm"] == "A_flash_baseline":
        raise SystemExit("candidate cannot equal reference")
    if campaign["candidate"]["arm"] != campaign["valid_selection"]["winner"]:
        raise SystemExit("campaign candidate is not the frozen valid winner")
    inference = campaign.get("primary_inference") or {}
    expected = {
        "metric": "macro_f1",
        "alpha": 0.05,
        "paired_randomization_repetitions": 20000,
        "paired_randomization_seed": 20260727,
        "paired_bootstrap_repetitions": 20000,
        "paired_bootstrap_seed": 20260727,
    }
    for field, value in expected.items():
        if inference.get(field) != value:
            raise SystemExit(f"campaign primary inference mismatch: {field}")
    valid_path = ROOT / campaign["valid_selection"]["file"]
    require_committed(valid_path)
    if sha256_file(valid_path) != campaign["valid_selection"]["sha256"]:
        raise SystemExit("valid selection hash mismatch")
    valid_analysis = json.loads(valid_path.read_text(encoding="utf-8"))
    if (
        valid_analysis.get("scope")
        != "PsySUICIDE official valid split; developmental selection only"
        or valid_analysis.get("n") != 1459
        or valid_analysis.get("selection", {}).get("winner")
        != campaign["candidate"]["arm"]
    ):
        raise SystemExit("campaign candidate does not match committed valid analysis")
    for role in ("reference", "candidate"):
        prereg = ROOT / campaign[role]["preregistration"]
        require_committed(prereg)
        if sha256_file(prereg) != campaign[role]["preregistration_sha256"]:
            raise SystemExit(f"{role} preregistration hash mismatch")
    return campaign, {
        "file": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "execution_commit": execution_commit,
    }


def validate_arm(path: Path, campaign: dict, role: str) -> tuple[dict[str, dict], dict]:
    arm = campaign[role]
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"{path.name}: expected {EXPECTED_N} rows, found {len(rows)}")
    required = {
        "id",
        "gold",
        "predicted",
        "ok",
        "invalid",
        "error",
        "split",
        "prompt_profile",
        "requested_model",
        "response_model",
        "provider",
        "fingerprint",
        "run_id",
        "seed",
        "thinking",
        "reasoning_effort",
        "case_sha256",
        "prompt_template_sha256",
        "dataset_manifest_sha256",
        "preregistration_sha256",
        "preregistration_commit",
        "usage",
    }
    by_id = {}
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise SystemExit(f"{path.name}:{index}: missing fields {missing}")
        row_id = str(row["id"])
        if row_id in by_id:
            raise SystemExit(f"{path.name}:{index}: duplicate normalized id")
        if row["gold"] not in LABELS:
            raise SystemExit(f"{path.name}:{index}: unknown gold")
        if row["predicted"] is not None and row["predicted"] not in LABELS:
            raise SystemExit(f"{path.name}:{index}: unknown prediction")
        if row["error"]:
            raise SystemExit(f"{path.name}:{index}: API error prevents confirmation")
        if not isinstance(row["usage"], dict):
            raise SystemExit(f"{path.name}:{index}: missing usage")
        expected = {
            "split": "test",
            "prompt_profile": arm["profile"],
            "requested_model": arm["model"],
            "response_model": arm["model"],
            "provider": "deepseek",
            "run_id": arm["run_id"],
            "seed": 42,
            "thinking": "enabled",
            "reasoning_effort": "high",
            "preregistration_sha256": arm["preregistration_sha256"],
        }
        for field, value in expected.items():
            if row[field] != value:
                raise SystemExit(
                    f"{path.name}:{index}: {field} expected {value!r}, got {row[field]!r}"
                )
        if (
            not row["fingerprint"]
            or not isinstance(row["preregistration_commit"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", row["preregistration_commit"])
        ):
            raise SystemExit(f"{path.name}:{index}: missing fingerprint or prereg commit")
        if bool(row["ok"]) != (row["predicted"] == row["gold"]):
            raise SystemExit(f"{path.name}:{index}: ok flag mismatch")
        by_id[row_id] = row
    unique = lambda field: sorted({row[field] for row in rows})
    for field in (
        "prompt_template_sha256",
        "dataset_manifest_sha256",
        "preregistration_sha256",
        "preregistration_commit",
    ):
        if len(unique(field)) != 1:
            raise SystemExit(f"{path.name}: mixed {field}")
    return by_id, {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "model": arm["model"],
        "prompt_profile": arm["profile"],
        "fingerprints": sorted({row["fingerprint"] for row in rows}),
        "prompt_template_sha256": unique("prompt_template_sha256")[0],
        "dataset_manifest_sha256": unique("dataset_manifest_sha256")[0],
        "preregistration_sha256": unique("preregistration_sha256")[0],
        "preregistration_commit": unique("preregistration_commit")[0],
    }


def macro_f1_delta(
    gold_indices: list[int],
    reference_indices: list[int],
    candidate_indices: list[int],
) -> float:
    support = [0] * len(LABELS)
    reference_predicted = [0] * len(LABELS)
    candidate_predicted = [0] * len(LABELS)
    reference_tp = [0] * len(LABELS)
    candidate_tp = [0] * len(LABELS)
    for gold, reference, candidate in zip(
        gold_indices, reference_indices, candidate_indices
    ):
        support[gold] += 1
        if reference >= 0:
            reference_predicted[reference] += 1
            if reference == gold:
                reference_tp[gold] += 1
        if candidate >= 0:
            candidate_predicted[candidate] += 1
            if candidate == gold:
                candidate_tp[gold] += 1

    def macro(predicted: list[int], true_positive: list[int]) -> float:
        total = 0.0
        for label in range(len(LABELS)):
            denominator = support[label] + predicted[label]
            total += (
                (2 * true_positive[label] / denominator) if denominator else 0.0
            ) / len(LABELS)
        return total

    return macro(candidate_predicted, candidate_tp) - macro(
        reference_predicted, reference_tp
    )


def paired_randomization_macro_f1(
    golds: list[str],
    reference: list[str | None],
    candidate: list[str | None],
    repetitions: int,
    seed: int,
) -> dict:
    label_index = {label: index for index, label in enumerate(LABELS)}
    gold_indices = [label_index[gold] for gold in golds]
    reference_indices = [
        label_index[prediction] if prediction in label_index else -1
        for prediction in reference
    ]
    candidate_indices = [
        label_index[prediction] if prediction in label_index else -1
        for prediction in candidate
    ]
    observed = macro_f1_delta(
        gold_indices, reference_indices, candidate_indices
    )
    rng = random.Random(seed)
    extreme = 0
    for _ in range(repetitions):
        permuted_reference = []
        permuted_candidate = []
        for reference_value, candidate_value in zip(
            reference_indices, candidate_indices
        ):
            if rng.getrandbits(1):
                permuted_reference.append(candidate_value)
                permuted_candidate.append(reference_value)
            else:
                permuted_reference.append(reference_value)
                permuted_candidate.append(candidate_value)
        delta = macro_f1_delta(
            gold_indices, permuted_reference, permuted_candidate
        )
        extreme += abs(delta) >= abs(observed) - 1e-15
    return {
        "observed_delta": observed,
        "two_sided_p": (extreme + 1) / (repetitions + 1),
        "repetitions": repetitions,
        "seed": seed,
    }


def analyze_pair(
    campaign: dict,
    campaign_meta: dict,
    reference: dict[str, dict],
    candidate: dict[str, dict],
    reference_meta: dict,
    candidate_meta: dict,
    pricing: dict,
) -> dict:
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
    inference = campaign["primary_inference"]
    randomization = paired_randomization_macro_f1(
        golds,
        reference_predictions,
        candidate_predictions,
        inference["paired_randomization_repetitions"],
        inference["paired_randomization_seed"],
    )
    bootstrap = bootstrap_metric_deltas(
        golds,
        reference_predictions,
        candidate_predictions,
        inference["paired_bootstrap_repetitions"],
        inference["paired_bootstrap_seed"],
    )
    reference_ok = [
        gold == prediction
        for gold, prediction in zip(golds, reference_predictions)
    ]
    candidate_ok = [
        gold == prediction
        for gold, prediction in zip(golds, candidate_predictions)
    ]
    candidate_only = sum(
        candidate_value and not reference_value
        for reference_value, candidate_value in zip(reference_ok, candidate_ok)
    )
    reference_only = sum(
        reference_value and not candidate_value
        for reference_value, candidate_value in zip(reference_ok, candidate_ok)
    )
    macro_delta = (
        candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
    )
    primary_superior = (
        macro_delta > 0
        and randomization["two_sided_p"] < inference["alpha"]
        and bootstrap["macro_f1_delta_ci95"][0] > 0
    )
    primary_inferior = (
        macro_delta < 0
        and randomization["two_sided_p"] < inference["alpha"]
        and bootstrap["macro_f1_delta_ci95"][1] < 0
    )
    if primary_superior:
        claim = "CANDIDATE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    elif primary_inferior:
        claim = "REFERENCE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    else:
        claim = "NO_DETECTED_PRIMARY_DIFFERENCE_NOT_A_TIE_OR_EQUIVALENCE"
    reference_rows = list(reference.values())
    candidate_rows = list(candidate.values())
    return {
        "schema_version": 1,
        "scope": "one frozen paired PsySUICIDE official test campaign",
        "campaign": campaign_meta,
        "n": EXPECTED_N,
        "reference": {
            **reference_meta,
            "accuracy": reference_metrics["accuracy"],
            "macro_f1": reference_metrics["macro_f1"],
            "weighted_f1": reference_metrics["weighted_f1"],
            "invalid": sum(bool(row["invalid"]) for row in reference_rows),
            "per_class": reference_metrics["per_class"],
            "top_confusions": top_confusions(golds, reference_predictions),
            "usage_and_cost": usage_cost(
                reference_rows, pricing, campaign["reference"]["model"]
            ),
        },
        "candidate": {
            **candidate_meta,
            "accuracy": candidate_metrics["accuracy"],
            "macro_f1": candidate_metrics["macro_f1"],
            "weighted_f1": candidate_metrics["weighted_f1"],
            "invalid": sum(bool(row["invalid"]) for row in candidate_rows),
            "per_class": candidate_metrics["per_class"],
            "top_confusions": top_confusions(golds, candidate_predictions),
            "usage_and_cost": usage_cost(
                candidate_rows, pricing, campaign["candidate"]["model"]
            ),
        },
        "paired_inference": {
            "primary_metric": "macro_f1",
            "delta_candidate_minus_reference": {
                "macro_f1": macro_delta,
                "weighted_f1": candidate_metrics["weighted_f1"]
                - reference_metrics["weighted_f1"],
                "accuracy": candidate_metrics["accuracy"]
                - reference_metrics["accuracy"],
            },
            "paired_randomization": randomization,
            "paired_bootstrap": bootstrap,
            "accuracy_exact_mcnemar_secondary": {
                "candidate_only_correct": candidate_only,
                "reference_only_correct": reference_only,
                "two_sided_p": exact_mcnemar(candidate_only, reference_only),
            },
            "claim": claim,
            "equivalence_tested": False,
        },
        "pricing_snapshot": pricing,
        "publishing_boundary": "aggregate confirmation only; no counseling text, ids, per-case gold/prediction assignments, or raw output",
    }


def selftest() -> None:
    golds = LABELS * 4
    reference = golds.copy()
    candidate = golds.copy()
    candidate[0] = LABELS[1]
    result = paired_randomization_macro_f1(
        golds, reference, candidate, repetitions=200, seed=1
    )
    assert result["observed_delta"] < 0
    assert 0 < result["two_sided_p"] <= 1
    assert exact_mcnemar(3, 0) == 0.25
    campaign = {
        "primary_inference": {
            "metric": "macro_f1",
            "alpha": 0.05,
            "paired_randomization_repetitions": 200,
            "paired_randomization_seed": 1,
            "paired_bootstrap_repetitions": 200,
            "paired_bootstrap_seed": 1,
        },
        "reference": {"model": "deepseek-v4-flash"},
        "candidate": {"model": "deepseek-v4-pro"},
    }
    reference_rows = {}
    candidate_rows = {}
    for index in range(EXPECTED_N):
        gold = LABELS[index % len(LABELS)]
        reference_prediction = (
            LABELS[(index + 1) % len(LABELS)] if index % 4 == 0 else gold
        )
        candidate_prediction = (
            LABELS[(index + 1) % len(LABELS)] if index % 7 == 0 else gold
        )
        common = {
            "id": f"fixture:{index}",
            "gold": gold,
            "invalid": False,
            "error": None,
            "case_sha256": f"{index:064x}",
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens": 2,
            },
        }
        reference_rows[common["id"]] = {
            **common,
            "predicted": reference_prediction,
            "ok": reference_prediction == gold,
        }
        candidate_rows[common["id"]] = {
            **common,
            "predicted": candidate_prediction,
            "ok": candidate_prediction == gold,
        }
    pricing = {
        "models": {
            "deepseek-v4-flash": {
                "input_cache_hit": 0.0028,
                "input_cache_miss": 0.14,
                "output": 0.28,
            },
            "deepseek-v4-pro": {
                "input_cache_hit": 0.003625,
                "input_cache_miss": 0.435,
                "output": 0.87,
            },
        }
    }
    paired = analyze_pair(
        campaign,
        {"file": "fixture", "sha256": "f" * 64, "execution_commit": "e" * 40},
        reference_rows,
        candidate_rows,
        {"dataset_manifest_sha256": "d" * 64, "rows": EXPECTED_N},
        {"dataset_manifest_sha256": "d" * 64, "rows": EXPECTED_N},
        pricing,
    )
    assert paired["paired_inference"]["claim"] == (
        "CANDIDATE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    )
    print(
        "PsySUICIDE confirmatory test analyzer selftest PASS: full synthetic pair, randomization, bootstrap, McNemar, frozen claim rule"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--pricing",
        type=Path,
        default=ROOT / "lib" / "deepseek_v4_pricing_2026-07-27.json",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.campaign or not args.out:
        raise SystemExit("--campaign and --out are required")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite analysis: {args.out}")
    campaign, campaign_meta = validate_campaign(args.campaign.resolve())
    reference, reference_meta = validate_arm(
        result_path(args.results_dir.resolve(), campaign, "reference"),
        campaign,
        "reference",
    )
    candidate, candidate_meta = validate_arm(
        result_path(args.results_dir.resolve(), campaign, "candidate"),
        campaign,
        "candidate",
    )
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    result = analyze_pair(
        campaign,
        campaign_meta,
        reference,
        candidate,
        reference_meta,
        candidate_meta,
        pricing,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "claim": result["paired_inference"]["claim"],
                "macro_f1_delta": result["paired_inference"][
                    "delta_candidate_minus_reference"
                ]["macro_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
